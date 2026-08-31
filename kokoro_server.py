#!/usr/bin/env python3
"""Kokoro TTS + Vision OCR HTTP server for the Chrome extension.

POST /tts       { "text": "..." }                        -> audio/wav
POST /ocr_tts   { "image": "<base64_png>" }              -> NDJSON text/audio stream
POST /panel-heartbeat  (empty body)                      -> liveness ping
GET  /health     -> status
"""

import argparse
import base64
import io
import ipaddress
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# --- Boot timing ---
# The native host redirects stderr to logs/server_spawner.log, so these lines
# carry wall-clock timestamps for the import phase that runs before
# setup_logging() configures the file handler. Manual runs see them on the TTY.
_BOOT_START = time.monotonic()

def _boot(msg: str) -> None:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} [BOOT] {msg}', file=sys.stderr, flush=True)

@contextmanager
def _phase(name: str):
    """Log how long a startup phase took, in ms.

    A failing phase logs 'failed after Nms' instead, so a crashed startup
    never contributes a success-shaped 'took Nms' sample to the baseline.
    """
    t0 = time.monotonic()
    try:
        yield
    except BaseException:
        ms = int((time.monotonic() - t0) * 1000)
        logging.info(f"[PHASE] {name} failed after {ms}ms")
        raise
    else:
        ms = int((time.monotonic() - t0) * 1000)
        logging.info(f"[PHASE] {name} took {ms}ms")

_boot('stdlib imports done')

SAMPLE_RATE = 24000
DEFAULT_VOICE = 'af_bella'
MODEL_REPO_ID = 'hexgrad/Kokoro-82M'
# Weights filename inside models--hexgrad--Kokoro-82M snapshots; verified
# against the live local snapshot listing (config.json + kokoro-v1_0.pth +
# voices/af_bella.pt, refs/main → f3ff3571...). Not derived from
# KModel.MODEL_NAMES, which requires the torch import this gate must precede.
MODEL_WEIGHTS_FILE = 'kokoro-v1_0.pth'

# --- Logging ---
LOGS_DIR = Path(__file__).parent / 'logs'

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / 'server.log'
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.root.addHandler(handler)
    # Echo to the console only when attached to a TTY; the native host redirects
    # stdout to a log file, which would otherwise duplicate every line.
    if sys.stdout and sys.stdout.isatty():
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.root.addHandler(stream_handler)
    logging.root.setLevel(logging.INFO)

# --- Conversation storage ---
CONVERSATIONS_DIR = Path(__file__).parent / 'conversations'
CURRENT_PTR = CONVERSATIONS_DIR / 'current.txt'
conv_lock = threading.Lock()

# --- Globals ---
pipeline = None
pipeline_lock = threading.Lock()
MODEL_LOADED = False
# Tri-state readiness of the Ollama (vision) backend, published on /health so
# the panel can wait out a cold start instead of burning the OCR retry budget.
OLLAMA_STATE = 'starting'
VISION_MODEL = os.environ.get('VISION_MODEL', 'gemma4:31b:cloud')
VISION_API_BASE = os.environ.get('VISION_API_BASE', 'http://127.0.0.1:11434')
VISION_API_KEY = os.environ.get('VISION_API_KEY', 'ollama')
OLLAMA_STARTUP_TIMEOUT = float(os.environ.get('OLLAMA_STARTUP_TIMEOUT', '15'))
# After a startup-timeout 'unavailable', a bounded background probe keeps
# watching the spawned `ollama serve` so a slow-but-successful bring-up can
# still upgrade the state to 'ready'.
OLLAMA_REPROBE_TIMEOUT = float(os.environ.get('OLLAMA_REPROBE_TIMEOUT', '60'))
TTS_WARMUP = os.environ.get('TTS_WARMUP', '1') != '0'
HF_OFFLINE_IF_CACHED = os.environ.get('HF_OFFLINE_IF_CACHED', '1') != '0'

# --- Managed lifetime (auto-stop) ---
# Servers spawned by the extension run with --managed: the side panel POSTs
# /panel-heartbeat every 15s, and if no heartbeat (or other in-flight activity)
# arrives within HEARTBEAT_GRACE seconds the server shuts itself down. A
# manually started server (no --managed) never self-stops.
MANAGED = False
HEARTBEAT_GRACE = float(os.environ.get('HEARTBEAT_GRACE', '90'))
_heartbeat_at = time.monotonic()
_heartbeat_lock = threading.Lock()


def touch_heartbeat() -> None:
    """Record liveness. In-flight requests call this too: the single-threaded
    server queues /panel-heartbeat behind long streams, so activity must refresh
    the watchdog directly instead of waiting for heartbeat requests to drain."""
    global _heartbeat_at
    with _heartbeat_lock:
        _heartbeat_at = time.monotonic()


def seconds_since_heartbeat() -> float:
    with _heartbeat_lock:
        return time.monotonic() - _heartbeat_at


def _watchdog_tick(server) -> bool:
    """One watchdog check. Returns True if the server was told to shut down."""
    idle = seconds_since_heartbeat()
    if idle > HEARTBEAT_GRACE:
        logging.info(f"[SERVER] Auto-stopping (no panel heartbeat for {idle:.0f}s)")
        server.shutdown()
        return True
    return False


def maybe_start_watchdog(server) -> None:
    """Watchdog only makes sense in extension-spawned (--managed) mode."""
    if MANAGED:
        def watchdog():
            while True:
                time.sleep(5)
                if _watchdog_tick(server):
                    return

        threading.Thread(target=watchdog, name='panel-watchdog', daemon=True).start()

OCR_SYSTEM_PROMPT_PLAIN = """Transcribe all readable text from this screenshot. Output ONLY the transcribed text, nothing else. Do not describe the image. Output plain text only: no markdown, no asterisks, no bullet markers, no emojis, no emphasis. Transcribe lists as plain lines."""
OCR_SYSTEM_PROMPT_CONSTRAINED = """You are an OCR assistant. Read the text in the image and follow the user's instructions exactly. Do not describe the image. Output plain text only: no markdown, no asterisks, no bullet markers, no emojis, no emphasis. Transcribe lists as plain lines."""


def new_conversation() -> str:
    """Create a new conversation, set it as current, and return its ID."""
    from datetime import datetime
    conv_id = datetime.now().strftime('conv_%Y%m%d_%H%M%S')
    conv_dir = CONVERSATIONS_DIR / conv_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / 'images').mkdir(exist_ok=True)
    conv_data = {
        'id': conv_id,
        'created': datetime.now().isoformat(),
        'turns': []
    }
    (conv_dir / 'conv.json').write_text(json.dumps(conv_data, indent=2), encoding='utf-8')
    with conv_lock:
        CURRENT_PTR.write_text(conv_id, encoding='utf-8')
    return conv_id


def ensure_conversation_dir() -> str:
    """Ensure conversations directory and current pointer exist; return current conversation_id."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    conv_id = None
    if CURRENT_PTR.exists():
        candidate = CURRENT_PTR.read_text().strip()
        conv_dir = CONVERSATIONS_DIR / candidate
        if conv_dir.is_dir() and (conv_dir / 'conv.json').exists():
            conv_id = candidate
    if conv_id is None:
        conv_id = new_conversation()
    return conv_id


def load_conversation(conv_id: str) -> dict:
    """Load conversation JSON from disk."""
    conv_path = CONVERSATIONS_DIR / conv_id / 'conv.json'
    return json.loads(conv_path.read_text(encoding='utf-8'))


def save_conversation(conv_id: str, data: dict) -> None:
    """Write conversation JSON to disk."""
    conv_path = CONVERSATIONS_DIR / conv_id / 'conv.json'
    conv_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def save_turn(conv_id: str, image_bytes: bytes, prompt: str, ocr_text: str) -> None:
    """Append a turn to the conversation and save the image to disk."""
    with conv_lock:
        conv_data = load_conversation(conv_id)
        turn_num = len(conv_data['turns']) // 2 + 1
        image_filename = f'{turn_num:03d}.png'
        image_dir = CONVERSATIONS_DIR / conv_id / 'images'
        image_dir.mkdir(exist_ok=True)
        image_path = image_dir / image_filename
        image_path.write_bytes(image_bytes)

        relative_image_path = f'{conv_id}/images/{image_filename}'

        conv_data['turns'].append({
            'role': 'user',
            'image_path': relative_image_path,
            'prompt': prompt
        })
        conv_data['turns'].append({
            'role': 'assistant',
            'text': ocr_text
        })
        save_conversation(conv_id, conv_data)


def _hf_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache directory exactly like
    huggingface_hub.constants, so the cache-warm check below agrees with where
    the import actually looks. Each level expands vars/user as the real code
    does, including the legacy HUGGINGFACE_HUB_CACHE name."""
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.path.expandvars(os.path.expanduser(os.environ['HF_HUB_CACHE'])))
    if os.environ.get('HUGGINGFACE_HUB_CACHE'):
        return Path(os.path.expandvars(os.path.expanduser(os.environ['HUGGINGFACE_HUB_CACHE'])))
    if os.environ.get('HF_HOME'):
        hf_home = Path(os.path.expandvars(os.path.expanduser(os.environ['HF_HOME'])))
    else:
        xdg = os.environ.get('XDG_CACHE_HOME')
        if xdg:
            hf_home = Path(os.path.expandvars(os.path.expanduser(xdg))) / 'huggingface'
        else:
            hf_home = Path(os.path.expanduser('~/.cache')) / 'huggingface'
    return hf_home / 'hub'


def _kokoro_model_cached() -> bool:
    """True when the Kokoro weights AND the pinned voice are already in the
    local HF cache, so no network round-trip is needed and HF_HUB_OFFLINE can
    be set safely. The voice file (voices/<name>.pt) is downloaded lazily at
    first synthesis by kokoro (.venv/Lib/site-packages/kokoro/pipeline.py:142,
    hf_hub_download), so a weights-only cache with HF_HUB_OFFLINE=1 would fail
    every /tts forever.

    Mirrors huggingface_hub's cache resolution (try_to_load_from_cache,
    huggingface_hub/file_download.py:1482-1573): refs/main → commit →
    snapshots/<commit>/ — NOT any complete-looking snapshots/* dir. A ref
    still pointing at a partially-downloaded snapshot (written before the
    blobs land) must keep the gate OPEN, or the lazy download can never run.
    Fail-open (False) on any missing/unreadable/empty ref so the speedup
    disarms silently rather than blocking the download. Kept hand-rolled
    instead of importing huggingface_hub here: its constants snapshot
    HF_HUB_OFFLINE at import time, and this runs before we arm it
    (get_pipeline imports kokoro only after this gate)."""
    try:
        slug = f'models--{MODEL_REPO_ID.replace("/", "--")}'
        root = _hf_cache_dir() / slug
        # Read verbatim, no strip: hub reads refs unstripped
        # (file_download.py:1557), so a newline/CRLF-padded ref resolves to a
        # nonexistent snapshot there — stripping here would arm offline mode
        # hub can't satisfy, making every /tts fail with no recovery.
        commit = (root / 'refs' / 'main').read_text(encoding='utf-8')
        snapshot = root / 'snapshots' / commit
        return (
            (snapshot / 'config.json').is_file()
            and (snapshot / MODEL_WEIGHTS_FILE).is_file()
            and (snapshot / 'voices' / f'{DEFAULT_VOICE}.pt').is_file()
        )
    except (OSError, ValueError):  # ValueError: undecodable ref file
        return False


def get_pipeline():
    global pipeline, MODEL_LOADED
    with pipeline_lock:
        if pipeline is None:
            with _phase('heavy imports'):
                # HF_HUB_OFFLINE must be set before the kokoro import:
                # huggingface_hub reads it at import time. The check stays
                # outside the device loop — an import crash is a real error,
                # not a CUDA failure to retry on cpu.
                if HF_OFFLINE_IF_CACHED and _kokoro_model_cached():
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    logging.info("[SERVER] HF cache warm; HF_HUB_OFFLINE=1")
                # No separate numpy/soundfile probes needed: kokoro's own
                # import chain pulls both in, so a broken dependency fails the
                # kokoro import right here; text_to_wav keeps its own imports
                # for name binding.
                from kokoro import KPipeline
            for device in ('cuda', 'cpu'):
                try:
                    logging.info(f"[SERVER] Loading Kokoro model on {device}...")
                    t0 = time.monotonic()
                    pipeline = KPipeline(lang_code='a', repo_id=MODEL_REPO_ID, device=device)
                    model_load_ms = int((time.monotonic() - t0) * 1000)
                    # Warmup lives here, under pipeline_lock, so it cannot
                    # race a real first request into a half-initialized pipe.
                    # Direct synthesis, not text_to_wav: that would re-enter
                    # get_pipeline and deadlock on the non-reentrant lock.
                    # MODEL_LOADED is set only after the warmup validates the
                    # model: a warmup failure is a load failure (panel stays
                    # offline) — reporting healthy while every /tts would 500
                    # is the worse failure mode.
                    if TTS_WARMUP:
                        with _phase('TTS warmup'):
                            for _ in pipeline('Hello.', voice=DEFAULT_VOICE):
                                pass
                    # The 'loaded' line emits only after the warmup validated
                    # the device, so a warmup failure never reads as a success.
                    # The ms number still covers construction only.
                    logging.info(f"[SERVER] Kokoro model loaded on {device} in {model_load_ms}ms.")
                    MODEL_LOADED = True
                    return pipeline
                except Exception as e:
                    # A failure after KPipeline() constructed (the warmup) must
                    # not leave a half-valid global for a later request to skip
                    # validation on.
                    pipeline = None
                    if device == 'cuda':
                        logging.warning(f"[SERVER] CUDA failed ({e}); falling back to CPU...")
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        raise
        return pipeline


def reset_pipeline():
    """Reset the Kokoro pipeline, clearing GPU memory. Next call to get_pipeline() will reload."""
    global pipeline, MODEL_LOADED
    with pipeline_lock:
        if pipeline is not None:
            logging.warning("[SERVER] Resetting Kokoro pipeline due to CUDA error...")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            pipeline = None
            MODEL_LOADED = False
            logging.info("[SERVER] Pipeline reset. Will reload on next request.")


def is_cuda_error(exc: BaseException) -> bool:
    """Check if an exception is CUDA-related (driver crash, OOM, device error)."""
    exc_name = type(exc).__name__
    msg = str(exc).lower()
    if 'cuda' in exc_name.lower() or 'cuda' in msg:
        return True
    if 'accelerator' in exc_name.lower():
        return True
    # torch exceptions
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return False


OCR_MAX_RETRIES = 3
OCR_TIMEOUT = 35  # per attempt — 3 retries × 25s

def _api_base_parts(api_base: str):
    from urllib.parse import urlsplit

    parsed = urlsplit(api_base)
    scheme = parsed.scheme or 'http'
    hostname = parsed.hostname or '127.0.0.1'
    port = parsed.port or (443 if scheme == 'https' else 80)
    return scheme, hostname, port


def _format_api_base(scheme: str, host: str, port: int) -> str:
    try:
        ipaddress.ip_address(host)
        if ':' in host and not host.startswith('['):
            host = f'[{host}]'
    except ValueError:
        pass
    return f'{scheme}://{host}:{port}'


def _read_linux_default_gateway() -> str | None:
    try:
        with open('/proc/net/route') as f:
            next(f, None)
            for line in f:
                fields = line.strip().split()
                if len(fields) < 3 or fields[1] != '00000000':
                    continue
                raw = bytes.fromhex(fields[2])
                return str(ipaddress.IPv4Address(raw[::-1]))
    except (OSError, ValueError):
        pass
    return None


def get_vision_api_bases() -> list[str]:
    """Return the mirrored and non-mirrored WSL Ollama API bases."""
    scheme, hostname, port = _api_base_parts(VISION_API_BASE)
    candidates = [_format_api_base(scheme, '127.0.0.1', port)]

    gateway = _read_linux_default_gateway()
    if gateway:
        candidates.append(_format_api_base(scheme, gateway, port))
    elif hostname not in ('127.0.0.1', 'localhost', '::1'):
        candidates.append(_format_api_base(scheme, hostname, port))

    # Keep exactly two addresses when a second WSL route exists, while avoiding
    # duplicates if WSL reports loopback as its default route.
    seen = set()
    unique = []
    for candidate in candidates:
        candidate = candidate.rstrip('/')
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _ollama_is_running(api_base: str, timeout: float = 1.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f'{api_base}/api/version', timeout=timeout):
            return True
    except (TimeoutError, urllib.error.URLError, OSError):
        return False


def ensure_ollama_running() -> None:
    """Start a local Ollama server if the configured API routes are offline."""
    global OLLAMA_STATE
    api_bases = get_vision_api_bases()
    if any(_ollama_is_running(api_base) for api_base in api_bases):
        logging.info(f"[OLLAMA] Already running at one of: {', '.join(api_bases)}")
        OLLAMA_STATE = 'ready'
        return

    ollama_path = shutil.which('ollama')
    if not ollama_path:
        logging.warning("[OLLAMA] Command not found; OCR will require Ollama to be started manually.")
        OLLAMA_STATE = 'unavailable'
        return

    log_path = Path(tempfile.gettempdir()) / 'kokoro_ollama.log'
    log_file = open(log_path, 'ab')
    popen_kwargs = {
        'stdout': log_file,
        'stderr': subprocess.STDOUT,
    }
    if os.name == 'nt':
        popen_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    else:
        popen_kwargs['start_new_session'] = True

    logging.info(f"[OLLAMA] Starting: {ollama_path} serve")
    try:
        process = subprocess.Popen([ollama_path, 'serve'], **popen_kwargs)
    except OSError as e:
        log_file.close()
        logging.error(f"[OLLAMA] Failed to start Ollama: {e}")
        OLLAMA_STATE = 'unavailable'
        return

    deadline = time.time() + OLLAMA_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            logging.error(f"[OLLAMA] Ollama exited early with code {process.returncode}; see {log_path}")
            log_file.close()
            OLLAMA_STATE = 'unavailable'
            return
        if any(_ollama_is_running(api_base) for api_base in api_bases):
            logging.info(f"[OLLAMA] Ready at one of: {', '.join(api_bases)}")
            log_file.close()
            OLLAMA_STATE = 'ready'
            return
        time.sleep(0.5)

    logging.warning(f"[OLLAMA] Started but did not respond within {OLLAMA_STARTUP_TIMEOUT:.0f}s; see {log_path}")
    log_file.close()
    OLLAMA_STATE = 'unavailable'
    # A slow start is not a failed start: the spawned `serve` keeps running,
    # so a bounded background probe can still upgrade the state.
    threading.Thread(target=_ollama_late_probe, args=(api_bases,),
                     name='ollama-late-probe', daemon=True).start()


def _ollama_late_probe(api_bases: list) -> None:
    """Bounded post-timeout watcher for 'unavailable' -> 'ready' upgrades only.

    Runs on its own daemon thread so the `ollama ensure` phase still ends on
    time. State transitions elsewhere are monotonic; this is the sole writer
    allowed to move the state back up, and only from 'unavailable'.
    """
    global OLLAMA_STATE
    try:
        deadline = time.time() + OLLAMA_REPROBE_TIMEOUT
        while time.time() < deadline:
            if OLLAMA_STATE != 'unavailable':
                return
            if any(_ollama_is_running(api_base) for api_base in api_bases):
                if OLLAMA_STATE == 'unavailable':
                    logging.info("[OLLAMA] Late probe: ready at one of: "
                                 + ', '.join(api_bases))
                    OLLAMA_STATE = 'ready'
                return
            time.sleep(0.5)
        logging.warning(f"[OLLAMA] Late probe: still not responding after "
                        f"{OLLAMA_REPROBE_TIMEOUT:.0f}s; giving up")
    except Exception as e:
        logging.error(f"[OLLAMA] Late probe failed ({e})", exc_info=True)


def ocr_image_stream(image_bytes: bytes, constraints: str = '', history_turns: list | None = None):
    """Yield OCR text fragments from the Ollama native streaming API."""
    import urllib.error
    import urllib.request

    b64 = base64.b64encode(image_bytes).decode('utf-8')

    user_text = "Transcribe all readable text from this image verbatim. Do not add any description or commentary."
    if constraints:
        user_text = constraints
    system_prompt = OCR_SYSTEM_PROMPT_CONSTRAINED if constraints else OCR_SYSTEM_PROMPT_PLAIN

    messages = [{"role": "system", "content": system_prompt}]

    if history_turns:
        for turn in history_turns:
            if turn['role'] == 'user':
                img_path = CONVERSATIONS_DIR / turn['image_path']
                img_b64 = base64.b64encode(img_path.read_bytes()).decode('utf-8')
                messages.append({
                    "role": "user",
                    "content": turn['prompt'],
                    "images": [img_b64]
                })
            elif turn['role'] == 'assistant':
                messages.append({"role": "assistant", "content": turn['text']})

    messages.append({
        "role": "user",
        "content": f"{user_text}\n\n[image attached]",
        "images": [b64]
    })

    payload = json.dumps({
        "model": VISION_MODEL,
        "messages": messages,
        "think": False,
        "stream": True,
        "keep_alive": 0
    }).encode('utf-8')

    api_bases = get_vision_api_bases()
    logging.info(f"[OCR] Streaming {len(image_bytes)} bytes to {VISION_MODEL} via {api_bases[0]}...")

    last_err = None
    for attempt in range(1, OCR_MAX_RETRIES + 1):
        for api_base in api_bases:
            url = f"{api_base}/api/chat"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={'Content-Type': 'application/json'}
            )
            t0 = time.time()
            chars = 0
            ttft_logged = False
            try:
                with urllib.request.urlopen(req, timeout=OCR_TIMEOUT) as resp:
                    for raw_line in resp:
                        if not raw_line.strip():
                            continue
                        event = json.loads(raw_line.decode('utf-8'))
                        if event.get('error'):
                            raise RuntimeError(event['error'])
                        fragment = event.get('message', {}).get('content', '')
                        if fragment:
                            if not ttft_logged:
                                ttft_ms = (time.time() - t0) * 1000
                                logging.info(f"[OCR] TTFT: {ttft_ms:.0f}ms")
                                ttft_logged = True
                            chars += len(fragment)
                            yield fragment
                        if event.get('done'):
                            elapsed = time.time() - t0
                            tps = chars / elapsed if elapsed > 0 else 0
                            logging.info(f"[OCR] Streamed {chars} chars in {elapsed:.1f}s from {api_base} (attempt {attempt}), TPS: {tps:.1f} chars/s")
                            return

            except (TimeoutError, urllib.error.URLError, OSError) as e:
                elapsed = time.time() - t0
                last_err = e
                logging.warning(f"[OCR] Attempt {attempt}/{OCR_MAX_RETRIES} failed for {api_base} after {elapsed:.1f}s: {type(e).__name__}: {e}")
            except Exception as e:
                elapsed = time.time() - t0
                last_err = e
                logging.error(f"[OCR] Stream failed for {api_base} after {elapsed:.1f}s: {type(e).__name__}: {e}")
        if attempt < OCR_MAX_RETRIES:
            time.sleep(1)

    raise RuntimeError(f"OCR failed after {OCR_MAX_RETRIES} attempts: {last_err}")


# --- Speech-friendly output cleanup ---
# The vision model decorates transcriptions with markdown and emoji regardless
# of the system prompt, and Kokoro reads markup tokens aloud ("star star").
# Code content is deliberately untouched — users who need verbatim symbols pass
# their own constraints; the sanitizer removes formatting, never content.
# Decision: docs/adr/0003-sanitize-ocr-output-before-tts.md
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # emoji and pictographs
    (0x2190, 0x21FF),    # arrows
    (0x2600, 0x27BF),    # misc symbols and dingbats
    (0x2B00, 0x2BFF),    # misc symbols and arrows
    (0xFE00, 0xFE0F),    # variation selectors (emoji presentation)
)
_ZWJ = chr(0x200D)  # joins emoji into sequences
_EMOJI_RE = re.compile(
    '[' + ''.join(f'{chr(lo)}-{chr(hi)}' for lo, hi in _EMOJI_RANGES) + _ZWJ + ']+')

# (pattern, replacement) applied in order; paired markup first so emphasis
# content survives, then structural markers.
_MARKUP_RULES = [
    (re.compile(r'\*\*\*(.+?)\*\*\*'), r'\1'),
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
    (re.compile(r'(?<![\w*])\*([^*\n]+?)\*(?![\w*])'), r'\1'),
    (re.compile(r'~~(.+?)~~'), r'\1'),
    (re.compile(r'!?\[([^\]]*)\]\([^)]*\)'), r'\1'),  # links/images -> alt text
    (re.compile(r'`+'), ''),
    (re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+', re.MULTILINE), ''),  # ATX headers
    (re.compile(r'^\s*(?:[-*_][ \t]*){3,}\s*$', re.MULTILINE), ''),  # hr rules
    # Bullets only at line start: an em dash or ">" mid-sentence is prose
    # ("Paris — the capital", "10 > 5"), not a list marker.
    (re.compile(r'^[ \t]*(?:[-*' + chr(0x2022) + chr(0x00B7) + chr(0x2023) + chr(0x25AA) + r'][ \t]+)', re.MULTILINE), ''),
    (re.compile(r'^[ \t]*\[[ xX]\][ \t]+', re.MULTILINE), ''),  # checkboxes
]


def sanitize_for_speech(text: str) -> str:
    """Strip markdown markup and emoji so transcriptions speak cleanly.

    Paired markup removed anywhere; line-start bullets removed (the \n segment
    split already gives the pause); emojis dropped silently. Orphan markers
    from a markup pair split across segments are trimmed at segment edges.
    """
    for pattern, replacement in _MARKUP_RULES:
        text = pattern.sub(replacement, text)
    text = _EMOJI_RE.sub('', text)
    # A markup pair split across a \n segment boundary leaves stray '*'s.
    text = re.sub(r'^[ \t*]+|[ \t*]+$', '', text, flags=re.MULTILINE)
    return re.sub(r'\n\s*\n', '\n', re.sub(r'[\t ]{2,}', ' ', text)).strip()


def pop_tts_segment(buffer: str, force: bool = False) -> tuple[str | None, str]:
    """Return a speakable prefix, keeping incomplete trailing text buffered."""
    if not buffer.strip():
        return None, ''

    max_chars = 700
    split_at = -1
    for idx, ch in enumerate(buffer):
        next_char = buffer[idx + 1] if idx + 1 < len(buffer) else ''
        if ch == '\n' or (ch in '.!?' and next_char.isspace()):
            split_at = idx + 1

    if split_at < 1:
        if not force and len(buffer) < max_chars:
            return None, buffer
        split_at = min(len(buffer), max_chars)

    segment = buffer[:split_at].strip()
    remainder = buffer[split_at:]
    return (segment or None), remainder


def text_to_wav(text: str, _retry: bool = True) -> bytes:
    """Generate TTS audio with the pinned voice and return WAV bytes.
    Retries once on CUDA errors."""
    import numpy as np
    import soundfile as sf

    pipe = get_pipeline()
    if pipe is None:  # unreachable: get_pipeline raises on load failure
        raise RuntimeError("Pipeline failed to load")
    try:
        all_audio = []
        for gs, ps, audio in pipe(text, voice=DEFAULT_VOICE):
            all_audio.append(audio)

        if not all_audio:
            raise RuntimeError("No audio generated")

        combined = np.concatenate(all_audio)
        buf = io.BytesIO()
        sf.write(buf, combined, SAMPLE_RATE, format='WAV')
        return buf.getvalue()
    except Exception as e:
        if _retry and is_cuda_error(e):
            logging.warning(f"[TTS] CUDA error, resetting pipeline and retrying: {e}")
            reset_pipeline()
            return text_to_wav(text, _retry=False)
        raise


class TTSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_json({'status': 'ok', 'model': 'kokoro-82M', 'vision': VISION_MODEL,
                            'streaming': True, 'model_loaded': MODEL_LOADED, 'managed': MANAGED,
                            'ollama': OLLAMA_STATE})
        elif self.path == '/conversation_state':
            self.handle_conversation_state()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/new_conversation':
            self.handle_new_conversation()
            return

        if self.path == '/panel-heartbeat':
            touch_heartbeat()
            self.send_json({'ok': True})
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return

        if self.path == '/tts':
            self.handle_tts(data)
        elif self.path == '/ocr_tts':
            self.handle_ocr_tts(data)
        else:
            self.send_error(404)

    def handle_tts(self, data):
        text = data.get('text', '').strip()
        if not text:
            self.send_error(400, 'No text provided')
            return
        if len(text) > 10000:
            text = text[:10000]

        try:
            t0 = time.time()
            wav_bytes = text_to_wav(text)
            elapsed = time.time() - t0
            tps = len(text) / elapsed if elapsed > 0 else 0
            logging.info(f"[TTS] {elapsed:.1f}s for {len(text)} chars, TPS: {tps:.1f} chars/s")
            touch_heartbeat()  # in-flight request; see touch_heartbeat
            self.send_wav(wav_bytes)
        except Exception as e:
            logging.error(f"[TTS ERROR] {e}")
            self.send_error(500, str(e))

    def handle_ocr_tts(self, data):
        """Screenshot -> streaming OCR -> segmented TTS pipeline."""
        image_b64 = data.get('image', '')
        constraints = data.get('constraints', '').strip()
        history_enabled = data.get('history', False)
        conversation_id = data.get('conversation_id', '')
        if not image_b64:
            self.send_error(400, 'No image provided')
            return

        try:
            image_bytes = base64.b64decode(image_b64)
            logging.info(f"[OCR_TTS] Received {len(image_bytes)} byte image")
            if constraints:
                logging.info(f"[OCR_TTS] Constraints: {constraints[:200]}")

            # Build user prompt text for history
            user_prompt = constraints if constraints else "Transcribe all readable text from this image verbatim. Do not add any description or commentary."

            # Auto-assign conversation when history is enabled but no ID provided
            if history_enabled and not conversation_id:
                conversation_id = ensure_conversation_dir()
                logging.info(f"[OCR_TTS] Auto-assigned conversation: {conversation_id}")

            # Load history turns if enabled
            history_turns = None
            if history_enabled and conversation_id:
                try:
                    conv_data = load_conversation(conversation_id)
                    history_turns = conv_data.get('turns', [])
                except (FileNotFoundError, json.JSONDecodeError):
                    history_turns = None

            t0 = time.time()
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.streaming_response_started = True

            buffer = ''
            full_text = []
            audio_chunks = 0

            for fragment in ocr_image_stream(image_bytes, constraints=constraints, history_turns=history_turns):
                touch_heartbeat()  # in-flight request; see touch_heartbeat
                full_text.append(fragment)
                buffer += fragment
                while True:
                    segment, buffer = pop_tts_segment(buffer)
                    if not segment:
                        break
                    segment = sanitize_for_speech(segment)
                    if not segment:
                        continue
                    self.send_stream_event({'type': 'text', 'text': segment})
                    try:
                        tts_start = time.time()
                        wav_bytes = text_to_wav(segment)
                        audio_chunks += 1
                        audio_b64 = base64.b64encode(wav_bytes).decode('ascii')
                        self.send_stream_event({'type': 'audio', 'text': segment, 'audio': audio_b64})
                        logging.info(f"[OCR_TTS] Chunk {audio_chunks}: {len(segment)} chars -> TTS in {time.time() - tts_start:.1f}s")
                    except Exception as tts_err:
                        logging.warning(f"[OCR_TTS] TTS chunk failed: {tts_err}")
                        if is_cuda_error(tts_err):
                            self.send_stream_event({'type': 'error', 'error': f'GPU error during audio generation: {tts_err}. Try again.'})
                            return
                        # Non-CUDA TTS errors: skip audio for this chunk but continue OCR
                        self.send_stream_event({'type': 'text', 'text': f'[Audio generation failed: {tts_err}]'})

            segment, buffer = pop_tts_segment(buffer, force=True)
            if segment:
                segment = sanitize_for_speech(segment)
            if segment:
                self.send_stream_event({'type': 'text', 'text': segment})
                try:
                    tts_start = time.time()
                    wav_bytes = text_to_wav(segment)
                    audio_chunks += 1
                    audio_b64 = base64.b64encode(wav_bytes).decode('ascii')
                    self.send_stream_event({'type': 'audio', 'text': segment, 'audio': audio_b64})
                    logging.info(f"[OCR_TTS] Chunk {audio_chunks}: {len(segment)} chars -> TTS in {time.time() - tts_start:.1f}s")
                except Exception as tts_err:
                    logging.warning(f"[OCR_TTS] Final TTS chunk failed: {tts_err}")
                    if is_cuda_error(tts_err):
                        self.send_stream_event({'type': 'error', 'error': f'GPU error during audio generation: {tts_err}. Try again.'})
                        return

            text = sanitize_for_speech(''.join(full_text))
            if not text:
                self.send_stream_event({'type': 'error', 'error': 'No text found in image'})
                return

            # Save turn to conversation history if enabled
            if history_enabled and conversation_id:
                try:
                    save_turn(conversation_id, image_bytes, user_prompt, text)
                except Exception as e:
                    logging.warning(f"[OCR_TTS] Warning: failed to save turn: {e}")

            total = time.time() - t0
            logging.info(f"[OCR_TTS] Total: {total:.1f}s, {len(text)} chars, {audio_chunks} audio chunks")
            done_event = {'type': 'done', 'text': text}
            if history_enabled and conversation_id:
                done_event['conversation_id'] = conversation_id
            self.send_stream_event(done_event)

        except Exception as e:
            logging.error(f"[OCR_TTS ERROR] {e}", exc_info=True)
            if getattr(self, 'streaming_response_started', False):
                self.send_stream_event({'type': 'error', 'error': str(e)})
            else:
                self.send_error(500, str(e))

    def handle_conversation_state(self):
        """Return current conversation ID and turn count."""
        try:
            conv_id = ensure_conversation_dir()
            conv_data = load_conversation(conv_id)
            self.send_json({
                'conversation_id': conv_id,
                'turn_count': len(conv_data['turns']) // 2
            })
        except Exception as e:
            self.send_error(500, str(e))

    def handle_new_conversation(self):
        """Create a new conversation and set it as current."""
        try:
            conv_id = new_conversation()
            self.send_json({'conversation_id': conv_id})
        except Exception as e:
            self.send_error(500, str(e))

    def send_stream_event(self, obj):
        touch_heartbeat()  # a stream still sending events is alive; see touch_heartbeat
        body = (json.dumps(obj) + '\n').encode('utf-8')
        self.wfile.write(body)
        self.wfile.flush()

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_wav(self, wav_bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(wav_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(wav_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass


def load_model():
    try:
        get_pipeline()  # pre-load model; sets MODEL_LOADED on success
    except Exception as e:
        # exc_info keeps the loader-thread traceback in logs/server.log; without
        # it a broken dependency fails with a one-line message and no cause.
        logging.error(f"[SERVER] Model load failed ({e}); cannot load until the cause is fixed — see traceback; panel stays offline.",
                      exc_info=True)


def ensure_ollama():
    try:
        with _phase('ollama ensure'):
            ensure_ollama_running()
    except Exception as e:
        # A crash before ensure_ollama_running's first state write (unreadable
        # VISION_API_BASE, unwritable log) must still land a terminal state,
        # or /health reports 'starting' forever and every panel click waits
        # out its full bounded OCR wait.
        global OLLAMA_STATE
        OLLAMA_STATE = 'unavailable'
        logging.error(f"[OLLAMA] ensure failed ({e})", exc_info=True)


def start_background_tasks():
    # Ollama bring-up and the Kokoro model load are independent (the model
    # loads from the HF cache, not from Ollama), so they race instead of
    # serializing behind each other.
    threading.Thread(target=load_model, name='model-loader', daemon=True).start()
    threading.Thread(target=ensure_ollama, name='ollama-ensure', daemon=True).start()


def main():
    global MANAGED
    parser = argparse.ArgumentParser(description='Kokoro TTS + Vision OCR server')
    parser.add_argument('--port', type=int, default=5912)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--managed', action='store_true',
                        help='auto-stop when panel heartbeats stop (extension-spawned mode)')
    args = parser.parse_args()
    MANAGED = args.managed

    setup_logging()
    logging.info(f"[SERVER] Stdlib imports took {time.monotonic() - _BOOT_START:.2f}s")
    logging.info(f"[SERVER] Starting on {args.host}:{args.port} (managed={MANAGED})")
    ensure_conversation_dir()

    # Allow large payloads (screenshots can be ~5MB base64)
    # Override both server and handler limits
    import http.server
    http.server.BaseHTTPRequestHandler.max_request_line = 10 * 1024 * 1024  # 10MB

    # Bind before loading the model so /health answers (reporting model_loaded:
    # false) while the ~10s load runs — the panel can tell "starting" from
    # "crashed". allow_reuse_address defaults to True, but on Windows
    # SO_REUSEADDR lets two processes silently share a port, so rebind races
    # must fail loudly with WSAEADDRINUSE instead.
    server = HTTPServer((args.host, args.port), TTSHandler, bind_and_activate=False)
    server.allow_reuse_address = False
    try:
        with _phase('socket bind'):
            server.server_bind()
            server.server_activate()
    except OSError as e:
        logging.error(f"[SERVER] Could not bind {args.host}:{args.port} — is another kokoro_server already running? ({e})")
        raise SystemExit(1)

    touch_heartbeat()  # start the grace period now; the panel takes over once online
    maybe_start_watchdog(server)

    start_background_tasks()

    # Anchored at _BOOT_START so 'startup' spans the import phase too — the
    # dominant cost (see issues #6-#10) — not just post-argparse bring-up.
    logging.info(f"[SERVER] Ready at http://{args.host}:{args.port} (startup {time.monotonic() - _BOOT_START:.2f}s)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("[SERVER] Shutting down.")
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
