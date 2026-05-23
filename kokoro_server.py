#!/usr/bin/env python3
"""Kokoro TTS + Vision OCR HTTP server for the Chrome extension.

POST /tts       { "text": "...", "voice": "af_bella" }  -> audio/wav
POST /ocr_tts   { "image": "<base64_png>" }              -> audio/wav (screenshot -> OCR -> TTS)
GET  /voices     -> list of voices
GET  /health     -> status
"""

import argparse
import base64
import ipaddress
import io
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

os.environ.setdefault('TTS_SKIP_WARM', '1')

import yaml
import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEFAULT_VOICE = 'af_bella'

# --- Globals ---
pipeline = None
pipeline_lock = threading.Lock()
VISION_MODEL = os.environ.get('VISION_MODEL', 'gemma4:31b:cloud')
VISION_API_BASE = os.environ.get('VISION_API_BASE', 'http://127.0.0.1:11434')
VISION_API_KEY = os.environ.get('VISION_API_KEY', 'ollama')
OLLAMA_STARTUP_TIMEOUT = float(os.environ.get('OLLAMA_STARTUP_TIMEOUT', '15'))

OCR_SYSTEM_PROMPT_PLAIN = """Transcribe all readable text from this screenshot. Output ONLY the transcribed text, nothing else. Do not describe the image."""
OCR_SYSTEM_PROMPT_CONSTRAINED = """You are an OCR assistant. Read the text in the image and follow the user's instructions exactly. Do not describe the image."""

# --- Config-driven text filter ---
CONFIG_PATH = Path(__file__).parent / 'config.yaml'
_skip_patterns: list[re.Pattern] = []
_min_line_length: int = 4

def load_config():
    global _skip_patterns, _min_line_length
    _skip_patterns = []
    _min_line_length = 4
    if not CONFIG_PATH.exists():
        print(f"[CONFIG] No config file at {CONFIG_PATH}, using defaults", flush=True)
        return
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        for p in cfg.get('skip_patterns', []):
            _skip_patterns.append(re.compile(p, re.IGNORECASE))
        _min_line_length = cfg.get('min_line_length', 4)
        print(f"[CONFIG] Loaded {len(_skip_patterns)} skip patterns, min_line_length={_min_line_length}", flush=True)
    except Exception as e:
        print(f"[CONFIG] Error loading {CONFIG_PATH}: {e}", flush=True)

load_config()

def filter_text(text: str) -> str:
    """Remove lines matching skip_patterns and short noise lines."""
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines matching any pattern
        if any(p.search(stripped) for p in _skip_patterns):
            continue
        # Skip very short lines (noise) unless they end with punctuation
        if len(stripped) < _min_line_length and not stripped[-1] in '.!?':
            continue
        kept.append(stripped)
    return '\n'.join(kept)


def get_pipeline():
    global pipeline
    with pipeline_lock:
        if pipeline is None:
            print("[SERVER] Loading Kokoro model...", flush=True)
            pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cuda')
            print("[SERVER] Kokoro model loaded.", flush=True)
        return pipeline


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
    api_bases = get_vision_api_bases()
    if any(_ollama_is_running(api_base) for api_base in api_bases):
        print(f"[OLLAMA] Already running at one of: {', '.join(api_bases)}", flush=True)
        return

    ollama_path = shutil.which('ollama')
    if not ollama_path:
        print("[OLLAMA] Command not found; OCR will require Ollama to be started manually.", flush=True)
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

    print(f"[OLLAMA] Starting: {ollama_path} serve", flush=True)
    try:
        process = subprocess.Popen([ollama_path, 'serve'], **popen_kwargs)
    except OSError as e:
        log_file.close()
        print(f"[OLLAMA] Failed to start Ollama: {e}", flush=True)
        return

    deadline = time.time() + OLLAMA_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"[OLLAMA] Ollama exited early with code {process.returncode}; see {log_path}", flush=True)
            log_file.close()
            return
        if any(_ollama_is_running(api_base) for api_base in api_bases):
            print(f"[OLLAMA] Ready at one of: {', '.join(api_bases)}", flush=True)
            log_file.close()
            return
        time.sleep(0.5)

    print(f"[OLLAMA] Started but did not respond within {OLLAMA_STARTUP_TIMEOUT:.0f}s; see {log_path}", flush=True)
    log_file.close()


def ocr_image(image_bytes: bytes, constraints: str = '') -> str:
    """Run vision OCR on a screenshot using the Ollama native API, with retries."""
    import urllib.request
    import urllib.error

    b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Build the user prompt: base instruction + optional constraints
    user_text = "Transcribe all readable text from this image verbatim. Do not add any description or commentary."
    if constraints:
        user_text = constraints
    system_prompt = OCR_SYSTEM_PROMPT_CONSTRAINED if constraints else OCR_SYSTEM_PROMPT_PLAIN

    # Use Ollama native /api/chat endpoint — /v1/chat/completions
    # incorrectly handles "think": false (causes thinking instead of disabling it)
    payload = json.dumps({
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_text}\n\n[image attached]", "images": [b64]}
        ],
        "think": False,
        "stream": False,
        "keep_alive": 0
    }).encode('utf-8')

    api_bases = get_vision_api_bases()

    print(f"[OCR] Sending {len(image_bytes)} bytes to {VISION_MODEL} via {api_bases[0]}...", flush=True)

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
            try:
                with urllib.request.urlopen(req, timeout=OCR_TIMEOUT) as resp:
                    result = json.loads(resp.read().decode('utf-8'))

                text = result.get('message', {}).get('content', '').strip()
                elapsed = time.time() - t0
                print(f"[OCR] Got {len(text)} chars in {elapsed:.1f}s from {api_base} (attempt {attempt})", flush=True)
                return text

            except (TimeoutError, urllib.error.URLError, OSError) as e:
                elapsed = time.time() - t0
                last_err = e
                print(f"[OCR] Attempt {attempt}/{OCR_MAX_RETRIES} failed for {api_base} after {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
        if attempt < OCR_MAX_RETRIES:
            time.sleep(1)  # brief pause before retry

    raise RuntimeError(f"OCR failed after {OCR_MAX_RETRIES} attempts: {last_err}")


def text_to_wav(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate TTS audio and return WAV bytes."""
    pipe = get_pipeline()
    all_audio = []
    for gs, ps, audio in pipe(text, voice=voice):
        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError("No audio generated")

    combined = np.concatenate(all_audio)
    buf = io.BytesIO()
    sf.write(buf, combined, SAMPLE_RATE, format='WAV')
    return buf.getvalue()


class TTSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_json({'status': 'ok', 'model': 'kokoro-82M', 'vision': VISION_MODEL,
                            'skip_patterns': len(_skip_patterns)})
        elif self.path == '/voices':
            self.send_json({'voices': [
                'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
                'am_adam', 'am_michael',
                'bf_emma', 'bf_isabella',
                'bm_george', 'bm_lewis',
            ]})
        elif self.path == '/config':
            self.send_json({
                'skip_patterns': [p.pattern for p in _skip_patterns],
                'min_line_length': _min_line_length,
                'config_path': str(CONFIG_PATH),
            })
        else:
            self.send_error(404)

    def do_POST(self):
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
        elif self.path == '/config/reload':
            load_config()
            self.send_json({'status': 'ok', 'skip_patterns': len(_skip_patterns),
                            'min_line_length': _min_line_length})
        else:
            self.send_error(404)

    def handle_tts(self, data):
        text = data.get('text', '').strip()
        voice = data.get('voice', DEFAULT_VOICE)
        if not text:
            self.send_error(400, 'No text provided')
            return
        if len(text) > 10000:
            text = text[:10000]

        try:
            t0 = time.time()
            wav_bytes = text_to_wav(text, voice)
            elapsed = time.time() - t0
            duration = len(wav_bytes) / (SAMPLE_RATE * 2)  # rough estimate
            print(f"[TTS] {elapsed:.1f}s for {len(text)} chars", flush=True)
            self.send_wav(wav_bytes)
        except Exception as e:
            print(f"[TTS ERROR] {e}", flush=True)
            self.send_error(500, str(e))

    def handle_ocr_tts(self, data):
        """Screenshot -> OCR -> TTS pipeline."""
        image_b64 = data.get('image', '')
        voice = data.get('voice', DEFAULT_VOICE)
        constraints = data.get('constraints', '').strip()
        if not image_b64:
            self.send_error(400, 'No image provided')
            return

        try:
            image_bytes = base64.b64decode(image_b64)
            print(f"[OCR_TTS] Received {len(image_bytes)} byte image", flush=True)
            if constraints:
                print(f"[OCR_TTS] Constraints: {constraints[:200]}", flush=True)

            # OCR
            t0 = time.time()
            text = ocr_image(image_bytes, constraints=constraints)
            ocr_time = time.time() - t0

            if not text.strip():
                self.send_error(400, 'No text found in image')
                return

            # Apply text filter (remove copyright, page numbers, etc.)
            original_len = len(text)
            text = filter_text(text)
            removed = original_len - len(text)
            if removed > 0:
                print(f"[FILTER] Removed {removed} chars from OCR text", flush=True)

            if not text.strip():
                self.send_error(400, 'No text remaining after filtering')
                return

            print(f"[OCR_TTS] OCR: {len(text)} chars in {ocr_time:.1f}s", flush=True)

            # TTS
            t1 = time.time()
            wav_bytes = text_to_wav(text, voice)
            tts_time = time.time() - t1

            total = time.time() - t0
            print(f"[OCR_TTS] Total: {total:.1f}s (OCR {ocr_time:.1f}s + TTS {tts_time:.1f}s)", flush=True)

            # Send JSON response: { "text": "ocr text", "audio": "<base64 wav>" }
            import base64 as b64
            audio_b64 = b64.b64encode(wav_bytes).decode('ascii')
            resp = json.dumps({'text': text, 'audio': audio_b64}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(resp)

        except Exception as e:
            print(f"[OCR_TTS ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

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


def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS + Vision OCR server')
    parser.add_argument('--port', type=int, default=5912)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    print(f"[SERVER] Starting on {args.host}:{args.port}", flush=True)
    ensure_ollama_running()
    get_pipeline()  # pre-load model

    # Allow large payloads (screenshots can be ~5MB base64)
    # Override both server and handler limits
    import http.server
    http.server.BaseHTTPRequestHandler.max_request_line = 10 * 1024 * 1024  # 10MB
    
    server = HTTPServer((args.host, args.port), TTSHandler)
    server.allow_reuse_address = True
    print(f"[SERVER] Ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down.", flush=True)
        server.server_close()


if __name__ == '__main__':
    main()
