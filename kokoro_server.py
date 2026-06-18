#!/usr/bin/env python3
"""Kokoro TTS + Vision OCR HTTP server for the Chrome extension.

POST /tts       { "text": "...", "voice": "af_bella" }  -> audio/wav
POST /ocr_tts   { "image": "<base64_png>" }              -> NDJSON text/audio stream
GET  /voices     -> list of voices
GET  /health     -> status
"""

import argparse
import base64
import ipaddress
import io
import json
import os
import shutil
import sys
import subprocess
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

os.environ.setdefault('TTS_SKIP_WARM', '1')

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEFAULT_VOICE = 'af_bella'

# --- Conversation storage ---
CONVERSATIONS_DIR = Path(__file__).parent / 'conversations'
CURRENT_PTR = CONVERSATIONS_DIR / 'current.txt'
conv_lock = threading.Lock()

# --- Globals ---
pipeline = None
pipeline_lock = threading.Lock()
VISION_MODEL = os.environ.get('VISION_MODEL', 'gemma4:31b:cloud')
VISION_API_BASE = os.environ.get('VISION_API_BASE', 'http://127.0.0.1:11434')
VISION_API_KEY = os.environ.get('VISION_API_KEY', 'ollama')
OLLAMA_STARTUP_TIMEOUT = float(os.environ.get('OLLAMA_STARTUP_TIMEOUT', '15'))

OCR_SYSTEM_PROMPT_PLAIN = """Transcribe all readable text from this screenshot. Output ONLY the transcribed text, nothing else. Do not describe the image."""
OCR_SYSTEM_PROMPT_CONSTRAINED = """You are an OCR assistant. Read the text in the image and follow the user's instructions exactly. Do not describe the image."""


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


def ocr_image_stream(image_bytes: bytes, constraints: str = '', history_turns: list | None = None):
    """Yield OCR text fragments from the Ollama native streaming API."""
    import urllib.request
    import urllib.error

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
    print(f"[OCR] Streaming {len(image_bytes)} bytes to {VISION_MODEL} via {api_bases[0]}...", flush=True)

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
                            chars += len(fragment)
                            yield fragment
                        if event.get('done'):
                            elapsed = time.time() - t0
                            print(f"[OCR] Streamed {chars} chars in {elapsed:.1f}s from {api_base} (attempt {attempt})", flush=True)
                            return

            except (TimeoutError, urllib.error.URLError, OSError) as e:
                elapsed = time.time() - t0
                last_err = e
                print(f"[OCR] Attempt {attempt}/{OCR_MAX_RETRIES} failed for {api_base} after {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
            except Exception as e:
                elapsed = time.time() - t0
                last_err = e
                print(f"[OCR] Stream failed for {api_base} after {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
        if attempt < OCR_MAX_RETRIES:
            time.sleep(1)

    raise RuntimeError(f"OCR failed after {OCR_MAX_RETRIES} attempts: {last_err}")


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
                            'streaming': True})
        elif self.path == '/voices':
            self.send_json({'voices': [
                'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
                'am_adam', 'am_michael',
                'bf_emma', 'bf_isabella',
                'bm_george', 'bm_lewis',
            ]})
        elif self.path == '/conversation_state':
            self.handle_conversation_state()
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
        elif self.path == '/new_conversation':
            self.handle_new_conversation()
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
        """Screenshot -> streaming OCR -> segmented TTS pipeline."""
        image_b64 = data.get('image', '')
        voice = data.get('voice', DEFAULT_VOICE)
        constraints = data.get('constraints', '').strip()
        history_enabled = data.get('history', False)
        conversation_id = data.get('conversation_id', '')
        if not image_b64:
            self.send_error(400, 'No image provided')
            return

        try:
            image_bytes = base64.b64decode(image_b64)
            print(f"[OCR_TTS] Received {len(image_bytes)} byte image", flush=True)
            if constraints:
                print(f"[OCR_TTS] Constraints: {constraints[:200]}", flush=True)

            # Build user prompt text for history
            user_prompt = constraints if constraints else "Transcribe all readable text from this image verbatim. Do not add any description or commentary."

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
                full_text.append(fragment)
                buffer += fragment
                while True:
                    segment, buffer = pop_tts_segment(buffer)
                    if not segment:
                        break
                    self.send_stream_event({'type': 'text', 'text': segment})
                    tts_start = time.time()
                    wav_bytes = text_to_wav(segment, voice)
                    audio_chunks += 1
                    audio_b64 = base64.b64encode(wav_bytes).decode('ascii')
                    self.send_stream_event({'type': 'audio', 'text': segment, 'audio': audio_b64})
                    print(f"[OCR_TTS] Chunk {audio_chunks}: {len(segment)} chars -> TTS in {time.time() - tts_start:.1f}s", flush=True)

            segment, buffer = pop_tts_segment(buffer, force=True)
            if segment:
                self.send_stream_event({'type': 'text', 'text': segment})
                tts_start = time.time()
                wav_bytes = text_to_wav(segment, voice)
                audio_chunks += 1
                audio_b64 = base64.b64encode(wav_bytes).decode('ascii')
                self.send_stream_event({'type': 'audio', 'text': segment, 'audio': audio_b64})
                print(f"[OCR_TTS] Chunk {audio_chunks}: {len(segment)} chars -> TTS in {time.time() - tts_start:.1f}s", flush=True)

            text = ''.join(full_text).strip()
            if not text:
                self.send_stream_event({'type': 'error', 'error': 'No text found in image'})
                return

            # Save turn to conversation history if enabled
            if history_enabled and conversation_id:
                try:
                    save_turn(conversation_id, image_bytes, user_prompt, text)
                except Exception as e:
                    print(f"[OCR_TTS] Warning: failed to save turn: {e}", flush=True)

            total = time.time() - t0
            print(f"[OCR_TTS] Total: {total:.1f}s, {len(text)} chars, {audio_chunks} audio chunks", flush=True)
            done_event = {'type': 'done', 'text': text}
            if history_enabled and conversation_id:
                done_event['conversation_id'] = conversation_id
            self.send_stream_event(done_event)

        except Exception as e:
            print(f"[OCR_TTS ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()
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


def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS + Vision OCR server')
    parser.add_argument('--port', type=int, default=5912)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    print(f"[SERVER] Starting on {args.host}:{args.port}", flush=True)
    ensure_ollama_running()
    ensure_conversation_dir()
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
