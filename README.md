# TTS Screen Reader

A Chrome extension that captures screenshots, runs OCR via a vision model, and reads text aloud using Kokoro TTS — all through a local Python server.

## How It Works

1. **Capture** — Click the button to screenshot the current tab
2. **OCR** — The screenshot is sent to Ollama's vision model for text recognition
3. **TTS** — Recognized text is streamed back and converted to speech in real time via Kokoro 82M

The pipeline streams audio incrementally — you hear text as it's being recognized, not after the entire OCR finishes.

The server starts and stops **automatically**: opening the side panel launches it (via Chrome
native messaging), and it shuts itself down ~90 s after the last panel (i.e. Chrome) closes.

## Architecture

```
Chrome side panel (panel.js)
    │
    ├── connectNative("com.vp1591.tts_server")   ← server is offline?
    │       └──► native_host/tts_native_host.py   (stateless spawner, exits after reporting)
    │               └──► spawns kokoro_server.py --managed
    │
    ├── GET /health        (polls while starting; requires model_loaded: true)
    ├── POST /panel-heartbeat every 15 s (drives the server's auto-stop watchdog)
    │
    ├── Screenshot (chrome.tabs.captureVisibleTab)
    ▼
kokoro_server.py :5912  (runs natively on Windows)
    │
    ├── POST /ocr_tts ──► Ollama Vision API (streaming OCR)
    │                       │
    │                       ▼
    │                   Kokoro TTS (CUDA, CPU fallback)
    │                       │
    │                       ▼
    │                   NDJSON stream ──► Chrome plays audio chunks
    │
    ├── POST /tts ──► text → Kokoro TTS → WAV
    ├── GET  /voices ──► available voice list
    ├── GET  /health ──► server status + model info
    ├── GET  /conversation_state ──► current conversation ID + turn count
    └── POST /new_conversation ──► create fresh conversation
```

### Server lifetime (auto-start / auto-stop)

- **Start:** the side panel connects to the `com.vp1591.tts_server` native messaging host,
  a small stdlib-only script that checks `/health`, spawns `kokoro_server.py --managed` if
  needed, waits for readiness (≤90 s), reports, and exits.
- **Stop:** while the panel is open it heartbeats every 15 s; a `--managed` server exits
  after `HEARTBEAT_GRACE` (default 90 s) of silence. Closing Chrome, a crash, or hibernate
  all just make the heartbeats stop — the panel restarts the server on demand next time.
- A server started **manually** (`python kokoro_server.py`, no `--managed`) never self-stops.
- Exactly one server/model runs regardless of how many panels or hosts connect: a second
  bind attempt fails loudly with `WSAEADDRINUSE` and its spawner reports the winner as ready.

## Prerequisites

- **Windows** with **Python 3.10–3.12** for the server venv — kokoro requires `>=3.10,<3.13`
  (on 3.13 its `numpy==1.26.4` pin has no wheels and the build fails)
- `torch` (CUDA 12.8 wheel), `kokoro`, `numpy`, `soundfile` — install order matters, see below
- **Ollama** running natively on Windows with a vision model (default: `gemma4:31b:cloud`)
- **Chrome**
- CUDA-capable GPU recommended (falls back to CPU automatically; OCR dominates latency)

## Installation

```powershell
# 1. Project venv (Python 3.12)
python -m venv .venv            # or: uv venv .venv --python 3.12 --seed

# 2. Python deps — install torch from the CUDA index FIRST, otherwise pip pulls a
#    multi-GB CPU-only torch as a kokoro dependency
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install kokoro numpy soundfile

# 3. Ollama vision model
ollama pull gemma4:31b:cloud

# 4. Register the native messaging host (no admin; writes under HKCU)
.venv\Scripts\python.exe native_host\install.py

#    Uninstall: .venv\Scripts\python.exe native_host\install.py --uninstall
#    Custom ID/interpreter: --extension-id <ID> / --python <path>
```

Note: **pip success ≠ TTS success** — the Kokoro G2P stage needs espeak-ng at runtime.
Verify with the smoke test below before assuming a working install.

## Loading the Extension in Chrome

1. Go to `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** and select the project directory
4. The extension icon appears in the toolbar; clicking it opens the side panel

The manifest pins the extension ID via its `key` field, so the ID is stable across load-path
changes. If you load the extension for the first time after this change, remove + re-add it
once — the ID changes once and localStorage preferences (voice/constraints/history) reset.

## Running the Server Manually

```bash
# Optional — the panel auto-starts it. Manual starts never self-stop.
python kokoro_server.py --port 5912 --host 127.0.0.1
```

The server automatically:
- Starts Ollama if it's not running
- Loads the Kokoro model in a background thread (with CUDA → CPU fallback)
- In `--managed` mode: shuts down when panel heartbeats stop (see lifetime above)

## First-run smoke test

```bash
./.venv/Scripts/python.exe kokoro_server.py   # leave running
curl -X POST http://127.0.0.1:5912/tts -H "Content-Type: application/json" -d "{\"text\":\"windows check\"}" -o out.wav
```

`out.wav` should be audible. If pip installed cleanly but this fails, the usual cause is the
espeak-ng based G2P stage (runtime dependency, not a pip dependency).

## Features

| Feature | Description |
|---------|-------------|
| **Streaming pipeline** | Audio plays incrementally as OCR progresses |
| **Auto-start / auto-stop** | Server starts when the panel opens; stops ~90 s after the last panel closes |
| **Multiple voices** | US/UK, male/female voices via Kokoro 82M |
| **Optional constraints** | Custom prompt to guide OCR output |
| **Conversation history** | Toggle to save multi-turn context for the vision model |
| **New conversation** | `+` button starts a fresh session |
| **CUDA recovery** | Automatically falls back to CPU on GPU errors |
| **Ollama auto-start** | Launches Ollama if not already running |

## Conversation History

When the **History** toggle is enabled:
- Each turn (screenshot + prompt + OCR result) is saved to `conversations/{conv_id}/conv.json`
- Screenshots are stored as PNG files in `conversations/{conv_id}/images/`
- Previous turns are sent to the vision model as multi-turn context
- Images are loaded from disk on demand (not held in RAM)

When disabled, the extension operates in stateless mode — no data is saved or sent as history.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ocr_tts` | Screenshot → OCR → TTS streaming pipeline |
| `POST` | `/tts` | Plain text → TTS audio |
| `POST` | `/panel-heartbeat` | Panel liveness ping (drives managed auto-stop) |
| `GET` | `/voices` | List available voices |
| `GET` | `/health` | Server status, `model_loaded`, `managed` |
| `GET` | `/conversation_state` | Current conversation ID and turn count |
| `POST` | `/new_conversation` | Create a new conversation |

### POST /ocr_tts

```json
{
  "image": "<base64_png>",
  "voice": "af_bella",
  "constraints": "Only read headers",
  "history": true,
  "conversation_id": "conv_20260618_105539"
}
```

The `history` and `conversation_id` fields are optional. When `history` is `true` and no `conversation_id` is provided, the server auto-assigns the current conversation.

Response: streaming NDJSON with `text`, `audio`, `error`, and `done` events.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL` | `gemma4:31b:cloud` | Ollama vision model name |
| `VISION_API_BASE` | `http://127.0.0.1:11434` | Ollama API URL |
| `VISION_API_KEY` | `ollama` | API key |
| `OLLAMA_STARTUP_TIMEOUT` | `15` | Seconds to wait for Ollama startup |
| `TTS_WARMUP` | `1` | Run a tiny warmup synthesis after model load (0 disables — skips CUDA kernel warm-up, first TTS chunk then costs 1.3-3.1s) |
| `HF_OFFLINE_IF_CACHED` | `1` | Set `HF_HUB_OFFLINE=1` automatically when the Kokoro model is present in the local HF cache (0 always fetches; needed to download non-cached voices) |
| `HEARTBEAT_GRACE` | `90` | Seconds of silence before a `--managed` server stops |

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Specified native messaging host not found` | Host not registered — run `python native_host\install.py`, check `HKCU\Software\Google\Chrome\NativeMessagingHosts\com.vp1591.tts_server` |
| `Access to the specified native messaging host is forbidden` | `allowed_origins` doesn't match your extension ID — rerun `install.py --extension-id <your-ID>` |
| `Native host has exited` | Run `python native_host\tts_native_host.py` by hand to see the error; check `logs/server_spawner.log` |
| Panel says offline even after reinstall | Copy your extension ID from `chrome://extensions` and pass it via `--extension-id` |
| pip installs fine but every TTS request errors | G2P/espeak-ng runtime failure — install `espeakng-loader`/`phonemizer-fork`, retest with the smoke test above |
| Server never stops | It wasn't started `--managed` (manual starts are intentionally persistent) |

## Logs

Server logs are written to `logs/server.log` in the project directory (`logs/server_spawner.log`
holds early-startup output from extension-spawned servers). Logged metrics include:

- **TTFT** — Time to first token (OCR latency)
- **TPS** — Tokens/chars per second (OCR and TTS throughput)
- **Errors** — Full tracebacks for all exceptions
- **Auto-stops** — `Auto-stopping (no panel heartbeat …)` lines from the watchdog

## Testing

```bash
# Project venv (Python 3.12); PYTHONDONTWRITEBYTECODE avoids stale __pycache__ breaking
# Chrome's unpacked-extension loader
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe -m pytest tests/ -v
```

If you run tests without it, delete bytecode dirs before reloading the extension
(Chrome rejects unpacked extensions containing `__pycache__/`):

```bash
find . -name '__pycache__' -type d -exec rm -rf {} +
```

## License

MIT