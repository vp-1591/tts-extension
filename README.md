# TTS Screen Reader

A Chrome extension that captures screenshots, runs OCR via a vision model, and reads text aloud using Kokoro TTS — all through a local Python server.

## How It Works

1. **Capture** — Click the button to screenshot the current tab
2. **OCR** — The screenshot is sent to Ollama's vision model for text recognition
3. **TTS** — Recognized text is streamed back and converted to speech in real time via Kokoro 82M

The pipeline streams audio incrementally — you hear text as it's being recognized, not after the entire OCR finishes.

## Architecture

```
Chrome Extension (popup / side panel)
    │
    ├── Screenshot (chrome.tabs.captureVisibleTab)
    │
    ▼
Python HTTP Server (kokoro_server.py :5912)
    │
    ├── POST /ocr_tts ──► Ollama Vision API (streaming OCR)
    │                       │
    │                       ▼
    │                   Kokoro TTS (CUDA/CPU)
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

## Prerequisites

- **Python 3.10+** with `kokoro`, `numpy`, `soundfile` packages
- **Ollama** with a vision model (default: `gemma4:31b:cloud`)
- **CUDA-capable GPU** (optional — falls back to CPU automatically)
- **Chrome** browser

## Installation

```bash
# Clone the repository
git clone https://github.com/vp-1591/tts-extension.git
cd tts-extension

# Install Python dependencies (in WSL or Linux)
pip install kokoro numpy soundfile

# Install the Ollama vision model
ollama pull gemma4:31b:cloud
```

## Running the Server

```bash
# Start the server (from WSL)
python3 ~/hermes-workdir/tts-extension/kokoro_server.py

# With custom options
python3 kokoro_server.py --port 5912 --host 127.0.0.1
```

The server automatically:
- Starts Ollama if it's not running
- Loads the Kokoro model (with CUDA → CPU fallback)
- Creates a default conversation directory

## Loading the Extension in Chrome

1. Go to `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** and select the project directory
4. The extension icon appears in your toolbar

## Features

| Feature | Description |
|---------|-------------|
| **Streaming pipeline** | Audio plays incrementally as OCR progresses |
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
| `GET` | `/voices` | List available voices |
| `GET` | `/health` | Server status and model info |
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
| `TTS_SKIP_WARM` | `1` | Skip Kokoro warm-up |

## Logs

Server logs are written to `logs/server.log` in the project directory. On WSL:

```
~/hermes-workdir/tts-extension/logs/server.log
```

Logged metrics include:
- **TTFT** — Time to first token (OCR latency)
- **TPS** — Tokens/chars per second (OCR and TTS throughput)
- **Errors** — Full tracebacks for all exceptions

## Testing

```bash
# From WSL
cd ~/hermes-workdir/tts-extension
python -m pytest tests/ -v
```

## License

MIT