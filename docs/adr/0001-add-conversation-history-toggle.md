# ADR 0001: Add Conversation History Toggle and New Conversation Button

## Status

Accepted

## Context

The TTS Screen Reader extension operates as a stateless one-shot pipeline — each screenshot capture is processed independently with no memory of previous turns. When users want the OCR vision model to consider previous screenshots and transcriptions (e.g., to maintain context across a multi-page document or a series of related screens), there is no way to provide that context.

The Ollama chat API supports multi-turn conversations via the `messages` array, but the current server only sends a single system+user turn pair per request.

## Decision

Add two UI controls to both the side panel and popup:

1. **"Save History" toggle** — when enabled, each turn (screenshot + prompt + OCR result) is saved to a JSON file on the server, and all previous turns are sent to the Ollama model in subsequent requests. When disabled, behavior is identical to the current stateless mode.

2. **"+" button** — creates a new conversation (new directory and JSON file) and updates a persistent pointer file. This persists between server restarts via `conversations/current.txt`.

### Storage format

- Each conversation lives in `conversations/{conv_id}/` with a `conv.json` file and `images/` subdirectory
- `conv.json` stores an array of turns, each with `role`, `image_path` (relative), and `prompt`/`text`
- Images are saved as PNG files on disk and loaded on demand (not held in RAM between requests)
- The current conversation pointer is stored in `conversations/current.txt`

### API changes

- `POST /ocr_tts` now accepts optional `history` (bool) and `conversation_id` (string) fields
- `GET /conversation_state` returns `{conversation_id, turn_count}`
- `POST /new_conversation` creates a new conversation and returns `{conversation_id}`
- The `done` NDJSON event now includes `conversation_id` when history is enabled

### UI design

- iOS-style toggle switch (36×20px) with cyan accent when active
- Circular "+" button (28px) for new conversations
- Subtle conversation indicator showing shortened ID and turn count
- No chat selection tabs — UI remains clean

## Consequences

- **Positive**: Users can now maintain multi-turn context with the vision model, enabling better OCR on related screenshots
- **Positive**: Server restart preserves the current conversation pointer
- **Positive**: Images are loaded from disk on demand, keeping memory usage low
- **Risk**: Multi-turn conversations with images can exceed Ollama's context window — this is not yet handled (future enhancement could limit history turns)
- **Risk**: Single-threaded Python server could race on conversation file writes — mitigated with `threading.Lock`
- **Negative**: Conversation data accumulates on disk — no automatic cleanup mechanism yet