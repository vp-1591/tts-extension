# ADR 0002: Add Structured Logging and Fix Conversation History Bugs

## Status

Accepted

## Context

Several bugs were found after the conversation history feature was deployed:

1. **Empty images directory** — When history is enabled but no `conversation_id` is stored in localStorage yet, the server receives `history=true` with an empty `conversation_id`. Since `save_turn()` requires a non-empty `conversation_id`, the first turn is never saved, and no images are written to disk.

2. **"+" button not working** — The `POST /new_conversation` endpoint was routed through the generic JSON body parser in `do_POST`, which calls `json.loads(body)`. Since the client sends no body with this request, `json.loads('')` raises `JSONDecodeError` and returns a 400 error.

3. **Gray text invisible on dark background** — Several CSS colors (`#445`, `#667`, `#888`) were nearly invisible on the `#1a1a2e` dark background.

4. **No persistent logging** — All server output went to stdout only, making it hard to diagnose issues after the fact. There was no TTFT, TPS, or error file logging.

## Decision

### Bug fixes

- **Auto-assign conversation**: When `history=true` but `conversation_id` is empty, the server calls `ensure_conversation_dir()` to get or create the current conversation. The assigned `conversation_id` is included in the `done` event so the client can persist it.
- **Early-route /new_conversation**: Move the `/new_conversation` handler check before the JSON body parser in `do_POST`, so it works without a request body.
- **Persist conversation_id in client**: `updateConvIndicator()` now also writes `conversation_id` to `localStorage`, so the first OCR request with history enabled already has a valid ID.
- **Brighten CSS**: Changed `#445` → `#778`, `#667` → `#99a`, `#99a` → `#b8c`, `#888` → `#aaa`, `.conv-indicator` → `#8892b0`.

### Logging

- Replace all `print()` calls with Python `logging` module (`logging.info`, `logging.warning`, `logging.error`)
- Add file handler writing to `logs/server.log` (auto-created on startup)
- Add TTFT (time to first token) logging in `ocr_image_stream()`
- Add TPS (chars/second) logging for both OCR and TTS
- Use `exc_info=True` for error logging instead of separate `traceback.print_exc()`
- Add `logs/` to `.gitignore`

## Consequences

- **Positive**: First OCR request with history enabled now correctly saves the turn and image
- **Positive**: "+" button creates a new conversation and updates the indicator
- **Positive**: All text elements are legible on the dark background
- **Positive**: Server logs persist to disk for post-mortem debugging
- **Positive**: TTFT and TPS metrics are available for performance monitoring
- **Risk**: Log files can grow unbounded — no rotation is implemented yet