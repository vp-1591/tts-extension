# ADR 0003: Sanitize Vision-Model OCR Output Before TTS

## Status

Accepted

## Context

The `/ocr_tts` pipeline streams the vision model's raw output straight into Kokoro (`pop_tts_segment` → `text_to_wav`). The model decorates transcriptions with markdown and emoji even when told not to, so spoken audio says "star star heading star star", "dash", or emoji names. The panel caption shows the same raw text (`panel.js` accumulates `text` events), so captions are noisy too.

The output must be spoken, so any formatting the phonemizer cannot pronounce is noise. The choice was between fixing the model's behavior at the prompt level, fixing the audio path at the code level, or both.

## Decision

Transcriptions are sanitized in two layers before speech:

1. **Prompt hardening** — both OCR system prompts request plain text only (no markdown, bullets, emojis, emphasis), keeping the visible transcription faithful to plain text even without sanitizer coverage.
2. **`sanitize_for_speech()`** — a deterministic floor applied to every popped segment (caption + audio events) and to the full text used for the `done` event and saved conversation history. Paired markers (`**`/`*`/`~~`, inline links→alt text, backticks), ATX headers, hr rules, line-start bullets, and checkboxes are removed; emoji/dingbat/arrow code points are dropped silently. Line-start stripping is enough for bullets because the `\n` split already produces a segment pause.

Behavior choices, from the UX session (2026-08-29):

- **Bullets** → marker removed; the segment boundary is the pause.
- **Numbered lists kept** — "1." reads as "one", preserving ordering.
- **Emojis skipped silently** (not verbalized, not panned with pauses).
- **Code content touches nothing** — the sanitizer removes formatting tokens only, never content symbols. Verbatim-symbol transcription is what user-defined constraints are for. The `__double-underscore__` bold rule is deliberately absent: it would eat identifiers like `__init__`.

The sanitizer runs on saved history too: history fed back to the vision model then contains plain text, which steers later turns away from markdown.

Alternatives rejected:
- **Prompt-only** — models drift mid-transcription; long outputs reintroduce artifacts (~20% observed).
- **Sanitizer-only** — leaves captions dependent on the cleanup, and the model wastes tokens on markup; both together give redundancy with no single point of failure.
- **Special-casing fenced code** (e.g. "code block, 3 lines" summary or symbol expansion) — rejected; user-defined constraints own that, and the general sanitizer must not mangle content it doesn't recognize.
- **Verbalizing emoji names** — maximally noisy, the very interruption users are escaping.

## Constraints

- The sanitizer removes formatting only — transcription content must survive verbatim (identifiers, prose em dashes, `10 > 5` comparisons).
- Mid-line `-`/`*`/`—`/`>` may be prose or math, not list markers; stripping is restricted to line starts.
- Numbered lists must remain speakable with their numbers.
- No new dependencies. Sanitizer must be fast (runs per segment in the streaming path, server is single-threaded per request).
- `/tts` (user-supplied text) is untouched — the concern is model output, not user input.

## Consequences

- **Positive**: Spoken audio and captions no longer contain "star star"/"dash"/emoji noise; transcripts and history are plain text, which also nudges follow-up model turns toward plain output.
- **Negative**: A markup pair split across a segment boundary leaves an orphan marker in the mid-segment caption (edge-trimmed at segment edges only); the whole-text cleanup for `done`/history handles it, so the two texts can differ by a stray token in rare cases.
- **Negative**: Aggressive emoji ranges mean a genuine transcription of, say, an arrow symbol won't be spoken — accepted: symbols read aloud by Kokoro were garbage anyway.
- **Follow-up**: If the model still leaks markup the regexes miss, extend `_MARKUP_RULES`; prompt hardening alone is not trusted as a guarantee.

## Validation

- `tests/test_kokoro_server.py::SanitizeForSpeechTests` — paired stars, headers/bullets/hr rules, numbered items kept, emoji stripped, links→alt text, code identifiers preserved, orphan-marker edge trimming.
- `tests/test_kokoro_server.py::OcrTtsStreamTests::test_stream_sanitizes_text_audio_and_done_events` — full `/ocr_tts` stream: no `*` or emoji in any text event, audio event text matches sanitized caption, `done` text equals the spoken transcript.
- Manual: read a screen containing a bulleted list via the panel; audio and caption show plain lines without markers.