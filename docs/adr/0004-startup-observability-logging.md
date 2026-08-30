# ADR 0004: Startup Observability via Stdlib Phase Timing

## Status

Accepted

## Context

Server startup takes ~12 s from process spawn to `model_loaded`, but almost none of it was visible in the logs. The heavy import phase (`from kokoro import KPipeline` pulls in torch/transformers/spacy/phonemizer) runs at module import, *before* `main()` calls `setup_logging()`, so `logs/server.log` begins only at `[SERVER] Starting on ...`. The raw stderr redirect (`logs/server_spawner.log`) had no timestamps. Within `main()`, the Ollama check and model load ran in a background thread with phase-level timing only for request handling (ADR 0002's TTFT/TPS), not startup.

The goal is a measurable baseline of every startup phase so that speedup work (tracked in issues #6–#10) can target measured costs and verify improvements. A research pass compared loguru, structlog, OpenTelemetry, prometheus_client, cloud observability, and profilers against stdlib on code complexity and log coverage; none offers built-in phase timing, and the project is a single local Windows process where OTel/metrics infrastructure has no backend.

## Decision

Measure startup with the standard library only, using four deliberately separate mechanisms, each matched to its sink:

- **`_boot()` stderr markers** (`[BOOT] <ts> <msg>` at module top, before and after the heavy imports) for the pre-logging import phase; the native host's stderr redirect gives these timestamps in `server_spawner.log`, and the last marker shows how far a crashing import got.
- **`_phase()` context manager** (`[PHASE] <name> took <ms>ms`) for phases timed after `setup_logging()`: socket bind, Ollama ensure. A phase whose body raises logs `[PHASE] <name> failed after Nms` instead, so a crashed startup never contributes a success-shaped sample to the baseline.
- **Inline `time.monotonic()`** where the log line is device-specific or totals a span: model load in `get_pipeline()` (`loaded ... in Xms`), total startup in `main()`'s `Ready at ... (startup X.XXs)` — anchored at `_BOOT_START`, since the import phase dominates the total and a post-`main()` anchor would read ~0.02 s forever.
- **`_host_log()` in the native host** (`[HOST] <ts> ...`) appending to `server_spawner.log`, so the host's spawn/ready/failed timeline lives in the same file as the child's stderr. Guarded with `except OSError`, writing before `spawn_server()` so the child cannot precede the spawn entry. Both the host lines and the child's stdout/stderr go through **one shared file handle** (`spawner_fp`, opened by `spawn_server()` and inherited by the child): the CRT append flag of a mode-`'ab'` open does not survive process inheritance, so a second, separate open would let the child write at its own stale file offset and silently overwrite the `[HOST]` lines appended past it.

No third-party library is adopted (loguru is already installed but has no timing features; adopting it would rewrite ~30 calls for zero coverage gain). `uptime_ms`/`model_load_ms` fields for `/health` were considered and cut in adversarial review: no consumer reads them (the panel reads only `model_loaded`), and cutting them also removed two cross-thread globals. Timeline semantics of the two ready gates (native host = HTTP up; panel = `model_loaded`) remain unchanged. Stdlib logging to `logs/server.log` with the `%(asctime)s %(levelname)s %(message)s` format remains unchanged (originally decided in ADR 0002, §Decision).

## Constraints

- Log-line format and `logs/server.log` as the sink must not change (ADR 0002 consumers).
- No new runtime dependencies; no panel/UI changes; speedups are deferred to issues #6–#10, not implemented here.
- Tests must keep the stub-import pattern (`numpy`/`soundfile`/`kokoro` stubbed before `import kokoro_server`); no `__pycache__/` may be created.
- Elapsed measurement uses `time.monotonic()`, not wall clock; the native host's message protocol and `spawn_server()` Popen args are frozen by tests.

## Consequences

- **Positive**: every startup phase is now attributable in logs; the 2026-08-30 baseline (imports 6.11 s, socket bind 0 ms, ollama ensure 77 ms warm / up to 15 s cold, model load 5.66 s) is reproducible from `logs/server.log` alone.
- **Positive**: speedup issues have measured numbers to target and verify against.
- **Negative**: `_boot()` writes two lines to stderr during every test run (fd-captured by pytest; visible with `-s`). Accepted for crash visibility of the import phase.
- **Negative**: five timing idioms coexist (`_boot`, `_phase`, two inline sites, `_host_log`). This split is intentional; do not unify without revisiting this ADR.
- **Follow-up**: import-phase profile comes from a one-off `python -X importtime -c "import kokoro_server"` (exits cleanly, never starts the server), not shipped tooling.

## Validation

- `tests/test_kokoro_server.py::StartupTimingTests` — `_boot()` emits `[BOOT]` with a date-stamped prefix; `_phase()` logs `[PHASE] <name> took Nms`, and `[PHASE] <name> failed after Nms` (no success shape) when the wrapped body raises; `get_pipeline()` logs `Kokoro model loaded on <device> in Nms` and sets `MODEL_LOADED`.
- `tests/test_native_host.py::HostLogTests` — `_host_log()` appends a timestamped `[HOST]` line (and keeps the shared handle open when one exists) and survives `OSError`. `SpawnServerTests` asserts the child's `Popen` stdout is the module's `spawner_fp`, i.e. the same handle `_host_log()` writes through.
- Manual: start `./.venv/Scripts/python.exe kokoro_server.py --managed` with no other server running; `logs/server.log` shows `Imports took X.XXs`, `[PHASE] socket bind took Xms`, `[PHASE] ollama ensure took Xms`, the model-load ms line, and `Ready at ... (startup X.XXs)`; an extension auto-start puts `[BOOT]` lines into `logs/server_spawner.log`.