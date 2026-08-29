## Rules

The Kokoro server runs natively on **Windows** under the project's Python 3.12 venv at
`.venv/Scripts/python.exe` (kokoro requires `>=3.10,<3.13`; 3.13 cannot install it). Do not
run the server or the tests from WSL; there is no WSL fallback.

The server is started and stopped automatically by the extension: the side panel spawns it
through the `com.vp1591.tts_server` native-messaging host (`native_host/install.py` registers
it once under HKCU). Extension-spawned servers run with `--managed` and self-stop after
`HEARTBEAT_GRACE` (90 s) without a panel heartbeat. Servers started manually — without
`--managed` — never self-stop.

To launch the server by hand:

```bash
./.venv/Scripts/python.exe kokoro_server.py
```

## Logs

Server logs are written to `logs/server.log` inside the project directory
(`logs/server_spawner.log` holds early-startup output from extension-spawned servers).
Logs include TTFT (time to first token), TPS (tokens/chars per second), auto-stop lines
from the managed watchdog, and all errors with tracebacks.

## Tests

Run with Windows Python 3.13 and `PYTHONDONTWRITEBYTECODE=1` so no `__pycache__/` is created:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe -m pytest tests/ -v
```

## __pycache__ warning

Chrome rejects loading extensions if the directory contains `__pycache__/` (filenames starting with `_` are reserved). If you run Python tests or import the server module without `PYTHONDONTWRITEBYTECODE=1`, `__pycache__/` gets created. Always delete it before loading the extension:

```bash
find . -name '__pycache__' -type d -exec rm -rf {} +
```

The `.gitignore` already excludes `__pycache__/`, but the directory can still appear locally after running tests. Check for it before pushing or reloading the extension.

## Test maintenance

- When changing anything add or update focused tests that cover the changed behavior and any reported regression.