## Rules

Run project tests from WSL because the Kokoro server and its runtime dependencies work from WSL.

Kokoro server works from WSL. Launch `kokoro_server` via WSL with:

```bash
python3 ~/hermes-workdir/tts-extension/kokoro_server.py
```

## Logs

Server logs are written to `logs/server.log` inside the project directory. In WSL the path is:

```
~/hermes-workdir/tts-extension/logs/server.log
```

Logs include TTFT (time to first token), TPS (tokens/chars per second), and all errors with tracebacks.

## __pycache__ warning

Chrome rejects loading extensions if the directory contains `__pycache__/` (filenames starting with `_` are reserved). If you run Python tests or import the server module, `__pycache__/` gets created. Always delete it before loading the extension:

```bash
find . -name '__pycache__' -type d -exec rm -rf {} +
```

The `.gitignore` already excludes `__pycache__/`, but the directory can still appear locally after running tests. Check for it before pushing or reloading the extension.

## Test maintenance

- When changing anything add or update focused tests that cover the changed behavior and any reported regression.