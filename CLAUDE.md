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

## Test maintenance

- When changing anything add or update focused tests that cover the changed behavior and any reported regression.