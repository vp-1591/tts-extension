## Rules

Run project tests from WSL because the Kokoro server and its runtime dependencies work from WSL.

Kokoro server works from WSL. Launch `kokoro_server` via WSL with:

```bash
python3 ~/hermes-workdir/tts-extension/kokoro_server.py
```

## Test maintenance

- When changing anything add or update focused tests that cover the changed behavior and any reported regression.