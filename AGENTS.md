## Rules

Run project tests from WSL because the Kokoro server and its runtime dependencies work from WSL.

Kokoro server works from WSL. Launch `kokoro_server` via WSL with:

```bash
python3 ~/hermes-workdir/tts-extension/kokoro_server.py
```
