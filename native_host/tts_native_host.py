#!/usr/bin/env python3
"""Chrome native-messaging host: spawns kokoro_server.py on demand, then exits.

Chrome cannot start processes from a web page, so the side panel connects here
via connectNative(). The host reports status over the native-messaging protocol
(4-byte little-endian length prefix + JSON on stdin/stdout): it checks /health,
spawns the server with --managed if offline, waits for readiness, and exits —
it holds no long-lived state, and the server's own heartbeat watchdog owns the
server lifetime.
"""

import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST_NAME = 'com.vp1591.tts_server'
SERVER_URL = 'http://127.0.0.1:5912'
REPO = Path(__file__).resolve().parents[1]
READY_TIMEOUT = 90
SPAWNER_LOG = REPO / 'logs' / 'server_spawner.log'
# Chrome rejects host->extension messages larger than 1 MB.
MAX_MESSAGE_BYTES = 1_000_000


def encode_message(obj) -> bytes:
    body = json.dumps(obj).encode('utf-8')
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError(f'native message too large: {len(body)} bytes')
    return struct.pack('<I', len(body)) + body


def read_message(stream) -> dict:
    raw_len = stream.read(4)
    if len(raw_len) < 4:
        raise EOFError('native host stdin closed')
    (length,) = struct.unpack('<I', raw_len)
    return json.loads(stream.read(length).decode('utf-8'))


def send_message(obj) -> None:
    sys.stdout.buffer.write(encode_message(obj))
    sys.stdout.buffer.flush()


def health(timeout: float = 1.0):
    """GET /health as a dict, or None when the server is unreachable."""
    try:
        with urllib.request.urlopen(f'{SERVER_URL}/health', timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except (OSError, ValueError):
        return None


# Handle the spawned child inherits for stdout/stderr, so _host_log() and the
# child share one file pointer. The CRT append flag of a mode-'ab' open does
# not survive process inheritance: a second, separate open lets the child
# write at its own stale offset, clobbering [HOST] lines appended past it.
spawner_fp = None


def spawn_server() -> subprocess.Popen:
    global spawner_fp
    SPAWNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    spawner_fp = open(SPAWNER_LOG, 'ab')
    # sys.executable is the interpreter the .bat wrapper was generated with,
    # so the server runs under the same (CUDA-capable) Python as the host.
    return subprocess.Popen(
        [sys.executable, '-X', 'utf8', str(REPO / 'kokoro_server.py'), '--managed'],
        stdout=spawner_fp,
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )


def set_binary_stdio() -> None:
    if os.name == 'nt':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def _spawner_log_tail(limit: int = 2000) -> str:
    try:
        data = SPAWNER_LOG.read_bytes()
        return data[-limit:].decode('utf-8', errors='replace')
    except OSError:
        return '(no spawner log written)'


def _host_log(msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} [HOST] {msg}\n'.encode('utf-8')
    try:
        if spawner_fp is None:
            # Child not spawned (or spawn failed) — safe to open transiently.
            SPAWNER_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(SPAWNER_LOG, 'ab') as f:
                f.write(line)
        else:
            spawner_fp.write(line)
            spawner_fp.flush()
    except OSError:
        # Logging must never kill the host before it answers Chrome.
        pass


def main() -> None:
    set_binary_stdio()
    started_at = time.monotonic()
    send_message({'type': 'starting'})

    if health() is None:
        _host_log('spawning kokoro_server.py --managed')
        spawn_server()

    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        state = health()
        if state is not None:
            # Covers both our own child and a server another host won the
            # double-open race with — /health is the single source of truth.
            send_message({'type': 'ready', 'model_loaded': bool(state.get('model_loaded'))})
            _host_log(f'ready after {time.monotonic() - started_at:.2f}s')
            return
        time.sleep(1)

    _host_log(f'failed after {time.monotonic() - started_at:.2f}s')
    send_message({'type': 'failed', 'error': _spawner_log_tail()})


if __name__ == '__main__':
    main()