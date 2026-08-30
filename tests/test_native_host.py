import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native_host'))

tts_native_host = importlib.import_module('tts_native_host')
install = importlib.import_module('install')


class NativeMessagingProtocolTests(unittest.TestCase):
    def test_encode_read_round_trip(self):
        message = {'type': 'ready', 'model_loaded': True}
        stream = io.BytesIO(tts_native_host.encode_message(message))

        self.assertEqual(tts_native_host.read_message(stream), message)

    def test_encode_prefixes_little_endian_u32_length(self):
        body = json.dumps({'type': 'starting'}).encode('utf-8')
        encoded = tts_native_host.encode_message({'type': 'starting'})

        self.assertEqual(encoded[:4], len(body).to_bytes(4, 'little'))
        self.assertEqual(encoded[4:], body)

    def test_encode_rejects_oversized_message(self):
        # Chrome's native messaging limit for host->extension messages is 1 MB.
        with self.assertRaises(ValueError):
            tts_native_host.encode_message({'data': 'x' * (tts_native_host.MAX_MESSAGE_BYTES + 1)})


class HealthTests(unittest.TestCase):
    def test_health_parses_json_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {'status': 'ok', 'model_loaded': False}).encode('utf-8')
        response.__exit__.return_value = False
        with patch('tts_native_host.urllib.request.urlopen', return_value=response):
            self.assertEqual(tts_native_host.health(), {'status': 'ok', 'model_loaded': False})

    def test_health_returns_none_when_unreachable(self):
        with patch('tts_native_host.urllib.request.urlopen', side_effect=OSError('refused')):
            self.assertIsNone(tts_native_host.health())


class ExtensionIdTests(unittest.TestCase):
    """Guards the installer's default allowed_origins ID.

    A Chrome extension ID is always 32 chars a-p (first 32 hex digits of
    SHA-256 of the manifest key's DER SPKI, mapped 0-f -> a-p). A truncated
    16-char constant yields 'Access to the specified native messaging host
    is forbidden' at runtime and survived unit tests until a live-browser
    test caught it.
    """

    def test_default_id_matches_manifest_key_derivation(self):
        import base64
        import hashlib

        manifest_key = json.loads((ROOT / 'manifest.json').read_text())['key']
        digest = hashlib.sha256(base64.b64decode(manifest_key)).hexdigest()
        derived = ''.join(chr(ord('a') + int(c, 16)) for c in digest[:32])

        self.assertEqual(install.DEFAULT_EXTENSION_ID, derived)
        self.assertRegex(install.DEFAULT_EXTENSION_ID, r'^[a-p]{32}$')


class SpawnServerTests(unittest.TestCase):
    def setUp(self):
        self._orig_fp = tts_native_host.spawner_fp
        tts_native_host.spawner_fp = None

    def tearDown(self):
        if tts_native_host.spawner_fp is not None:
            tts_native_host.spawner_fp.close()
        tts_native_host.spawner_fp = self._orig_fp

    def test_spawn_server_launches_managed_server_from_repo(self):
        import tempfile
        popen = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(tts_native_host, 'SPAWNER_LOG', Path(tmp_dir) / 'spawner.log'), \
                patch('tts_native_host.subprocess.Popen', return_value=popen) as popen_ctor:
            tts_native_host.spawn_server()

            popen_ctor.assert_called_once()
            args, kwargs = popen_ctor.call_args
            self.assertEqual(args[0][1:], ['-X', 'utf8', str(tts_native_host.REPO / 'kokoro_server.py'),
                                           '--managed'])
            self.assertEqual(kwargs['cwd'], tts_native_host.REPO)
            # The child must inherit the same handle _host_log() writes through:
            # CRT append flags don't survive process inheritance, so a second
            # open would let the child write at stale offsets and clobber
            # [HOST] lines appended past them.
            self.assertIs(kwargs['stdout'], tts_native_host.spawner_fp)
            # Close before the TemporaryDirectory context exits, or Windows
            # blocks its cleanup while the handle is still open.
            tts_native_host.spawner_fp.close()
            tts_native_host.spawner_fp = None
        import os
        try:
            os.unlink(tmp_dir + '/spawner.log')
            print('DBG-UNLINK-OK')
        except OSError as e:
            print('DBG-UNLINK-FAIL', e)
            print('DBG-open-handles?', 'unknown')


class LogTailTests(unittest.TestCase):
    def test_log_tail_returns_message_when_missing(self):
        with patch.object(tts_native_host, 'SPAWNER_LOG', Path('Z:/nonexistent/log')):
            self.assertEqual(tts_native_host._spawner_log_tail(), '(no spawner log written)')

    def test_log_tail_truncates_to_limit(self):
        import tempfile
        with tempfile.NamedTemporaryFile('wb', delete=False) as tmp:
            tmp.write(b'x' * 3000)
            tmp_path = Path(tmp.name)
        try:
            with patch.object(tts_native_host, 'SPAWNER_LOG', tmp_path):
                tail = tts_native_host._spawner_log_tail(limit=100)
            self.assertEqual(len(tail), 100)
        finally:
            tmp_path.unlink(missing_ok=True)


class HostLogTests(unittest.TestCase):
    def test_host_log_appends_timestamped_line(self):
        import os
        import tempfile
        fd, name = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        tmp_path = Path(name)
        try:
            with patch.object(tts_native_host, 'SPAWNER_LOG', tmp_path):
                tts_native_host._host_log('x')

            content = tmp_path.read_text(encoding='utf-8')
            self.assertIn('[HOST] x', content)
            self.assertRegex(content, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[HOST\] x')
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_host_log_writes_through_shared_handle(self):
        buf = io.BytesIO()
        with patch.object(tts_native_host, 'spawner_fp', buf):
            tts_native_host._host_log('spawned')

        self.assertIn(b'[HOST] spawned', buf.getvalue())
        # Handle must stay open — the spawned child still writes through it.
        self.assertFalse(buf.closed)

    def test_host_log_survives_oserror(self):
        # Opening an existing directory raises PermissionError (an OSError) on
        # Windows, so the guard is exercised deterministically; the previous
        # drive-letter probing was a silent no-op on machines with those
        # drives mapped and could create files outside the repo.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(tts_native_host, 'SPAWNER_LOG', Path(tmp_dir)):
            tts_native_host._host_log('x')  # must not raise