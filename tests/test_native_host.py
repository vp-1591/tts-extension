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


class SpawnServerTests(unittest.TestCase):
    def test_spawn_server_launches_managed_server_from_repo(self):
        popen = MagicMock()
        with patch('tts_native_host.subprocess.Popen', return_value=popen) as popen_ctor:
            tts_native_host.spawn_server()

        popen_ctor.assert_called_once()
        args, kwargs = popen_ctor.call_args
        self.assertEqual(args[0][1:], ['-X', 'utf8', str(tts_native_host.REPO / 'kokoro_server.py'),
                                       '--managed'])
        self.assertEqual(kwargs['cwd'], tts_native_host.REPO)


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