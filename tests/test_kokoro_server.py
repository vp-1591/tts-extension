import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault('numpy', types.SimpleNamespace(concatenate=lambda chunks: chunks))
sys.modules.setdefault('soundfile', types.SimpleNamespace(write=lambda *args, **kwargs: None))
sys.modules.setdefault('kokoro', types.SimpleNamespace(KPipeline=FakePipeline))

kokoro_server = importlib.import_module('kokoro_server')


class FlushableBytesIO(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1


class KokoroServerTests(unittest.TestCase):
    def test_pop_tts_segment_waits_for_sentence_or_size_limit(self):
        segment, remainder = kokoro_server.pop_tts_segment('partial sentence')

        self.assertIsNone(segment)
        self.assertEqual(remainder, 'partial sentence')

    def test_pop_tts_segment_splits_at_sentence_boundary(self):
        segment, remainder = kokoro_server.pop_tts_segment('First sentence. Second')

        self.assertEqual(segment, 'First sentence.')
        self.assertEqual(remainder, ' Second')

    def test_pop_tts_segment_does_not_split_domain_like_text(self):
        segment, remainder = kokoro_server.pop_tts_segment('Open abc.gmail.com before continuing')

        self.assertIsNone(segment)
        self.assertEqual(remainder, 'Open abc.gmail.com before continuing')

    def test_pop_tts_segment_flushes_final_sentence_without_trailing_space(self):
        segment, remainder = kokoro_server.pop_tts_segment('Final sentence.', force=True)

        self.assertEqual(segment, 'Final sentence.')
        self.assertEqual(remainder, '')

    def test_pop_tts_segment_forces_remaining_text(self):
        segment, remainder = kokoro_server.pop_tts_segment('final fragment', force=True)

        self.assertEqual(segment, 'final fragment')
        self.assertEqual(remainder, '')

    def test_send_stream_event_writes_ndjson_and_flushes(self):
        handler = types.SimpleNamespace(wfile=FlushableBytesIO())

        kokoro_server.TTSHandler.send_stream_event(handler, {'type': 'text', 'text': 'hello'})

        self.assertEqual(handler.wfile.flush_count, 1)
        payload = handler.wfile.getvalue().decode('utf-8')
        self.assertTrue(payload.endswith('\n'))
        self.assertEqual(json.loads(payload), {'type': 'text', 'text': 'hello'})

    def test_manifest_all_urls_permission_for_side_panel_capture(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())

        self.assertIn('activeTab', manifest['permissions'])
        self.assertIn('<all_urls>', manifest['host_permissions'])
        self.assertIn('http://127.0.0.1:5912/*', manifest['host_permissions'])


if __name__ == '__main__':
    unittest.main()
