import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


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


class ConversationTests(unittest.TestCase):
    """Tests for conversation history storage and retrieval."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.orig_conv_dir = kokoro_server.CONVERSATIONS_DIR
        self.orig_current_ptr = kokoro_server.CURRENT_PTR
        # Redirect conversation storage to temp directory
        kokoro_server.CONVERSATIONS_DIR = Path(self.tmp_dir)
        kokoro_server.CURRENT_PTR = Path(self.tmp_dir) / 'current.txt'

    def tearDown(self):
        # Restore original paths
        kokoro_server.CONVERSATIONS_DIR = self.orig_conv_dir
        kokoro_server.CURRENT_PTR = self.orig_current_ptr
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_new_conversation_creates_dir_and_pointer(self):
        conv_id = kokoro_server.new_conversation()

        # Verify directory structure
        conv_dir = Path(self.tmp_dir) / conv_id
        self.assertTrue(conv_dir.is_dir())
        self.assertTrue((conv_dir / 'images').is_dir())
        self.assertTrue((conv_dir / 'conv.json').exists())

        # Verify conv.json content
        data = json.loads((conv_dir / 'conv.json').read_text(encoding='utf-8'))
        self.assertEqual(data['id'], conv_id)
        self.assertEqual(data['turns'], [])

        # Verify pointer file
        ptr_content = (Path(self.tmp_dir) / 'current.txt').read_text().strip()
        self.assertEqual(ptr_content, conv_id)

    def test_ensure_conversation_dir_uses_existing(self):
        conv_id = kokoro_server.new_conversation()
        result = kokoro_server.ensure_conversation_dir()
        self.assertEqual(result, conv_id)

    def test_ensure_conversation_dir_creates_when_missing(self):
        # No current.txt exists yet
        self.assertFalse((Path(self.tmp_dir) / 'current.txt').exists())
        result = kokoro_server.ensure_conversation_dir()
        self.assertTrue(result.startswith('conv_'))
        self.assertTrue((Path(self.tmp_dir) / 'current.txt').exists())

    def test_ensure_conversation_dir_recovers_from_invalid_pointer(self):
        # Write invalid pointer
        (Path(self.tmp_dir) / 'current.txt').write_text('nonexistent_conv_id')
        result = kokoro_server.ensure_conversation_dir()
        self.assertTrue(result.startswith('conv_'))
        # Should have created a new conversation
        self.assertTrue((Path(self.tmp_dir) / result / 'conv.json').exists())

    def test_save_turn_writes_image_and_updates_json(self):
        conv_id = kokoro_server.new_conversation()
        image_data = b'\x89PNG\r\n\x1a\nfake_image_data'

        kokoro_server.save_turn(conv_id, image_data, 'Read this image', 'Hello World')

        # Verify image file was created
        conv_dir = Path(self.tmp_dir) / conv_id
        image_file = conv_dir / 'images' / '001.png'
        self.assertTrue(image_file.exists())
        self.assertEqual(image_file.read_bytes(), image_data)

        # Verify conv.json was updated
        data = json.loads((conv_dir / 'conv.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['turns']), 2)
        self.assertEqual(data['turns'][0]['role'], 'user')
        self.assertEqual(data['turns'][0]['prompt'], 'Read this image')
        self.assertEqual(data['turns'][0]['image_path'], f'{conv_id}/images/001.png')
        self.assertEqual(data['turns'][1]['role'], 'assistant')
        self.assertEqual(data['turns'][1]['text'], 'Hello World')

    def test_save_turn_multiple_turns(self):
        conv_id = kokoro_server.new_conversation()
        kokoro_server.save_turn(conv_id, b'img1', 'prompt1', 'text1')
        kokoro_server.save_turn(conv_id, b'img2', 'prompt2', 'text2')

        data = json.loads((Path(self.tmp_dir) / conv_id / 'conv.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['turns']), 4)
        self.assertEqual(data['turns'][0]['image_path'], f'{conv_id}/images/001.png')
        self.assertEqual(data['turns'][2]['image_path'], f'{conv_id}/images/002.png')

    def test_load_conversation(self):
        conv_id = kokoro_server.new_conversation()
        data = kokoro_server.load_conversation(conv_id)
        self.assertEqual(data['id'], conv_id)
        self.assertIsInstance(data['created'], str)
        self.assertEqual(data['turns'], [])

    def test_is_cuda_error_detects_cuda_exceptions(self):
        # Test CUDA-related error detection
        self.assertTrue(kokoro_server.is_cuda_error(RuntimeError('CUDA error: unknown error')))
        self.assertTrue(kokoro_server.is_cuda_error(Exception('cuda out of memory')))

        # Test non-CUDA errors (mock torch to avoid import issues on Windows)
        with patch.dict('sys.modules', {'torch': MagicMock(cuda=MagicMock(OutOfMemoryError=type('OOM', (), {})))}):
            self.assertFalse(kokoro_server.is_cuda_error(RuntimeError('Connection refused')))
            self.assertFalse(kokoro_server.is_cuda_error(TimeoutError('request timed out')))


class HandlerTests(unittest.TestCase):
    """Tests for HTTP handler routing."""

    def test_do_post_new_conversation_no_body_required(self):
        """Verify /new_conversation works without a JSON body (bug fix)."""
        handler = kokoro_server.TTSHandler.__new__(kokoro_server.TTSHandler)
        handler.path = '/new_conversation'
        handler.headers = {}
        handler.rfile = io.BytesIO(b'')
        handler.wfile = FlushableBytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        # This should NOT raise a JSONDecodeError
        # The handler should route directly without parsing body
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_conv_dir = kokoro_server.CONVERSATIONS_DIR
            orig_current_ptr = kokoro_server.CURRENT_PTR
            kokoro_server.CONVERSATIONS_DIR = Path(tmp_dir)
            kokoro_server.CURRENT_PTR = Path(tmp_dir) / 'current.txt'

            try:
                result = kokoro_server.TTSHandler.do_POST(handler)
                # The handler should have called handle_new_conversation
                # which creates a conversation and sends JSON response
            finally:
                kokoro_server.CONVERSATIONS_DIR = orig_conv_dir
                kokoro_server.CURRENT_PTR = orig_current_ptr


if __name__ == '__main__':
    unittest.main()