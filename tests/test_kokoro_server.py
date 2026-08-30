import contextlib
import importlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_manifest_allows_native_messaging_and_pins_id(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())

        self.assertIn('nativeMessaging', manifest['permissions'])
        # The "key" field pins the unpacked extension ID so the native host
        # manifest's allowed_origins can stay a single constant.
        self.assertTrue(manifest.get('key'))


class SanitizeForSpeechTests(unittest.TestCase):
    def test_strips_paired_stars(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('**Big heading** body'), 'Big heading body')

    def test_strips_headers_bullets_and_rules(self):
        text = '# Title\n- first\n* second\n---\nPlain tail.'
        self.assertEqual(kokoro_server.sanitize_for_speech(text), 'Title\nfirst\nsecond\nPlain tail.')

    def test_keeps_numbered_items(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('1. Get milk\n2. Buy eggs'), '1. Get milk\n2. Buy eggs')

    def test_strips_emoji_silently(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('Great! 🎉\nWarning ⚠️ sign'), 'Great!\nWarning sign')

    def test_link_replaced_by_text(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('See [docs](https://example.com/x) now'), 'See docs now')

    def test_code_content_untouched(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('call foo_bar_baz here'), 'call foo_bar_baz here')

    def test_inline_backticks_removed_content_kept(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('run `npm install` first'), 'run npm install first')

    def test_arithmetic_plus_minus_survives(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('2 - 3 items'), '2 - 3 items')

    def test_orphan_markers_trimmed_at_edges(self):
        self.assertEqual(kokoro_server.sanitize_for_speech('**Heading.'), 'Heading.')


class OcrTtsStreamTests(unittest.TestCase):
    def make_handler(self):
        handler = kokoro_server.TTSHandler.__new__(kokoro_server.TTSHandler)
        handler.rfile = io.BytesIO(b'')
        handler.wfile = FlushableBytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_stream_sanitizes_text_audio_and_done_events(self):
        handler = self.make_handler()
        fragments = ['**Big heading**\n', '- first bullet 🎉\n', 'Plain tail.']
        with patch.object(kokoro_server, 'ocr_image_stream', return_value=iter(fragments)), \
             patch.object(kokoro_server, 'text_to_wav', return_value=b'RIFF'):
            kokoro_server.TTSHandler.handle_ocr_tts(handler, {'image': 'aW1n', 'voice': 'af_bella'})

        events = [json.loads(line) for line in handler.wfile.getvalue().decode('utf-8').splitlines() if line]
        texts = [e['text'] for e in events if e['type'] == 'text']
        spoken = '\n'.join(texts)

        self.assertNotIn('*', spoken)
        self.assertNotIn('🎉', spoken)
        self.assertEqual(spoken, 'Big heading\nfirst bullet\nPlain tail.')
        self.assertTrue(events[-1]['type'] == 'done')
        self.assertEqual(events[-1]['text'], spoken)
        self.assertTrue(any(e['type'] == 'audio' and e['text'] == 'Big heading' for e in events))


class ManagedLifetimeTests(unittest.TestCase):
    def setUp(self):
        # Reset globals each test so ordering can never leak state.
        self.orig_managed = kokoro_server.MANAGED
        kokoro_server.MANAGED = False

    def tearDown(self):
        kokoro_server.MANAGED = self.orig_managed

    def make_handler(self):
        handler = kokoro_server.TTSHandler.__new__(kokoro_server.TTSHandler)
        handler.rfile = io.BytesIO(b'')
        handler.wfile = FlushableBytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_heartbeat_endpoint_updates_last_seen(self):
        handler = self.make_handler()
        handler.path = '/panel-heartbeat'
        handler.headers = {}
        kokoro_server._heartbeat_at = time.monotonic() - kokoro_server.HEARTBEAT_GRACE
        try:
            kokoro_server.TTSHandler.do_POST(handler)
            self.assertLess(kokoro_server.seconds_since_heartbeat(), 5)
            self.assertIn(b'"ok"', handler.wfile.getvalue())
        finally:
            kokoro_server.touch_heartbeat()

    def test_watchdog_tick_shuts_down_after_grace(self):
        fake_server = MagicMock()
        kokoro_server._heartbeat_at = time.monotonic() - (kokoro_server.HEARTBEAT_GRACE + 5)

        self.assertTrue(kokoro_server._watchdog_tick(fake_server))
        fake_server.shutdown.assert_called_once()

    def test_watchdog_tick_spares_fresh_heartbeat(self):
        fake_server = MagicMock()
        kokoro_server.touch_heartbeat()

        self.assertFalse(kokoro_server._watchdog_tick(fake_server))
        fake_server.shutdown.assert_not_called()

    def test_watchdog_not_started_when_unmanaged(self):
        with patch('kokoro_server.threading.Thread') as thread_ctor:
            kokoro_server.maybe_start_watchdog(MagicMock())
        thread_ctor.assert_not_called()

    def test_watchdog_started_when_managed(self):
        kokoro_server.MANAGED = True
        with patch('kokoro_server.threading.Thread') as thread_ctor:
            kokoro_server.maybe_start_watchdog(MagicMock())
        thread_ctor.assert_called_once()

    def test_stream_event_refreshes_heartbeat(self):
        handler = self.make_handler()
        kokoro_server._heartbeat_at = time.monotonic() - kokoro_server.HEARTBEAT_GRACE
        try:
            kokoro_server.TTSHandler.send_stream_event(handler, {'type': 'text', 'text': 'hi'})
            self.assertLess(kokoro_server.seconds_since_heartbeat(), 5)
        finally:
            kokoro_server.touch_heartbeat()

    def test_health_reports_model_loaded_and_managed(self):
        handler = self.make_handler()
        handler.path = '/health'
        handler.headers = {}
        kokoro_server.MODEL_LOADED = True
        kokoro_server.MANAGED = True
        try:
            kokoro_server.TTSHandler.do_GET(handler)
            body = json.loads(handler.wfile.getvalue())
            self.assertTrue(body['model_loaded'])
            self.assertTrue(body['managed'])
        finally:
            kokoro_server.MODEL_LOADED = False
            kokoro_server.MANAGED = False

    def test_get_vision_api_bases_single_candidate_on_windows(self):
        # Windows never reads /proc/net/route, so only the loopback base remains.
        with patch.object(kokoro_server, '_read_linux_default_gateway', return_value=None), \
             patch.object(kokoro_server, 'VISION_API_BASE', 'http://127.0.0.1:11434'):
            self.assertEqual(kokoro_server.get_vision_api_bases(), ['http://127.0.0.1:11434'])


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
                kokoro_server.TTSHandler.do_POST(handler)
            finally:
                kokoro_server.CONVERSATIONS_DIR = orig_conv_dir
                kokoro_server.CURRENT_PTR = orig_current_ptr


class StartupTimingTests(unittest.TestCase):
    """Tests for startup phase-timing observability."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False

    def test_boot_helper_writes_timestamped_stderr(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            kokoro_server._boot('test message')

        self.assertIn('[BOOT] test message', stderr.getvalue())
        self.assertRegex(stderr.getvalue(), r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    def test_phase_logs_elapsed(self):
        with self.assertLogs(level='INFO') as logs:
            with kokoro_server._phase('x'):
                pass

        self.assertIn('[PHASE] x took', logs.output[-1])
        self.assertRegex(logs.output[-1], r'took \d+ms$')

    def test_phase_logs_failed_shape_when_body_raises(self):
        # A crashing phase must not contribute a success-shaped 'took Nms'
        # line to the startup baseline.
        with self.assertRaises(ValueError), self.assertLogs(level='INFO') as logs:
            with kokoro_server._phase('x'):
                raise ValueError('boom')

        self.assertRegex(logs.output[-1], r'\[PHASE\] x failed after \d+ms$')

    def test_get_pipeline_logs_model_load_ms(self):
        with self.assertLogs(level='INFO') as logs:
            result = kokoro_server.get_pipeline()

        self.assertIsNotNone(result)
        self.assertTrue(kokoro_server.MODEL_LOADED)
        loaded_lines = [line for line in logs.output
                        if re.search(r'Kokoro model loaded on \w+ in \d+ms\.', line)]
        self.assertTrue(loaded_lines, 'expected a "loaded ... in Xms" line')


class DeferredHeavyImportsTests(unittest.TestCase):
    """Issue #6: numpy/soundfile/kokoro must not be imported at module level."""

    def test_module_has_no_heavy_import_attributes(self):
        for attr in ('np', 'sf', 'KPipeline'):
            self.assertFalse(hasattr(kokoro_server, attr),
                             f"'{attr}' should be deferred, not a module attribute")

    def test_text_to_wav_works_with_deferred_imports(self):
        pipe = MagicMock()
        pipe.return_value = iter([(None, None, 'chunk')])
        with patch.object(kokoro_server, 'get_pipeline', return_value=pipe):
            wav = kokoro_server.text_to_wav('Hello.')

        self.assertEqual(wav, b'')


class HfOfflineGatingTests(unittest.TestCase):
    """Issue #9: set HF_HUB_OFFLINE=1 before the kokoro import when the model is cached."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False
        self._env_backup = {k: os.environ.get(k) for k in
                            ('HF_HUB_CACHE', 'HF_HOME', 'HF_HUB_OFFLINE')}
        # StartupTimingTests hit the real warm cache and leave this set.
        os.environ.pop('HF_HUB_OFFLINE', None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False

    def test_hf_cache_dir_prefers_explicit_hf_hub_cache(self):
        with patch.dict(os.environ, {'HF_HUB_CACHE': 'X:/hub', 'HF_HOME': 'Y:/home'}):
            self.assertEqual(kokoro_server._hf_cache_dir(), Path('X:/hub'))

    def test_hf_cache_dir_derives_from_hf_home(self):
        with patch.dict(os.environ, {'HF_HOME': 'Y:/home'}, clear=False):
            os.environ.pop('HF_HUB_CACHE', None)
            self.assertEqual(kokoro_server._hf_cache_dir(), Path('Y:/home/hub'))

    def test_hf_cache_dir_falls_back_to_default_location(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_HUB_CACHE', None)
            os.environ.pop('HF_HOME', None)
            self.assertEqual(kokoro_server._hf_cache_dir(),
                             Path.home() / '.cache' / 'huggingface' / 'hub')

    def _make_cache(self, root: Path, with_pth: bool = True) -> Path:
        snapshot = root / 'models--hexgrad--Kokoro-82M' / 'snapshots' / 'abc123'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        if with_pth:
            (snapshot / 'kokoro-v1_0.pth').write_bytes(b'weights')
        return root

    def test_kokoro_model_cached_true_when_snapshot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_cache(Path(tmp))
            with patch.object(kokoro_server, '_hf_cache_dir', return_value=root):
                self.assertTrue(kokoro_server._kokoro_model_cached())

    def test_kokoro_model_cached_false_when_weights_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_cache(Path(tmp), with_pth=False)
            with patch.object(kokoro_server, '_hf_cache_dir', return_value=root):
                self.assertFalse(kokoro_server._kokoro_model_cached())

    def test_kokoro_model_cached_false_when_cache_missing(self):
        with patch.object(kokoro_server, '_hf_cache_dir',
                          return_value=Path('Z:/does-not-exist/hub')):
            self.assertFalse(kokoro_server._kokoro_model_cached())

    def test_get_pipeline_sets_hf_hub_offline_when_cache_warm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_cache(Path(tmp))
            with patch.object(kokoro_server, '_hf_cache_dir', return_value=root):
                with self.assertLogs(level='INFO') as logs:
                    kokoro_server.get_pipeline()

        self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '1')
        self.assertTrue(any('HF cache warm' in line for line in logs.output))

    def test_get_pipeline_leaves_offline_unset_when_cache_cold(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(kokoro_server, '_hf_cache_dir',
                              return_value=Path(tmp) / 'empty'):
                kokoro_server.get_pipeline()

        self.assertNotIn('HF_HUB_OFFLINE', os.environ)


if __name__ == '__main__':
    unittest.main()