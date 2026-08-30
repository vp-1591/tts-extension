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


# Test doubles for the heavy deps, faithful to their real APIs:
# - kokoro.pipeline.KPipeline.__call__ (pipeline.py:351-371) takes
#   (text, voice=..., speed=1) and raises ValueError when voice is None;
#   it yields objects iterable as (graphemes, phonemes, audio).
# - soundfile.write(data, samplerate, **kwargs) is recorded but writes
#   nothing, so the WAV buffer stays byte-inert (b'').
PIPE_CALLS = []  # (text, voice) observed at the pipe boundary; reset per test
SF_WRITES = []  # (data, samplerate, kwargs) from sf.write; reset per test


class FakePipeline:
    def __init__(self, lang_code, repo_id=None, model=True, device=None):
        pass

    def __call__(self, text, voice=None, speed=1):
        if voice is None:  # mirrors kokoro.pipeline.KPipeline.__call__
            raise ValueError('Specify a voice')
        PIPE_CALLS.append((text, voice))
        return iter([('Hello.', 'hˈɛloʊ', [0.0, 0.1])])


sys.modules.setdefault('numpy', types.SimpleNamespace(concatenate=lambda chunks: chunks))
sys.modules.setdefault('soundfile', types.SimpleNamespace(
    write=lambda file, data, samplerate, **kwargs: SF_WRITES.append((data, samplerate, kwargs))))
sys.modules.setdefault('kokoro', types.SimpleNamespace(KPipeline=FakePipeline))

kokoro_server = importlib.import_module('kokoro_server')


class FlushableBytesIO(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1


class SingleVoiceTestBase(unittest.TestCase):
    """Runs through the real get_pipeline() -> FakePipeline path (never mocks
    get_pipeline/text_to_wav). _hf_cache_dir is pointed at an empty dir so
    get_pipeline cannot read the warm machine's cache and flip HF_HUB_OFFLINE."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False
        # Warmup adds its own pipe call; TtsWarmupTests covers it, these tests
        # only want the request boundary.
        self._warmup = kokoro_server.TTS_WARMUP
        kokoro_server.TTS_WARMUP = False
        PIPE_CALLS.clear()
        SF_WRITES.clear()
        self._tmp = tempfile.mkdtemp()
        patcher = patch.object(kokoro_server, '_hf_cache_dir', return_value=Path(self._tmp))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def tearDown(self):
        kokoro_server.TTS_WARMUP = self._warmup
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False

    def make_handler(self):
        # Same construction as ManagedLifetimeTests.make_handler: a bare
        # handler instance around buffered streams.
        handler = kokoro_server.TTSHandler.__new__(kokoro_server.TTSHandler)
        handler.rfile = io.BytesIO(b'')
        handler.wfile = FlushableBytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler


class SingleVoiceTests(SingleVoiceTestBase):
    """The server pins af_bella; a request-level voice must have no surface."""

    def test_text_to_wav_uses_default_voice(self):
        kokoro_server.text_to_wav('Hello.')

        self.assertEqual(PIPE_CALLS, [('Hello.', 'af_bella')])
        # numpy.concatenate returns its input, so the recorded data is the
        # per-chunk audio list the pipeline yielded.
        data, samplerate, kwargs = SF_WRITES[0]
        self.assertEqual(data, [[0.0, 0.1]])
        self.assertEqual(samplerate, kokoro_server.SAMPLE_RATE)
        self.assertEqual(kwargs, {'format': 'WAV'})

    def test_tts_handler_ignores_voice_field(self):
        handler = self.make_handler()
        body = json.dumps({'text': 'hi', 'voice': 'am_adam'}).encode()
        handler.path = '/tts'
        handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = io.BytesIO(body)

        kokoro_server.TTSHandler.do_POST(handler)

        self.assertEqual(PIPE_CALLS, [('hi', 'af_bella')])


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
            kokoro_server.TTSHandler.handle_ocr_tts(handler, {'image': 'aW1n'})

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
        # Point _hf_cache_dir at an empty dir so _kokoro_model_cached() runs
        # for real (returns False) but the warm machine's cache can't flip
        # HF_HUB_OFFLINE=1 into the test-process env; warmup is off so the
        # load-only path is what's timed. See TtsWarmupTests for the warmup.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with patch.object(kokoro_server, '_hf_cache_dir', return_value=Path(tmp)), \
             patch.object(kokoro_server, 'TTS_WARMUP', False), \
             self.assertLogs(level='INFO') as logs:
            result = kokoro_server.get_pipeline()

        self.assertIsNotNone(result)
        self.assertTrue(kokoro_server.MODEL_LOADED)
        loaded_lines = [line for line in logs.output
                        if re.search(r'Kokoro model loaded on \w+ in \d+ms\.', line)]
        self.assertTrue(loaded_lines, 'expected a "loaded ... in Xms" line')


class DeferredHeavyImportsTests(SingleVoiceTestBase):
    """Issue #6: numpy/soundfile/kokoro must not be imported at module level."""

    def test_module_has_no_heavy_import_attributes(self):
        for attr in ('np', 'sf', 'KPipeline'):
            self.assertFalse(hasattr(kokoro_server, attr),
                             f"'{attr}' should be deferred, not a module attribute")

    def test_text_to_wav_works_with_deferred_imports(self):
        # The function-level imports must bind to the stub modules through the
        # full real path, not a mocked get_pipeline.
        kokoro_server.text_to_wav('Hello.')

        self.assertEqual(PIPE_CALLS, [('Hello.', 'af_bella')])
        self.assertEqual(len(SF_WRITES), 1)


class BrokenDependencyTests(unittest.TestCase):
    """Issue #2: a broken heavy dependency must fail model load (MODEL_LOADED
    stays false) instead of being swallowed by the warmup. The real import
    machinery is the boundary — a None entry in sys.modules makes both import
    statements and importlib.import_module raise a genuine ImportError."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False
        # Empty cache dir so the warm machine's cache can't flip HF_HUB_OFFLINE
        # into the test-process env (see TtsWarmupTests.setUp, same patch).
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        patcher = patch.object(kokoro_server, '_hf_cache_dir', return_value=Path(self._tmp))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False

    def test_get_pipeline_propagates_broken_dependency(self):
        for broken in ('soundfile', 'numpy'):
            with self.subTest(module=broken), \
                 patch.dict(sys.modules, {broken: None}), \
                 self.assertRaises(ImportError):
                kokoro_server.get_pipeline()
            self.assertFalse(kokoro_server.MODEL_LOADED)

    def test_load_model_logs_traceback_on_failure(self):
        with patch.dict(sys.modules, {'kokoro': None}), \
             self.assertLogs(level='INFO') as logs:
            kokoro_server.load_model()  # must not raise

        self.assertTrue(any('Model load failed' in line for line in logs.output))
        failed = [r for r in logs.records if 'Model load failed' in r.getMessage()]
        self.assertTrue(failed)
        self.assertIsNotNone(failed[0].exc_info, 'expected a real traceback')


class HfOfflineGatingTests(unittest.TestCase):
    """Issue #9: set HF_HUB_OFFLINE=1 before the kokoro import when the model is cached."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False
        self._env_backup = {k: os.environ.get(k) for k in
                            ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME',
                             'XDG_CACHE_HOME', 'HF_HUB_OFFLINE')}
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
            os.environ.pop('HUGGINGFACE_HUB_CACHE', None)
            self.assertEqual(kokoro_server._hf_cache_dir(), Path('X:/hub'))

    def test_hf_cache_dir_derives_from_hf_home(self):
        with patch.dict(os.environ, {'HF_HOME': 'Y:/home'}, clear=False):
            os.environ.pop('HF_HUB_CACHE', None)
            os.environ.pop('HUGGINGFACE_HUB_CACHE', None)
            self.assertEqual(kokoro_server._hf_cache_dir(), Path('Y:/home/hub'))

    def test_hf_cache_dir_falls_back_to_default_location(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_HUB_CACHE', None)
            os.environ.pop('HUGGINGFACE_HUB_CACHE', None)
            os.environ.pop('HF_HOME', None)
            self.assertEqual(kokoro_server._hf_cache_dir(),
                             Path.home() / '.cache' / 'huggingface' / 'hub')

    def test_hf_cache_dir_uses_legacy_huggingface_hub_cache(self):
        # HUGGINGFACE_HUB_CACHE sits between HF_HUB_CACHE and HF_HOME in the
        # huggingface_hub precedence chain and gets the same expansion.
        with patch.dict(os.environ, {'HUGGINGFACE_HUB_CACHE': 'X:/legacy'}):
            os.environ.pop('HF_HUB_CACHE', None)
            self.assertEqual(kokoro_server._hf_cache_dir(), Path('X:/legacy'))

    def test_hf_cache_dir_expands_tilde_in_legacy_cache(self):
        with patch.dict(os.environ, {'HUGGINGFACE_HUB_CACHE': '~/legacy'}):
            os.environ.pop('HF_HUB_CACHE', None)
            self.assertEqual(kokoro_server._hf_cache_dir(), Path.home() / 'legacy')

    def test_hf_cache_dir_derives_from_xdg_cache_home(self):
        with patch.dict(os.environ, {'XDG_CACHE_HOME': 'Y:/xdg'}, clear=False):
            os.environ.pop('HF_HUB_CACHE', None)
            os.environ.pop('HUGGINGFACE_HUB_CACHE', None)
            os.environ.pop('HF_HOME', None)
            self.assertEqual(kokoro_server._hf_cache_dir(),
                             Path('Y:/xdg/huggingface/hub'))

    def test_hf_cache_dir_expands_tilde_in_hf_hub_cache(self):
        with patch.dict(os.environ, {'HF_HUB_CACHE': '~/hfcache'}):
            self.assertEqual(kokoro_server._hf_cache_dir(), Path.home() / 'hfcache')

    def _make_cache(self, root: Path, with_pth: bool = True) -> Path:
        slug = f"models--{kokoro_server.MODEL_REPO_ID.replace('/', '--')}"
        snapshot = root / slug / 'snapshots' / 'abc123'
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


class BackgroundTasksTests(unittest.TestCase):
    """Issue #7: ollama bring-up must not serialize behind the Kokoro model load."""

    def test_start_background_tasks_starts_two_threads(self):
        threads = []
        def fake_thread(**kw):
            # A bare namespace, not a MagicMock: 'name' is reserved on mocks
            # and kwargs are not exposed via __getitem__.
            ns = types.SimpleNamespace(start=MagicMock(), **kw)
            threads.append(ns)
            return ns
        with patch.object(kokoro_server.threading, 'Thread', side_effect=fake_thread):
            kokoro_server.start_background_tasks()

        self.assertEqual(len(threads), 2)
        loader, ollama = threads
        self.assertEqual(loader.target, kokoro_server.load_model)
        self.assertEqual(loader.name, 'model-loader')
        self.assertTrue(loader.daemon)
        self.assertEqual(ollama.target, kokoro_server.ensure_ollama)
        self.assertEqual(ollama.name, 'ollama-ensure')
        self.assertTrue(ollama.daemon)

    def test_ensure_ollama_runs_phase(self):
        with patch.object(kokoro_server, 'ensure_ollama_running') as run:
            with self.assertLogs(level='INFO') as logs:
                kokoro_server.ensure_ollama()

        run.assert_called_once_with()
        self.assertTrue(any('[PHASE] ollama ensure took' in line for line in logs.output))


class BrokenPipeline:
    """KPipeline that constructs fine but fails at synthesis time."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text, voice=None, speed=1):
        raise RuntimeError('boom')


class TtsWarmupTests(unittest.TestCase):
    """Issue #8: get_pipeline() exercises the model with a throwaway request
    before its first real use. The warmup runs inside get_pipeline under
    pipeline_lock, so each test drives the real get_pipeline; only _hf_cache_dir
    and TTS_WARMUP are patched."""

    def setUp(self):
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False
        PIPE_CALLS.clear()
        SF_WRITES.clear()
        self._warmup = kokoro_server.TTS_WARMUP
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def tearDown(self):
        kokoro_server.TTS_WARMUP = self._warmup
        kokoro_server.pipeline = None
        kokoro_server.MODEL_LOADED = False

    def _point_cache_at(self, path):
        patcher = patch.object(kokoro_server, '_hf_cache_dir', return_value=Path(path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_pipeline_warms_up_when_enabled(self):
        self._point_cache_at(self._tmp)
        with self.assertLogs(level='INFO') as logs:
            kokoro_server.get_pipeline()

        self.assertTrue(any('[PHASE] TTS warmup took' in line for line in logs.output))
        self.assertTrue(kokoro_server.MODEL_LOADED)
        # PIPE_CALLS observes at the pipe boundary: if the warmup were ever
        # re-routed through text_to_wav it would re-enter get_pipeline on the
        # held non-reentrant lock and hang this test to its timeout.
        self.assertIn(('Hello.', 'af_bella'), PIPE_CALLS)

    def test_get_pipeline_skips_warmup_when_disabled(self):
        kokoro_server.TTS_WARMUP = False
        self._point_cache_at(self._tmp)
        with self.assertLogs(level='INFO') as logs:
            kokoro_server.get_pipeline()

        self.assertFalse(any('TTS warmup' in line for line in logs.output))
        self.assertEqual(PIPE_CALLS, [])

    def test_get_pipeline_survives_warmup_failure(self):
        self._point_cache_at(self._tmp)
        with patch.dict(sys.modules, {'kokoro': types.SimpleNamespace(KPipeline=BrokenPipeline)}), \
             self.assertLogs(level='INFO') as logs:
            kokoro_server.get_pipeline()  # must not raise

        self.assertTrue(any('TTS warmup failed' in line for line in logs.output))
        self.assertTrue(kokoro_server.MODEL_LOADED)
        self.assertIsNotNone(kokoro_server.pipeline)


if __name__ == '__main__':
    unittest.main()