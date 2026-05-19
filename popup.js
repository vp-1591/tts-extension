const SERVER = 'http://127.0.0.1:5912';
let audioCtx = null;
let currentSource = null;
let playing = false;

const btnRead = document.getElementById('btn-read');
const statusEl = document.getElementById('status');
const voiceSelect = document.getElementById('voice');
const ocrTextEl = document.getElementById('ocr-text');
const errorEl = document.getElementById('error');

const savedVoice = localStorage.getItem('tts-voice');
if (savedVoice) voiceSelect.value = savedVoice;
voiceSelect.addEventListener('change', () => {
  localStorage.setItem('tts-voice', voiceSelect.value);
});

async function checkHealth() {
  try {
    const r = await fetch(`${SERVER}/health`);
    const d = await r.json();
    statusEl.textContent = `✓ Server online (${d.vision || '?'})`;
    statusEl.className = 'status ok';
    btnRead.disabled = false;
  } catch {
    statusEl.textContent = '✗ Server offline — start kokoro_server.py';
    statusEl.className = 'status err';
    btnRead.disabled = true;
  }
}

btnRead.addEventListener('click', async () => {
  if (playing) { stop(); return; }
  errorEl.style.display = 'none';
  ocrTextEl.style.display = 'none';

  // 1. Capture screenshot of the current tab
  btnRead.textContent = '📸 Capturing screen...';
  btnRead.disabled = true;

  let dataUrl;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png', quality: 85 });
  } catch (e) {
    showError('Cannot capture screen: ' + e.message);
    resetButton();
    return;
  }

  // 2. Extract base64 data (strip data:image/png;base64, prefix)
  const base64 = dataUrl.split(',')[1];
  btnRead.textContent = '🔍 Running OCR...';

  // 3. Send to server: screenshot -> OCR -> TTS
  try {
    const t0 = Date.now();
    const r = await fetch(`${SERVER}/ocr_tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64, voice: voiceSelect.value }),
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(errText || `Server error ${r.status}`);
    }

    btnRead.textContent = '⏳ Decoding audio...';

    const result = await r.json();
    const ocrText = result.text || '';
    const audioB64 = result.audio || '';

    // Show OCR text
    if (ocrText) {
      ocrTextEl.textContent = ocrText.length > 1500
        ? ocrText.slice(0, 1500) + '...'
        : ocrText;
      ocrTextEl.style.display = 'block';
    }

    // Decode base64 WAV to ArrayBuffer
    const audioBuf = base64ToArrayBuffer(audioB64);

    // Play audio
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(audioBuf);
    const source = audioCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(audioCtx.destination);
    currentSource = source;

    const duration = decoded.duration;
    const elapsed = (Date.now() - t0) / 1000;
    btnRead.textContent = `⏹ Stop (${Math.round(duration)}s)`;
    btnRead.classList.add('playing');
    btnRead.disabled = false;
    playing = true;

    source.onended = () => {
      playing = false;
      resetButton();
      if (audioCtx) { audioCtx.close().catch(() => {}); }
      audioCtx = null;
      currentSource = null;
    };

    source.start(0);

  } catch (e) {
    showError(e.message);
    stop();
  }
});

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function stop() {
  if (currentSource) { try { currentSource.stop(); } catch {} }
  if (audioCtx) { audioCtx.close().catch(() => {}); }
  audioCtx = null;
  currentSource = null;
  playing = false;
  resetButton();
}

function resetButton() {
  btnRead.textContent = '📸 Read My Screen';
  btnRead.classList.remove('playing');
  btnRead.disabled = false;
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = 'block';
}

checkHealth();