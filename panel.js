const SERVER = 'http://127.0.0.1:5912';
let audioCtx = null;
let currentSource = null;
let currentSources = [];
let currentAbortController = null;
let playing = false;

const btnRead = document.getElementById('btn-read');
const statusEl = document.getElementById('status');
const voiceSelect = document.getElementById('voice');
const constraintsEl = document.getElementById('constraints');
const ocrTextEl = document.getElementById('ocr-text');
const errorEl = document.getElementById('error');
const historyToggle = document.getElementById('history-toggle');
const btnNewConv = document.getElementById('btn-new-conv');
const convIndicator = document.getElementById('conv-indicator');

// Restore saved preferences
const savedVoice = localStorage.getItem('tts-voice');
if (savedVoice) voiceSelect.value = savedVoice;
voiceSelect.addEventListener('change', () => {
  localStorage.setItem('tts-voice', voiceSelect.value);
});

const savedConstraints = localStorage.getItem('tts-constraints');
if (savedConstraints) constraintsEl.value = savedConstraints;
constraintsEl.addEventListener('input', () => {
  localStorage.setItem('tts-constraints', constraintsEl.value);
});

// History toggle state
const savedHistory = localStorage.getItem('tts-history-enabled');
if (savedHistory === 'true') historyToggle.checked = true;
historyToggle.addEventListener('change', () => {
  localStorage.setItem('tts-history-enabled', historyToggle.checked);
});

async function updateConvIndicator() {
  try {
    const r = await fetch(`${SERVER}/conversation_state`);
    const d = await r.json();
    const turns = d.turn_count || 0;
    const shortId = d.conversation_id.split('_').slice(-2).join('_');
    convIndicator.textContent = turns > 0 ? `${shortId} (${turns})` : shortId;
    localStorage.setItem('tts-conversation-id', d.conversation_id);
  } catch {
    convIndicator.textContent = '';
  }
}

btnNewConv.addEventListener('click', async () => {
  try {
    const r = await fetch(`${SERVER}/new_conversation`, { method: 'POST' });
    const d = await r.json();
    localStorage.setItem('tts-conversation-id', d.conversation_id);
    await updateConvIndicator();
  } catch {
    // silently fail
  }
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
    // Build request payload with optional constraints
    const payload = { image: base64, voice: voiceSelect.value };
    const constraints = constraintsEl.value.trim();
    if (constraints) {
      payload.constraints = constraints;
    }
    if (historyToggle.checked) {
      payload.history = true;
      const convId = localStorage.getItem('tts-conversation-id');
      if (convId) {
        payload.conversation_id = convId;
      }
    }

    currentAbortController = new AbortController();
    const r = await fetch(`${SERVER}/ocr_tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal,
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(errText || `Server error ${r.status}`);
    }

    btnRead.textContent = '⏳ Waiting for audio...';
    await playStreamingResponse(r, 3000);

  } catch (e) {
    if (e.name === 'AbortError') return;
    showError(e.message);
    stop();
  }
});

async function playStreamingResponse(response, textLimit) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  let ocrText = '';
  let nextStart = 0;
  let activeSources = 0;
  let streamDone = false;
  let totalDuration = 0;

  async function handleEvent(event) {
    if (event.type === 'error') {
      throw new Error(event.error || 'Streaming OCR/TTS failed');
    }
    if (event.type === 'text') {
      ocrText += event.text;
      ocrTextEl.textContent = ocrText.length > textLimit
        ? ocrText.slice(0, textLimit) + '...'
        : ocrText;
      ocrTextEl.style.display = 'block';
      return;
    }
    if (event.type === 'audio') {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        btnRead.classList.add('playing');
        btnRead.disabled = false;
        playing = true;
      }

      const decoded = await audioCtx.decodeAudioData(base64ToArrayBuffer(event.audio));
      const source = audioCtx.createBufferSource();
      source.buffer = decoded;
      source.connect(audioCtx.destination);
      currentSource = source;
      currentSources.push(source);

      const startAt = Math.max(audioCtx.currentTime, nextStart);
      nextStart = startAt + decoded.duration;
      totalDuration += decoded.duration;
      activeSources += 1;
      btnRead.textContent = `⏹ Stop (${Math.round(totalDuration)}s)`;

      source.onended = () => {
        activeSources -= 1;
        currentSources = currentSources.filter((s) => s !== source);
        if (streamDone && activeSources === 0) {
          stop();
        }
      };

      source.start(startAt);
      return;
    }
    if (event.type === 'done') {
      if (event.conversation_id) {
        localStorage.setItem('tts-conversation-id', event.conversation_id);
      }
      if (historyToggle.checked) {
        updateConvIndicator();
      }
      streamDone = true;
      if (activeSources === 0) {
        stop();
      }
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += decoder.decode(value, { stream: true });
    const lines = pending.split('\n');
    pending = lines.pop();
    for (const line of lines) {
      if (line.trim()) {
        await handleEvent(JSON.parse(line));
      }
    }
  }

  pending += decoder.decode();
  if (pending.trim()) {
    await handleEvent(JSON.parse(pending));
  }
}

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
  for (const source of currentSources) { try { source.stop(); } catch {} }
  currentSources = [];
  if (currentAbortController) { currentAbortController.abort(); }
  currentAbortController = null;
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
updateConvIndicator();
