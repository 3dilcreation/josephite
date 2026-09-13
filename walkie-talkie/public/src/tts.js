// Playback for text that arrived in place of audio.
//
// The point of the degraded path is that the receiving end should not feel like
// it switched apps: you still hold the phone to your ear and hear a sentence.
// Each peer keeps a stable synthetic voice so a channel with several people on
// it stays followable by ear alone.

const voiceByPeer = new Map();

export function ttsSupported() {
  return typeof globalThis.speechSynthesis !== 'undefined';
}

export function listVoices() {
  if (!ttsSupported()) return [];
  return speechSynthesis.getVoices();
}

// Voices load asynchronously in Chrome; resolve once the list is populated.
export function voicesReady(timeoutMs = 2000) {
  if (!ttsSupported()) return Promise.resolve([]);
  const existing = speechSynthesis.getVoices();
  if (existing.length) return Promise.resolve(existing);
  return new Promise((resolve) => {
    const done = () => resolve(speechSynthesis.getVoices());
    speechSynthesis.addEventListener('voiceschanged', done, { once: true });
    setTimeout(done, timeoutMs);
  });
}

export function voiceForPeer(peerId, voices = listVoices()) {
  if (!voices.length) return null;
  if (voiceByPeer.has(peerId)) return voiceByPeer.get(peerId);
  const preferred = voices.filter((v) => v.lang.startsWith('en'));
  const pool = preferred.length ? preferred : voices;
  const voice = pool[hash(peerId) % pool.length];
  voiceByPeer.set(peerId, voice);
  return voice;
}

export function speak(text, { peerId = 'unknown', rate = 1.05, onend = () => {} } = {}) {
  if (!ttsSupported() || !text) { onend(); return null; }
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = voiceForPeer(peerId);
  if (voice) { utterance.voice = voice; utterance.lang = voice.lang; }
  utterance.rate = rate;
  utterance.onend = onend;
  utterance.onerror = onend;
  speechSynthesis.speak(utterance);
  return utterance;
}

export function stopSpeaking() {
  if (ttsSupported()) speechSynthesis.cancel();
}

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}
