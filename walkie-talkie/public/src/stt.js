// Speech-to-text, used only when the link is too poor to carry audio.
//
// Two backends behind one interface:
//
//   webspeech - SpeechRecognition. Present on Chrome for Android, zero install.
//               Caveat worth knowing: on most Chrome builds this ships audio to
//               Google's servers, so it is NOT usable once you are genuinely off
//               the network. It is the convenient default, not the offline one.
//
//   whisper   - whisper-tiny running in WASM via transformers.js. The model is
//               fetched once (~40 MB) and then lives in the browser cache, so
//               every later transcription is fully on-device. This is the
//               backend that actually holds up with no infrastructure.
//
// Both expose begin() / feed() / end(). feed() is ignored by the webspeech
// backend, which opens its own microphone stream.

export const WHISPER_MODEL = 'Xenova/whisper-tiny.en';
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

export function sttSupport() {
  const SR = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
  return {
    webspeech: Boolean(SR),
    whisper: typeof WebAssembly === 'object',
  };
}

export function createTranscriber(provider, options = {}) {
  if (provider === 'whisper') return new WhisperTranscriber(options);
  return new WebSpeechTranscriber(options);
}

class WebSpeechTranscriber {
  constructor({ lang = 'en-US', onInterim = () => {} } = {}) {
    const SR = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
    if (!SR) throw new Error('SpeechRecognition unavailable in this browser');
    this.lang = lang;
    this.onInterim = onInterim;
    this.SR = SR;
  }

  begin() {
    this.finalText = '';
    this.rec = new this.SR();
    this.rec.lang = this.lang;
    this.rec.interimResults = true;
    this.rec.continuous = true;
    this.settled = new Promise((resolve) => { this.resolve = resolve; });

    this.rec.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const alt = event.results[i][0];
        if (event.results[i].isFinal) this.finalText += alt.transcript;
        else interim += alt.transcript;
      }
      this.onInterim((this.finalText + interim).trim());
    };
    this.rec.onerror = (event) => { this.error = event.error; };
    this.rec.onend = () => this.resolve(this.finalText.trim());
    this.rec.start();
  }

  feed() { /* backend captures its own audio */ }

  async end() {
    if (!this.rec) return '';
    this.rec.stop();
    const text = await this.settled;
    this.rec = null;
    if (!text && this.error) throw new Error(`speech recognition failed: ${this.error}`);
    return text;
  }
}

class WhisperTranscriber {
  constructor({ onInterim = () => {}, model = WHISPER_MODEL } = {}) {
    this.onInterim = onInterim;
    this.model = model;
    this.chunks = [];
    this.frames = 0;
  }

  begin() {
    this.chunks = [];
    this.frames = 0;
  }

  feed(float32, sampleRate) {
    // Store at the incoming rate; resampling happens once at the end so the
    // hot path stays a plain array push.
    this.chunks.push({ data: Float32Array.from(float32), sampleRate });
    this.frames += float32.length;
    if (this.frames % 16000 < float32.length) this.onInterim('listening…');
  }

  async end() {
    if (!this.chunks.length) return '';
    const pcm = resampleTo16k(this.chunks);
    const pipe = await loadWhisper(this.model);
    const out = await pipe(pcm, { chunk_length_s: 30, return_timestamps: false });
    this.chunks = [];
    return (out?.text || '').trim();
  }
}

let whisperPromise = null;

// Resolves the transformers.js pipeline, downloading the model on first use.
// Exposed so the UI can offer an explicit "download voice pack" button while
// the device still has a connection.
export function loadWhisper(model = WHISPER_MODEL) {
  if (!whisperPromise) {
    whisperPromise = (async () => {
      const mod = await import(/* @vite-ignore */ `${TRANSFORMERS_URL}`);
      mod.env.allowLocalModels = false;
      return mod.pipeline('automatic-speech-recognition', model);
    })().catch((err) => {
      whisperPromise = null;
      throw err;
    });
  }
  return whisperPromise;
}

// Linear resample of the captured buffers down to the 16 kHz mono whisper wants.
export function resampleTo16k(chunks, target = 16000) {
  const sourceRate = chunks[0].sampleRate;
  const total = chunks.reduce((n, c) => n + c.data.length, 0);
  const joined = new Float32Array(total);
  let at = 0;
  for (const c of chunks) { joined.set(c.data, at); at += c.data.length; }
  if (sourceRate === target) return joined;

  const ratio = sourceRate / target;
  const outLength = Math.floor(joined.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const low = Math.floor(pos);
    const high = Math.min(low + 1, joined.length - 1);
    const frac = pos - low;
    out[i] = joined[low] * (1 - frac) + joined[high] * frac;
  }
  return out;
}
