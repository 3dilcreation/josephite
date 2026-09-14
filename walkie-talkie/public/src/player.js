// Receive-side playback.
//
// Two paths, both ending in sound coming out of the earpiece:
//   - audio clips are queued and played in order, so overlapping talkers do not
//     turn into noise (a walkie-talkie channel is half-duplex by nature)
//   - text that arrived instead of audio is spoken with the sender's assigned
//     voice, so the degraded path still sounds like a person talking
//
// Live streaming through MediaSource is attempted only when the sender marked
// the transmission live; any append error falls back to the clip path, because
// a stall is worse than a second of latency.

import { speak } from './tts.js';

// Ceiling on how long one queued item may hold the channel. Speech synthesis
// that never reports completion -- a device with no installed voice, a page
// backgrounded mid-utterance -- would otherwise wedge the queue permanently and
// the user would simply stop hearing anyone.
const MAX_ITEM_MS = 30_000;

export class Player {
  constructor() {
    this.queue = [];
    this.playing = false;
    this.live = new Map();
  }

  beginLive(id, meta) {
    if (!meta.live || typeof MediaSource === 'undefined') return;
    if (!MediaSource.isTypeSupported?.(meta.mimeType)) return;

    const audio = new Audio();
    const mediaSource = new MediaSource();
    const url = URL.createObjectURL(mediaSource);
    audio.src = url;
    const entry = { audio, mediaSource, url, buffer: null, pending: [], failed: false, nextSeq: 0, jitter: new Map() };
    this.live.set(id, entry);

    mediaSource.addEventListener('sourceopen', () => {
      try {
        entry.buffer = mediaSource.addSourceBuffer(meta.mimeType);
        entry.buffer.addEventListener('updateend', () => this.#drain(entry));
        entry.buffer.addEventListener('error', () => { entry.failed = true; });
        this.#drain(entry);
        audio.play().catch(() => { entry.failed = true; });
      } catch {
        entry.failed = true;
      }
    }, { once: true });
  }

  feedLive(id, seq, chunk) {
    const entry = this.live.get(id);
    if (!entry || entry.failed) return false;
    // Reorder around the unreliable media channel before handing bytes to MSE,
    // which will reject anything out of sequence.
    entry.jitter.set(seq, chunk);
    while (entry.jitter.has(entry.nextSeq)) {
      entry.pending.push(entry.jitter.get(entry.nextSeq));
      entry.jitter.delete(entry.nextSeq);
      entry.nextSeq += 1;
    }
    // A gap we have waited too long for is a lost chunk; skip past it.
    if (entry.jitter.size > 8) {
      entry.nextSeq = Math.min(...entry.jitter.keys());
    }
    this.#drain(entry);
    return true;
  }

  endLive(id) {
    const entry = this.live.get(id);
    if (!entry) return false;
    this.live.delete(id);
    // The element keeps the blob URL alive until it has finished playing out
    // what was appended, so revoking waits for the end of playback.
    entry.audio.addEventListener('ended', () => URL.revokeObjectURL(entry.url), { once: true });
    if (entry.failed) { URL.revokeObjectURL(entry.url); return false; }
    try {
      if (entry.mediaSource.readyState === 'open' && !entry.buffer?.updating) {
        entry.mediaSource.endOfStream();
      }
    } catch { /* already torn down */ }
    return true;
  }

  // The sender gave up on this transmission, so nothing further is coming and
  // what has arrived is a fragment. Release it rather than playing half a word.
  abortLive(id) {
    const entry = this.live.get(id);
    if (!entry) return false;
    this.live.delete(id);
    entry.pending.length = 0;
    entry.jitter.clear();
    try { entry.audio.pause(); } catch { /* never started */ }
    entry.audio.removeAttribute('src');
    entry.audio.load();
    URL.revokeObjectURL(entry.url);
    return true;
  }

  enqueueClip(blob, { onstart = () => {}, onend = () => {} } = {}) {
    this.queue.push({ blob, onstart, onend });
    this.#next();
  }

  enqueueSpoken(text, peerId, { onstart = () => {}, onend = () => {} } = {}) {
    this.queue.push({ text, peerId, onstart, onend });
    this.#next();
  }

  #drain(entry) {
    if (!entry.buffer || entry.buffer.updating || !entry.pending.length || entry.failed) return;
    try {
      entry.buffer.appendBuffer(entry.pending.shift());
    } catch {
      entry.failed = true;
    }
  }

  #next() {
    if (this.playing || !this.queue.length) return;
    const item = this.queue.shift();
    this.playing = true;
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(watchdog);
      this.playing = false;
      item.onend();
      this.#next();
    };
    const watchdog = setTimeout(finish, MAX_ITEM_MS);

    item.onstart();
    if (item.text !== undefined) {
      speak(item.text, { peerId: item.peerId, onend: finish });
      return;
    }
    const url = URL.createObjectURL(item.blob);
    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); finish(); };
    audio.onerror = () => { URL.revokeObjectURL(url); finish(); };
    audio.play().catch(() => { URL.revokeObjectURL(url); finish(); });
  }
}
