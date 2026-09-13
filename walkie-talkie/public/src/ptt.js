// The transmit half of push-to-talk.
//
// The idea this app is built around: capture two representations of every
// transmission at once -- compressed audio and a locally produced transcript --
// and decide which one actually goes on the wire at release time, or mid-hold if
// the link collapses while you are still speaking. Capturing both costs a little
// CPU and nothing in bandwidth, and it means a link that degrades halfway
// through a sentence still delivers the sentence.

import { TYPE, encode, randomId, idToHex } from './protocol.js';
import { TIER, profileFor } from './link-quality.js';
import { createTranscriber } from './stt.js';

const WORKLET_SOURCE = `
class TapProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor('tap-processor', TapProcessor);
`;

export class PushToTalk extends EventTarget {
  constructor({ mesh, selfId, displayName, sttProvider = 'webspeech', alwaysTranscribe = true }) {
    super();
    this.mesh = mesh;
    this.selfId = selfId;
    // mesh.selfId is authoritative once the server has confirmed the join.
    this.displayName = displayName;
    this.sttProvider = sttProvider;
    this.alwaysTranscribe = alwaysTranscribe;
    this.stream = null;
    this.active = null;
  }

  // Asking for the microphone once, up front, keeps the permission prompt out
  // of the moment the user is trying to say something urgent.
  async prime() {
    if (this.stream) return this.stream;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    return this.stream;
  }

  get isTransmitting() {
    return Boolean(this.active);
  }

  async press() {
    if (this.active) return;
    await this.prime();

    const tier = this.mesh.worstTier();
    const profile = profileFor(tier);
    const id = randomId();

    const session = {
      id,
      idHex: idToHex(id),
      tier,
      profile,
      startedAt: Date.now(),
      seq: 0,
      bytesSent: 0,
      transcriber: null,
      recorder: null,
      tap: null,
      sentStart: false,
      aborted: false,
    };
    this.active = session;

    // Text path. Runs alongside audio so the transcript is ready the moment it
    // is needed, rather than started after the link has already failed.
    if (profile.mode === 'transcribe' || this.alwaysTranscribe) {
      try {
        session.transcriber = createTranscriber(this.sttProvider, {
          onInterim: (text) => this.#emit('interim', { text }),
        });
        session.transcriber.begin();
        if (this.sttProvider === 'whisper') session.tap = await this.#startTap(session);
      } catch (err) {
        session.transcriber = null;
        this.#emit('warn', { message: `transcription unavailable: ${err.message}` });
      }
    }

    // Audio path, skipped entirely when the link cannot carry it.
    if (profile.mode !== 'transcribe') this.#startRecorder(session);

    this.#emit('state', { transmitting: true, tier, mode: profile.mode });
  }

  async release() {
    const session = this.active;
    if (!session) return null;
    this.active = null;

    const tierNow = this.mesh.worstTier();
    // A link that fell to RED while the button was held: throw away the audio we
    // were streaming and deliver the transcript instead.
    const collapsed = tierNow === TIER.RED && session.profile.mode !== 'transcribe';

    let transcript = '';
    if (session.transcriber) {
      try { transcript = await session.transcriber.end(); } catch (err) {
        this.#emit('warn', { message: `transcription failed: ${err.message}` });
      }
    }
    session.tap?.stop();

    if (session.recorder && session.recorder.state !== 'inactive') {
      await new Promise((resolve) => {
        session.recorder.addEventListener('stop', resolve, { once: true });
        session.recorder.stop();
      });
    }

    const durationMs = Date.now() - session.startedAt;
    const meta = {
      from: this.mesh.selfId || this.selfId,
      name: this.displayName,
      durationMs,
      tier: session.tier,
    };

    if (session.profile.mode === 'transcribe' || collapsed) {
      if (!transcript) {
        this.#emit('warn', {
          message: collapsed
            ? 'link dropped mid-transmission and nothing could be transcribed'
            : 'link too weak for audio and nothing could be transcribed',
        });
        this.#emit('state', { transmitting: false });
        return null;
      }
      const delivered = this.mesh.broadcast(encode({
        type: TYPE.VOICE_AS_TEXT,
        id: session.id,
        meta: { ...meta, spokenAs: 'text', collapsed, sourceTier: tierNow },
        payload: transcript,
      }));
      this.#emit('sent', { kind: 'voice-as-text', text: transcript, delivered, meta });
      this.#emit('state', { transmitting: false });
      return { kind: 'voice-as-text', text: transcript, delivered };
    }

    if (session.sentStart) {
      this.mesh.broadcast(encode({
        type: TYPE.AUDIO_END,
        id: session.id,
        seq: session.seq,
        meta: { ...meta, transcript, bytes: session.bytesSent },
      }));
    }
    this.#emit('sent', {
      kind: 'audio',
      bytes: session.bytesSent,
      transcript,
      durationMs,
      meta,
    });
    this.#emit('state', { transmitting: false });
    return { kind: 'audio', bytes: session.bytesSent, transcript };
  }

  #startRecorder(session) {
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(this.stream, {
      mimeType,
      audioBitsPerSecond: session.profile.audioBitsPerSecond,
    });
    session.recorder = recorder;

    recorder.addEventListener('dataavailable', async (event) => {
      if (!event.data.size || session.aborted) return;
      const buf = new Uint8Array(await event.data.arrayBuffer());

      if (!session.sentStart) {
        session.sentStart = true;
        this.mesh.broadcast(encode({
          type: TYPE.AUDIO_START,
          id: session.id,
          meta: {
            from: this.mesh.selfId || this.selfId,
            name: this.displayName,
            mimeType,
            tier: session.tier,
            live: session.profile.live,
          },
        }));
      }

      session.bytesSent += buf.length;
      this.mesh.broadcast(encode({
        type: TYPE.AUDIO_CHUNK,
        id: session.id,
        seq: session.seq++,
        payload: buf,
      }), { media: session.profile.live });
    });

    recorder.start(session.profile.timesliceMs);
  }

  // Raw PCM tap for the on-device recogniser, which needs samples rather than
  // an encoded stream.
  async #startTap(session) {
    const ctx = new (globalThis.AudioContext || globalThis.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(this.stream);

    try {
      const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const node = new AudioWorkletNode(ctx, 'tap-processor');
      node.port.onmessage = (event) => session.transcriber?.feed(event.data, ctx.sampleRate);
      source.connect(node);
      // Keep the graph alive without routing microphone audio back to the speaker.
      node.connect(ctx.destination);
      return { stop: () => { node.disconnect(); source.disconnect(); ctx.close(); } };
    } catch {
      // Older WebViews: ScriptProcessorNode is deprecated but still present.
      const node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (event) => {
        session.transcriber?.feed(event.inputBuffer.getChannelData(0), ctx.sampleRate);
      };
      source.connect(node);
      node.connect(ctx.destination);
      return { stop: () => { node.disconnect(); source.disconnect(); ctx.close(); } };
    }
  }

  #emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { detail }));
  }
}

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const type of candidates) {
    if (globalThis.MediaRecorder?.isTypeSupported?.(type)) return type;
  }
  return '';
}
