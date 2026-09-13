// Peer mesh over WebRTC data channels.
//
// The signalling server exists only to introduce peers to each other; once the
// handshake completes every frame travels device-to-device, and the server can
// disappear without dropping a conversation. No STUN or TURN is configured on
// purpose: host candidates are all you need on one Wi-Fi or hotspot, and
// reaching for a public STUN server would quietly make an "offline" app depend
// on the internet.

import { TYPE, encode, decode, randomId, idToHex, dedupeKey, bumpHops } from './protocol.js';
import { LinkQuality, TIER } from './link-quality.js';

const PING_INTERVAL_MS = 1500;
const PING_TIMEOUT_MS = 3000;
// Every fourth ping carries padding, so the link gets a capacity number without
// waiting for somebody to press the talk button. Latency alone would happily
// call a link clear that cannot actually move 24 kbps of Opus.
const PROBE_EVERY = 4;
const PROBE_BYTES = 8192;

export class Mesh extends EventTarget {
  constructor({ signalUrl, selfId, displayName, channel = 'main' }) {
    super();
    this.signalUrl = signalUrl;
    this.selfId = selfId;
    this.displayName = displayName;
    this.channel = channel;
    this.peers = new Map();   // peerId -> peer record
    this.seen = new Map();    // message id hex -> timestamp, for relay dedupe
    this.ws = null;
    this.closed = false;
    this.pingRound = 0;
    // Set to a TIER value to pin the grade regardless of measurements. The
    // degraded path is the whole point of the app and is otherwise only
    // reachable by physically walking out of range, which is a poor demo.
    this.forcedTier = null;
  }

  connect() {
    this.ws = new WebSocket(this.signalUrl);
    this.ws.addEventListener('open', () => {
      this.#signal({ kind: 'join', id: this.selfId, name: this.displayName, channel: this.channel });
      this.#emit('signal-state', { state: 'open' });
    });
    this.ws.addEventListener('message', (event) => this.#onSignal(JSON.parse(event.data)));
    this.ws.addEventListener('close', () => {
      this.#emit('signal-state', { state: 'closed' });
      // Peers already connected keep working; only new introductions stop.
      if (!this.closed) setTimeout(() => this.connect(), 2000);
    });
    this.ws.addEventListener('error', () => this.#emit('signal-state', { state: 'error' }));

    this.pingTimer = setInterval(() => this.#pingAll(), PING_INTERVAL_MS);
  }

  close() {
    this.closed = true;
    clearInterval(this.pingTimer);
    for (const peer of this.peers.values()) peer.pc.close();
    this.peers.clear();
    this.ws?.close();
  }

  get peerList() {
    return [...this.peers.values()]
      .filter((p) => p.ctrl?.readyState === 'open')
      .map((p) => ({ id: p.id, name: p.name, ...p.quality.snapshot() }));
  }

  // Worst tier across connected peers: a broadcast is only as good as the peer
  // least able to receive it.
  worstTier() {
    if (this.forcedTier) return this.forcedTier;
    const tiers = this.peerList.map((p) => p.tier);
    if (!tiers.length) return TIER.RED;
    if (tiers.includes(TIER.RED)) return TIER.RED;
    if (tiers.includes(TIER.AMBER)) return TIER.AMBER;
    return TIER.GREEN;
  }

  broadcast(frame, { media = false } = {}) {
    let sent = 0;
    for (const peer of this.peers.values()) {
      const ch = media && peer.media?.readyState === 'open' ? peer.media : peer.ctrl;
      if (ch?.readyState !== 'open') continue;
      // Back-pressure guard: dropping a chunk beats stalling the whole channel.
      if (media && ch.bufferedAmount > 512 * 1024) continue;
      ch.send(frame);
      sent += 1;
    }
    return sent;
  }

  // ---- signalling -------------------------------------------------------

  #signal(msg) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  async #onSignal(msg) {
    switch (msg.kind) {
      case 'joined':
        // The server may have had to disambiguate our id; everything we send
        // from here on has to carry the one it actually routes.
        if (msg.id !== this.selfId) {
          this.selfId = msg.id;
          this.#emit('self-id', { id: msg.id });
        }
        break;
      case 'peers':
        // We are the newcomer, so we make the offers.
        for (const p of msg.peers) if (p.id !== this.selfId) this.#dial(p);
        break;
      case 'peer-joined':
        // Existing members wait to be dialled; this avoids glare.
        this.#ensurePeer(msg.peer.id, msg.peer.name);
        break;
      case 'peer-left':
        this.#dropPeer(msg.id);
        break;
      case 'offer': {
        const peer = this.#ensurePeer(msg.from, msg.name);
        await peer.pc.setRemoteDescription(msg.description);
        const answer = await peer.pc.createAnswer();
        await peer.pc.setLocalDescription(answer);
        this.#signal({ kind: 'answer', to: msg.from, from: this.selfId, description: answer });
        break;
      }
      case 'answer': {
        const peer = this.peers.get(msg.from);
        if (peer) await peer.pc.setRemoteDescription(msg.description);
        break;
      }
      case 'candidate': {
        const peer = this.peers.get(msg.from);
        if (peer && msg.candidate) {
          try { await peer.pc.addIceCandidate(msg.candidate); } catch { /* late candidate */ }
        }
        break;
      }
    }
  }

  async #dial(info) {
    const peer = this.#ensurePeer(info.id, info.name);
    peer.ctrl = this.#wireChannel(peer, peer.pc.createDataChannel('ctrl', { ordered: true }));
    peer.media = this.#wireChannel(
      peer,
      // Audio prefers freshness over completeness: a chunk that arrives late is
      // worth less than the next one arriving on time.
      peer.pc.createDataChannel('media', { ordered: false, maxRetransmits: 0 }),
    );
    const offer = await peer.pc.createOffer();
    await peer.pc.setLocalDescription(offer);
    this.#signal({ kind: 'offer', to: info.id, from: this.selfId, name: this.displayName, description: offer });
  }

  #ensurePeer(id, name) {
    let peer = this.peers.get(id);
    if (peer) {
      if (name) peer.name = name;
      return peer;
    }
    const pc = new RTCPeerConnection({ iceServers: [] });
    peer = {
      id,
      name: name || id.slice(0, 6),
      pc,
      ctrl: null,
      media: null,
      quality: new LinkQuality(),
      pending: new Map(), // ping nonce -> sent timestamp
      inbox: new Map(),   // message id hex -> assembling transmission
    };
    this.peers.set(id, peer);

    pc.addEventListener('icecandidate', (event) => {
      if (event.candidate) {
        this.#signal({ kind: 'candidate', to: id, from: this.selfId, candidate: event.candidate });
      }
    });
    pc.addEventListener('datachannel', (event) => this.#wireChannel(peer, event.channel));
    pc.addEventListener('connectionstatechange', () => {
      if (['failed', 'closed'].includes(pc.connectionState)) this.#dropPeer(id);
      this.#emit('peers', { peers: this.peerList });
    });
    return peer;
  }

  #wireChannel(peer, channel) {
    channel.binaryType = 'arraybuffer';
    if (channel.label === 'ctrl') peer.ctrl = channel;
    if (channel.label === 'media') peer.media = channel;

    channel.addEventListener('open', () => {
      if (channel.label === 'ctrl') {
        channel.send(encode({
          type: TYPE.HELLO,
          id: randomId(),
          meta: { name: this.displayName, from: this.selfId },
        }));
        this.#emit('peers', { peers: this.peerList });
      }
    });
    channel.addEventListener('close', () => this.#emit('peers', { peers: this.peerList }));
    channel.addEventListener('message', (event) => this.#onFrame(peer, event.data));
    return channel;
  }

  #dropPeer(id) {
    const peer = this.peers.get(id);
    if (!peer) return;
    peer.pc.close();
    this.peers.delete(id);
    this.#emit('peers', { peers: this.peerList });
  }

  // ---- frames -----------------------------------------------------------

  #onFrame(peer, data) {
    let frame;
    try { frame = decode(data); } catch { return; }

    switch (frame.type) {
      case TYPE.HELLO:
        peer.name = frame.meta.name || peer.name;
        this.#emit('peers', { peers: this.peerList });
        return;
      case TYPE.PING:
        peer.ctrl?.readyState === 'open' && peer.ctrl.send(encode({
          type: TYPE.PONG, id: frame.id, meta: { nonce: frame.meta.nonce },
        }));
        return;
      case TYPE.PONG: {
        const sent = peer.pending.get(frame.meta.nonce);
        if (sent) {
          peer.pending.delete(frame.meta.nonce);
          const elapsed = performance.now() - sent.at;
          peer.quality.recordPing(true, elapsed);
          // A padded probe also tells us roughly what the link can carry. It is
          // a proxy, not a true one-way measurement, but it is available within
          // seconds of connecting and real audio supersedes it as soon as any
          // arrives.
          if (sent.bytes) peer.quality.recordThroughput(sent.bytes, elapsed);
          this.#emit('quality', { peerId: peer.id, ...peer.quality.snapshot() });
        }
        return;
      }
    }

    // Everything below is relayable content, so dedupe before acting on it.
    const key = dedupeKey(frame);
    if (this.seen.has(key)) return;
    this.#remember(key);

    if (frame.type === TYPE.AUDIO_CHUNK || frame.type === TYPE.AUDIO_START || frame.type === TYPE.AUDIO_END) {
      this.#onAudioFrame(peer, frame);
    } else if (frame.type === TYPE.TEXT || frame.type === TYPE.VOICE_AS_TEXT) {
      this.#emit('message', {
        kind: frame.type === TYPE.VOICE_AS_TEXT ? 'voice-as-text' : 'text',
        from: frame.meta.from || peer.id,
        name: frame.meta.name || peer.name,
        text: new TextDecoder().decode(frame.payload),
        meta: frame.meta,
        at: Date.now(),
      });
    }

    // Forward on, so a peer two rooms away that only one of us can hear still
    // gets the traffic.
    if (bumpHops(data)) {
      for (const other of this.peers.values()) {
        if (other.id === peer.id) continue;
        const ch = frame.type === TYPE.AUDIO_CHUNK ? (other.media || other.ctrl) : other.ctrl;
        if (ch?.readyState === 'open') ch.send(data);
      }
    }
  }

  #onAudioFrame(peer, frame) {
    const idHex = idToHex(frame.id);
    if (frame.type === TYPE.AUDIO_START) {
      peer.inbox.set(idHex, { chunks: [], meta: frame.meta, startedAt: Date.now() });
      this.#emit('audio-start', { from: peer.id, name: peer.name, id: idHex, meta: frame.meta });
      return;
    }
    const entry = peer.inbox.get(idHex);
    if (!entry) return;

    if (frame.type === TYPE.AUDIO_CHUNK) {
      entry.chunks.push(frame.payload);
      this.#emit('audio-chunk', { from: peer.id, id: idHex, chunk: frame.payload, seq: frame.seq });
      return;
    }
    peer.inbox.delete(idHex);
    const bytes = entry.chunks.reduce((n, c) => n + c.length, 0);
    peer.quality.recordThroughput(bytes, Date.now() - entry.startedAt);
    this.#emit('audio-end', {
      from: peer.id,
      name: peer.name,
      id: idHex,
      // The closing frame carries what could only be known once the talker let
      // go -- duration and the transcript -- so it layers over the opening one.
      meta: { ...entry.meta, ...frame.meta, bytesReceived: bytes },
      blob: new Blob(entry.chunks, { type: entry.meta.mimeType || 'audio/webm' }),
    });
  }

  #pingAll() {
    this.pingRound += 1;
    const probe = this.pingRound % PROBE_EVERY === 0;

    for (const peer of this.peers.values()) {
      if (peer.ctrl?.readyState !== 'open') continue;
      // Skip the probe on a channel that is already backed up; adding 8 KB to a
      // struggling link would measure the congestion we just caused.
      const padded = probe && peer.ctrl.bufferedAmount < 64 * 1024;
      const nonce = Math.random().toString(36).slice(2, 10);
      peer.pending.set(nonce, { at: performance.now(), bytes: padded ? PROBE_BYTES : 0 });
      peer.ctrl.send(encode({
        type: TYPE.PING,
        id: randomId(),
        meta: { nonce },
        payload: padded ? new Uint8Array(PROBE_BYTES) : null,
      }));

      setTimeout(() => {
        if (peer.pending.delete(nonce)) {
          peer.quality.recordPing(false);
          this.#emit('quality', { peerId: peer.id, ...peer.quality.snapshot() });
        }
      }, PING_TIMEOUT_MS);
      peer.quality.poll();
    }
    this.#emit('peers', { peers: this.peerList });
  }

  #remember(key) {
    this.seen.set(key, Date.now());
    if (this.seen.size > 512) {
      const cutoff = Date.now() - 60_000;
      for (const [seenKey, at] of this.seen) if (at < cutoff) this.seen.delete(seenKey);
    }
  }

  #emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { detail }));
  }
}
