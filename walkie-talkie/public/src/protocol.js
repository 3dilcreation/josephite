// Wire protocol for peer-to-peer frames sent over WebRTC data channels.
//
// Every frame is a single ArrayBuffer:
//   byte  0        type
//   bytes 1..16    message id (16 random bytes, stable across relays)
//   bytes 17..20   sequence number (uint32 BE, per-message)
//   byte  21       hop count (incremented by each relay)
//   bytes 22..23   meta length (uint16 BE)
//   bytes 24..     meta (UTF-8 JSON), then the binary payload
//
// Keeping the header fixed-width means a relay can bump the hop count and
// forward without parsing the payload at all.

export const TYPE = {
  HELLO: 0x01,
  PING: 0x02,
  PONG: 0x03,
  AUDIO_START: 0x10,
  AUDIO_CHUNK: 0x11,
  AUDIO_END: 0x12,
  TEXT: 0x20,
  VOICE_AS_TEXT: 0x21,
  ACK: 0x30,
};

export const HEADER_BYTES = 24;
export const MAX_HOPS = 4;

const enc = new TextEncoder();
const dec = new TextDecoder();

export function randomId() {
  const b = new Uint8Array(16);
  globalThis.crypto.getRandomValues(b);
  return b;
}

export function idToHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function encode({ type, id, seq = 0, hops = 0, meta = {}, payload = null }) {
  // An empty meta object is encoded as zero bytes rather than "{}", so the
  // frames sent most often (pings, acks) stay at exactly the header size.
  const hasMeta = meta && Object.keys(meta).length > 0;
  const metaBytes = hasMeta ? enc.encode(JSON.stringify(meta)) : new Uint8Array(0);
  if (metaBytes.length > 0xffff) throw new RangeError('meta too large');
  const payloadBytes = toBytes(payload);
  const buf = new Uint8Array(HEADER_BYTES + metaBytes.length + payloadBytes.length);
  const view = new DataView(buf.buffer);

  buf[0] = type;
  buf.set(id, 1);
  view.setUint32(17, seq >>> 0);
  buf[21] = hops & 0xff;
  view.setUint16(22, metaBytes.length);
  buf.set(metaBytes, HEADER_BYTES);
  buf.set(payloadBytes, HEADER_BYTES + metaBytes.length);
  return buf.buffer;
}

export function decode(buffer) {
  const buf = new Uint8Array(buffer);
  if (buf.length < HEADER_BYTES) throw new RangeError('frame shorter than header');
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const metaLen = view.getUint16(22);
  const metaEnd = HEADER_BYTES + metaLen;
  if (metaEnd > buf.length) throw new RangeError('meta length overruns frame');

  return {
    type: buf[0],
    id: buf.slice(1, 17),
    seq: view.getUint32(17),
    hops: buf[21],
    meta: metaLen ? JSON.parse(dec.decode(buf.subarray(HEADER_BYTES, metaEnd))) : {},
    payload: buf.slice(metaEnd),
  };
}

// Bump the hop counter in place. Returns false when the frame has already
// travelled far enough that forwarding it again would just add churn.
// Dedupe key for relay suppression.
//
// The message id alone is not enough: a single transmission deliberately reuses
// one id across AUDIO_START, its chunks and AUDIO_END so a relay can associate
// them, so keying on the id would make every frame after the first look like a
// duplicate of it. Type and sequence are what actually distinguish them.
export function dedupeKey(frame) {
  return `${frame.type}:${idToHex(frame.id)}:${frame.seq}`;
}

export function bumpHops(buffer) {
  const buf = new Uint8Array(buffer);
  const next = buf[21] + 1;
  if (next > MAX_HOPS) return false;
  buf[21] = next;
  return true;
}

function toBytes(payload) {
  if (!payload) return new Uint8Array(0);
  if (payload instanceof Uint8Array) return payload;
  if (payload instanceof ArrayBuffer) return new Uint8Array(payload);
  if (ArrayBuffer.isView(payload)) return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  if (typeof payload === 'string') return enc.encode(payload);
  throw new TypeError('unsupported payload type');
}
