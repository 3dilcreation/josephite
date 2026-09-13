import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TYPE, HEADER_BYTES, MAX_HOPS, encode, decode, randomId, idToHex, bumpHops,
  dedupeKey,
} from '../public/src/protocol.js';

test('round-trips a text frame', () => {
  const id = randomId();
  const frame = encode({
    type: TYPE.TEXT,
    id,
    seq: 42,
    meta: { from: 'abc', name: 'Ravi' },
    payload: 'meet at the gate',
  });
  const out = decode(frame);

  assert.equal(out.type, TYPE.TEXT);
  assert.equal(idToHex(out.id), idToHex(id));
  assert.equal(out.seq, 42);
  assert.equal(out.hops, 0);
  assert.deepEqual(out.meta, { from: 'abc', name: 'Ravi' });
  assert.equal(new TextDecoder().decode(out.payload), 'meet at the gate');
});

test('round-trips binary audio payloads unchanged', () => {
  const audio = new Uint8Array(1024);
  for (let i = 0; i < audio.length; i++) audio[i] = (i * 7) & 0xff;
  const out = decode(encode({ type: TYPE.AUDIO_CHUNK, id: randomId(), seq: 3, payload: audio }));
  assert.deepEqual(out.payload, audio);
});

test('empty meta and payload stay at the header size', () => {
  const frame = encode({ type: TYPE.PING, id: randomId() });
  assert.equal(frame.byteLength, HEADER_BYTES);
  const out = decode(frame);
  assert.deepEqual(out.meta, {});
  assert.equal(out.payload.length, 0);
});

test('bumpHops increments in place and stops at the limit', () => {
  const frame = encode({ type: TYPE.TEXT, id: randomId(), payload: 'x' });
  for (let i = 1; i <= MAX_HOPS; i++) {
    assert.equal(bumpHops(frame), true);
    assert.equal(decode(frame).hops, i);
  }
  assert.equal(bumpHops(frame), false, 'refuses to forward past MAX_HOPS');
  assert.equal(decode(frame).hops, MAX_HOPS, 'hop count is left untouched on refusal');
});

test('rejects frames shorter than a header', () => {
  assert.throws(() => decode(new Uint8Array(10).buffer), RangeError);
});

test('rejects a meta length that overruns the frame', () => {
  const frame = new Uint8Array(encode({ type: TYPE.TEXT, id: randomId(), payload: 'hi' }));
  new DataView(frame.buffer).setUint16(22, 9999);
  assert.throws(() => decode(frame.buffer), RangeError);
});

test('dedupe keys separate the frames of one transmission', () => {
  // Regression: a transmission reuses one message id across START, its chunks
  // and END so relays can group them. Keying relay suppression on the id alone
  // made every frame after START look like a duplicate, and the receiver never
  // saw the END that completes the clip.
  const id = randomId();
  const start = decode(encode({ type: TYPE.AUDIO_START, id, meta: { mimeType: 'audio/webm' } }));
  const chunk0 = decode(encode({ type: TYPE.AUDIO_CHUNK, id, seq: 0, payload: new Uint8Array([1]) }));
  const chunk1 = decode(encode({ type: TYPE.AUDIO_CHUNK, id, seq: 1, payload: new Uint8Array([2]) }));
  const end = decode(encode({ type: TYPE.AUDIO_END, id, seq: 2 }));

  const keys = [start, chunk0, chunk1, end].map(dedupeKey);
  assert.equal(new Set(keys).size, 4, 'every frame of a transmission is distinct');
});

test('dedupe keys still collapse a genuinely repeated frame', () => {
  const id = randomId();
  const once = encode({ type: TYPE.TEXT, id, payload: 'hello' });
  const relayed = encode({ type: TYPE.TEXT, id, payload: 'hello' });
  bumpHops(relayed); // a relay changes the hop count but not the identity
  assert.equal(dedupeKey(decode(once)), dedupeKey(decode(relayed)));
});
