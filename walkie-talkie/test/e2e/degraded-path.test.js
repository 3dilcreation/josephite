// End-to-end coverage of the thing this app exists for: what actually crosses
// the gap when the link cannot carry audio. Two real browser peers, real
// WebRTC, real MediaRecorder; only the speech engines are doubles.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startServer, launchBrowser, openPeer, waitForPeer, hold, entries } from './harness.js';

let server;
let browser;

test.before(async () => {
  server = await startServer();
  browser = await launchBrowser();
});

test.after(async () => {
  await browser?.close();
  await server?.stop();
});

let channelSeq = 0;

// Each test gets its own channel so peers left over from an earlier one cannot
// satisfy a wait or absorb a broadcast.
async function pair(t, { transcript = 'move up to the ridge', senderQuery = '' } = {}) {
  const channel = `t${channelSeq++}`;
  const sender = await openPeer(browser, { name: 'Alice', channel, transcript, query: senderQuery, origin: server.origin });
  const receiver = await openPeer(browser, { name: 'Bob', channel, origin: server.origin });
  t.after(async () => { await sender.__close(); await receiver.__close(); });

  await waitForPeer(sender, 'Bob');
  await waitForPeer(receiver, 'Alice');
  // Let the padded probe land so the grade reflects a measurement.
  await sender.waitForTimeout(5000);
  return { sender, receiver };
}

test('a clear link carries voice as audio', async (t) => {
  const { sender, receiver } = await pair(t);

  assert.match(await sender.textContent('#tier-label'), /CLEAR/);
  await hold(sender, 1500);
  await receiver.waitForTimeout(2500);

  const received = (await entries(receiver)).filter((e) => e.kind === 'audio');
  assert.equal(received.length, 1, 'receiver logged one voice transmission');
  assert.match(received[0].badge, /^voice \d+\.\ds · \d+ KB/);
  assert.equal(received[0].who, 'Alice');

  // Audio was carried, so nothing should have been read aloud.
  assert.deepEqual(await receiver.evaluate(() => window.__spoken), []);
});

test('a link that cannot carry audio sends the transcript and speaks it', async (t) => {
  const line = 'bring the rope, north gate';
  const { sender, receiver } = await pair(t, { transcript: line, senderQuery: '?tier=red' });

  assert.match(await sender.textContent('#tier-label'), /POOR/);
  await hold(sender, 1200);
  await receiver.waitForTimeout(2500);

  const received = (await entries(receiver)).filter((e) => e.kind === 'voice-as-text');
  assert.equal(received.length, 1, 'receiver logged one degraded transmission');
  assert.equal(received[0].body, line, 'the words survive the downgrade');
  assert.match(received[0].badge, /spoken → text/);

  // The point of the degraded path: the far end still hears a voice.
  const spoken = await receiver.evaluate(() => window.__spoken);
  assert.equal(spoken.length, 1);
  assert.equal(spoken[0].text, line);
  assert.ok(spoken[0].voice, 'the sender was assigned a synthetic voice');

  // No audio should have been attempted at all.
  const audio = (await entries(receiver)).filter((e) => e.kind === 'audio');
  assert.equal(audio.length, 0);
});

test('a link that collapses mid-sentence still delivers the sentence', async (t) => {
  const line = 'we are pinned down at the bridge';
  const { sender, receiver } = await pair(t, { transcript: line });

  assert.match(await sender.textContent('#tier-label'), /CLEAR/);

  // Start transmitting on a clear link, then drop the grade while the button is
  // still held -- the case the dual-path capture exists for.
  await hold(sender, 1500, async () => {
    await sender.waitForTimeout(600);
    await sender.evaluate(() => { window.__mesh.forcedTier = 'red'; });
  });
  await receiver.waitForTimeout(3000);

  const degraded = (await entries(receiver)).filter((e) => e.kind === 'voice-as-text');
  assert.equal(degraded.length, 1, 'the sentence arrived as text');
  assert.equal(degraded[0].body, line);
  assert.match(degraded[0].badge, /link dropped → text/);

  const spoken = await receiver.evaluate(() => window.__spoken);
  assert.deepEqual(spoken.map((s) => s.text), [line]);
});

test('an abandoned transmission is not left half-open at the receiver', async (t) => {
  const { sender, receiver } = await pair(t, { transcript: 'abandon this one' });

  await hold(sender, 1500, async () => {
    await sender.waitForTimeout(600);
    await sender.evaluate(() => { window.__mesh.forcedTier = 'red'; });
  });
  await receiver.waitForTimeout(3000);

  // The partial audio the receiver had already started assembling has to be
  // torn down, or every collapse leaks a buffer and a media element.
  const dangling = await receiver.evaluate(() => ({
    inbox: [...window.__mesh.peers.values()].reduce((n, p) => n + p.inbox.size, 0),
    live: window.__player.live.size,
  }));
  assert.deepEqual(dangling, { inbox: 0, live: 0 });
});
