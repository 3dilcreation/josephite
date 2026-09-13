import test from 'node:test';
import assert from 'node:assert/strict';
import { LinkQuality, TIER, profileFor } from '../public/src/link-quality.js';

// A controllable clock, so staleness is tested without real waiting.
function clock(start = 1_000_000) {
  let t = start;
  return { now: () => t, advance: (ms) => { t += ms; } };
}

function feed(link, { count, rtt, bytes, elapsed, answered = true }) {
  for (let i = 0; i < count; i++) {
    link.recordPing(answered, rtt);
    if (bytes) link.recordThroughput(bytes, elapsed);
  }
}

test('a healthy link settles on GREEN', () => {
  const link = new LinkQuality();
  feed(link, { count: 8, rtt: 40, bytes: 20000, elapsed: 1000 });
  assert.equal(link.tier, TIER.GREEN);
});

test('a slow but usable link settles on AMBER', () => {
  const link = new LinkQuality();
  feed(link, { count: 8, rtt: 400, bytes: 3000, elapsed: 1000 });
  assert.equal(link.tier, TIER.AMBER);
});

test('heavy loss drives the link to RED', () => {
  const link = new LinkQuality();
  feed(link, { count: 10, rtt: 50, bytes: 20000, elapsed: 1000 });
  assert.equal(link.tier, TIER.GREEN);
  for (let i = 0; i < 10; i++) link.recordPing(false);
  assert.equal(link.tier, TIER.RED);
});

test('downgrades faster than it upgrades', () => {
  const link = new LinkQuality({ downgradeSamples: 2, upgradeSamples: 4 });
  feed(link, { count: 8, rtt: 40, bytes: 20000, elapsed: 1000 });
  assert.equal(link.tier, TIER.GREEN);

  // Two bad samples are enough to leave GREEN.
  feed(link, { count: 2, rtt: 900, bytes: 200, elapsed: 1000 });
  assert.equal(link.tier, TIER.AMBER);

  // Climbing back needs more evidence than falling did: the same two samples,
  // now good, must not be enough.
  feed(link, { count: 2, rtt: 20, bytes: 60000, elapsed: 1000 });
  assert.equal(link.tier, TIER.AMBER, 'two good samples do not restore GREEN');

  feed(link, { count: 4, rtt: 20, bytes: 60000, elapsed: 1000 });
  assert.equal(link.tier, TIER.GREEN);
});

test('a single outlier sample does not flip the tier', () => {
  const link = new LinkQuality();
  feed(link, { count: 10, rtt: 40, bytes: 30000, elapsed: 1000 });
  link.recordPing(false);
  assert.equal(link.tier, TIER.GREEN, 'one dropped ping is noise, not a state change');
});

test('a silent link decays to RED without any new samples', () => {
  const c = clock();
  const link = new LinkQuality({ now: c.now, sampleTimeoutMs: 8000 });
  feed(link, { count: 8, rtt: 40, bytes: 20000, elapsed: 1000 });
  assert.equal(link.tier, TIER.GREEN);

  c.advance(9000);
  link.poll();
  link.poll();
  assert.equal(link.tier, TIER.RED);
  assert.equal(link.snapshot().stale, true);
});

test('loss rate is windowed, so old failures age out', () => {
  const link = new LinkQuality({ lossWindow: 5 });
  for (let i = 0; i < 5; i++) link.recordPing(false);
  assert.equal(link.lossRate, 1);
  for (let i = 0; i < 5; i++) link.recordPing(true, 30);
  assert.equal(link.lossRate, 0);
});

test('each tier maps to a distinct transmit profile', () => {
  assert.equal(profileFor(TIER.GREEN).mode, 'stream');
  assert.equal(profileFor(TIER.AMBER).mode, 'clip');
  assert.equal(profileFor(TIER.RED).mode, 'transcribe');
  assert.ok(profileFor(TIER.GREEN).audioBitsPerSecond > profileFor(TIER.AMBER).audioBitsPerSecond);
  assert.equal(profileFor(TIER.RED).audioBitsPerSecond, 0, 'RED must not try to send audio');
});

test('a fresh fast link is not held at RED for want of a throughput sample', () => {
  // Regression: grading an unmeasured link as zero goodput pinned it to RED,
  // which meant it never transmitted audio, which meant goodput stayed unknown.
  const link = new LinkQuality();
  for (let i = 0; i < 8; i++) link.recordPing(true, 25);
  assert.equal(link.goodput, null);
  assert.equal(link.tier, TIER.GREEN);
});

test('a measured slow link still overrides a good RTT', () => {
  // Latency and capacity are independent: a link can answer pings instantly and
  // still be unable to move audio, so the measurement has to win.
  const fast = new LinkQuality();
  feed(fast, { count: 8, rtt: 25, bytes: 3000, elapsed: 1000 });
  assert.equal(fast.tier, TIER.AMBER, '3 kB/s carries low-rate audio but not a stream');

  const crawling = new LinkQuality();
  feed(crawling, { count: 8, rtt: 25, bytes: 300, elapsed: 1000 });
  assert.equal(crawling.tier, TIER.RED, '300 B/s is below even the 10 kbps clip profile');
});
