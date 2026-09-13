// Per-peer link grading.
//
// This is the piece the rest of the app hangs off: it watches round-trip time,
// loss and measured goodput, and reports one of three tiers. The transmit path
// reads the tier to decide whether a held PTT button turns into streamed Opus,
// a slow buffered clip, or a locally transcribed sentence.
//
// Deliberately hysteretic. A tier that flaps between GREEN and RED mid-sentence
// is worse than one that is slightly stale, because each switch costs a codec
// restart at both ends.

export const TIER = { GREEN: 'green', AMBER: 'amber', RED: 'red' };

export const DEFAULTS = {
  // A tier's thresholds are the *floor* it needs to hold.
  green: { maxRtt: 180, maxLoss: 0.03, minGoodput: 6000 },   // bytes/sec
  amber: { maxRtt: 700, maxLoss: 0.20, minGoodput: 1200 },
  // Consecutive samples required before the reported tier actually moves.
  // Downgrades are quicker than upgrades: being pessimistic costs quality,
  // being optimistic costs a dropped transmission.
  downgradeSamples: 2,
  upgradeSamples: 4,
  rttAlpha: 0.3,        // EWMA weight for new RTT samples
  goodputAlpha: 0.3,
  lossWindow: 20,       // pings retained for the loss ratio
  sampleTimeoutMs: 8000, // no fresh sample for this long => assume RED
};

const RANK = { [TIER.GREEN]: 2, [TIER.AMBER]: 1, [TIER.RED]: 0 };

export class LinkQuality {
  constructor(options = {}) {
    this.opts = { ...DEFAULTS, ...options };
    this.opts.green = { ...DEFAULTS.green, ...(options.green || {}) };
    this.opts.amber = { ...DEFAULTS.amber, ...(options.amber || {}) };
    this.now = options.now || (() => Date.now());

    this.rtt = null;
    this.goodput = null;
    this.pings = [];        // booleans, newest last: true = answered
    this.tier = TIER.AMBER; // start cautious; a fresh link has no evidence yet
    this.candidate = TIER.AMBER;
    this.candidateCount = 0;
    this.lastSampleAt = this.now();
  }

  recordPing(answered, rttMs) {
    this.pings.push(Boolean(answered));
    if (this.pings.length > this.opts.lossWindow) this.pings.shift();
    if (answered && Number.isFinite(rttMs)) {
      this.rtt = ewma(this.rtt, rttMs, this.opts.rttAlpha);
    }
    this.lastSampleAt = this.now();
    return this.#settle();
  }

  // Called after a chunk is confirmed delivered, so the number reflects what
  // the link actually carried rather than what we optimistically queued.
  recordThroughput(bytes, elapsedMs) {
    if (elapsedMs > 0 && bytes > 0) {
      this.goodput = ewma(this.goodput, (bytes * 1000) / elapsedMs, this.opts.goodputAlpha);
      this.lastSampleAt = this.now();
    }
    return this.#settle();
  }

  get lossRate() {
    if (!this.pings.length) return 0;
    const lost = this.pings.reduce((n, ok) => n + (ok ? 0 : 1), 0);
    return lost / this.pings.length;
  }

  // The tier the raw numbers justify right now, before hysteresis.
  rawTier() {
    if (this.now() - this.lastSampleAt > this.opts.sampleTimeoutMs) return TIER.RED;
    const rtt = this.rtt ?? Infinity;
    const loss = this.lossRate;

    // An unmeasured link is not a slow one. Grading a fresh peer on a goodput of
    // zero would pin it to RED, which would stop it ever sending the audio that
    // produces a goodput sample in the first place -- so an absent measurement
    // simply does not participate until one exists.
    const goodput = this.goodput;
    const meets = (t) =>
      rtt <= t.maxRtt && loss <= t.maxLoss && (goodput === null || goodput >= t.minGoodput);
    if (meets(this.opts.green)) return TIER.GREEN;
    if (meets(this.opts.amber)) return TIER.AMBER;
    return TIER.RED;
  }

  // Re-evaluates without a new sample, so a link that has gone silent still
  // decays to RED instead of sitting on a stale GREEN.
  poll() {
    return this.#settle();
  }

  snapshot() {
    return {
      tier: this.tier,
      rtt: this.rtt,
      goodput: this.goodput,
      lossRate: this.lossRate,
      stale: this.now() - this.lastSampleAt > this.opts.sampleTimeoutMs,
    };
  }

  #settle() {
    const raw = this.rawTier();
    if (raw === this.tier) {
      this.candidate = raw;
      this.candidateCount = 0;
      return this.tier;
    }
    if (raw !== this.candidate) {
      this.candidate = raw;
      this.candidateCount = 0;
    }
    this.candidateCount += 1;

    const needed = RANK[raw] < RANK[this.tier]
      ? this.opts.downgradeSamples
      : this.opts.upgradeSamples;

    if (this.candidateCount >= needed) {
      this.tier = raw;
      this.candidateCount = 0;
    }
    return this.tier;
  }
}

function ewma(prev, next, alpha) {
  return prev === null ? next : prev * (1 - alpha) + next * alpha;
}

// Transmit settings each tier implies. AMBER keeps audio but trades latency for
// resilience; RED gives up on carrying audio at all.
export function profileFor(tier) {
  switch (tier) {
    case TIER.GREEN:
      return { mode: 'stream', audioBitsPerSecond: 24000, timesliceMs: 220, live: true };
    case TIER.AMBER:
      return { mode: 'clip', audioBitsPerSecond: 10000, timesliceMs: 1000, live: false };
    default:
      return { mode: 'transcribe', audioBitsPerSecond: 0, timesliceMs: 0, live: false };
  }
}
