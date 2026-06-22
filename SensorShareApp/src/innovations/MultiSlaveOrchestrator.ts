/**
 * Innovation #6 — Multi-Slave Orchestrator
 * Manages up to 8 simultaneous slave connections.
 * Provides round-robin querying, fan-out commands, and aggregate statistics.
 */

import { SensorBundle, SlaveRecord, CommandPayload, WSMessage } from '../types';

type SendFn = (targetUserId: string, type: string, payload: unknown) => void;

interface SlaveEntry {
  record: SlaveRecord;
  latestBundle: SensorBundle | null;
  history: SensorBundle[];
}

const MAX_SLAVES = 8;
const MAX_HISTORY = 100;

export class MultiSlaveOrchestrator {
  private slaves: Map<string, SlaveEntry> = new Map();
  private sendFn: SendFn;

  constructor(sendFn: SendFn) {
    this.sendFn = sendFn;
  }

  registerSlave(record: SlaveRecord): boolean {
    if (this.slaves.size >= MAX_SLAVES && !this.slaves.has(record.userId)) {
      console.warn('[Orchestrator] Max slaves reached');
      return false;
    }
    this.slaves.set(record.userId, { record, latestBundle: null, history: [] });
    return true;
  }

  unregisterSlave(userId: string): void {
    this.slaves.delete(userId);
  }

  updateBundle(bundle: SensorBundle): void {
    const entry = this.slaves.get(bundle.userId);
    if (!entry) return;
    entry.latestBundle = bundle;
    entry.history.push(bundle);
    if (entry.history.length > MAX_HISTORY) entry.history.shift();
    // Update latency
    if (bundle.latencyMs !== undefined) {
      entry.record.latencyMs = bundle.latencyMs;
    }
    if (bundle.healthScore !== undefined) {
      entry.record.healthScore = bundle.healthScore;
    }
    entry.record.lastSeen = Date.now();
  }

  // Broadcast a command to ALL connected slaves
  broadcastCommand(command: CommandPayload['command'], value?: unknown): void {
    for (const [userId] of this.slaves) {
      this.sendFn(userId, 'COMMAND', { command, value, targetUserId: userId } as CommandPayload);
    }
  }

  // Send command to specific slave
  sendCommand(userId: string, command: CommandPayload['command'], value?: unknown): void {
    this.sendFn(userId, 'COMMAND', { command, value, targetUserId: userId } as CommandPayload);
  }

  // Aggregate health score across all slaves
  get aggregateHealth(): number {
    const scores = [...this.slaves.values()]
      .map(e => e.record.healthScore)
      .filter(s => s > 0);
    if (scores.length === 0) return 0;
    return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
  }

  // Average latency across all slaves
  get averageLatency(): number {
    const lats = [...this.slaves.values()].map(e => e.record.latencyMs);
    if (lats.length === 0) return 0;
    return Math.round(lats.reduce((a, b) => a + b, 0) / lats.length);
  }

  get slaveCount(): number { return this.slaves.size; }

  getSlaveList(): SlaveRecord[] {
    return [...this.slaves.values()].map(e => e.record);
  }

  getLatestBundle(userId: string): SensorBundle | null {
    return this.slaves.get(userId)?.latestBundle ?? null;
  }

  getAllLatestBundles(): SensorBundle[] {
    return [...this.slaves.values()]
      .map(e => e.latestBundle)
      .filter((b): b is SensorBundle => b !== null);
  }
}
