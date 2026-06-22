/**
 * Innovation #4 — Precision Time-Sync Protocol
 * Implements a simplified NTP-style clock offset calculation.
 * Master sends T1, slave echoes T1+T2, master computes offset using T3.
 *
 *   offset = ((T2 - T1) + (T3 - T4)) / 2
 *   RTT    = (T4 - T1) - (T3 - T2)
 */

import { TimeSyncResult } from '../types';

export class TimeSyncProtocol {
  private pendingT1: Map<string, number> = new Map();
  private _offset = 0;
  private _rtt = 0;
  private samples: number[] = [];
  private readonly maxSamples = 8;

  /** Called by Master just before sending TIME_SYNC_REQUEST */
  createRequest(requestId: string): { requestId: string; t1: number } {
    const t1 = Date.now();
    this.pendingT1.set(requestId, t1);
    return { requestId, t1 };
  }

  /** Called by Slave when it receives TIME_SYNC_REQUEST — echoes T2 */
  createResponse(requestId: string, t1: number): { requestId: string; t1: number; t2: number } {
    return { requestId, t1, t2: Date.now() };
  }

  /** Called by Master when it receives TIME_SYNC_RESPONSE */
  processResponse(requestId: string, t1Echo: number, t2: number): TimeSyncResult | null {
    const t1 = this.pendingT1.get(requestId);
    if (!t1) return null;
    this.pendingT1.delete(requestId);

    const t3 = t2;  // slave's receive time (echo)
    const t4 = Date.now(); // master's receive time

    const offset = ((t2 - t1Echo) + (t3 - t4)) / 2;
    const rtt = (t4 - t1) - (t3 - t2);

    this.samples.push(offset);
    if (this.samples.length > this.maxSamples) this.samples.shift();

    // Median filter for stability
    const sorted = [...this.samples].sort((a, b) => a - b);
    this._offset = sorted[Math.floor(sorted.length / 2)];
    this._rtt = rtt;

    return { offset: this._offset, roundTripMs: rtt, serverTime: t2 };
  }

  /** Apply to any local timestamp to get clock-synced version */
  sync(localTimestamp: number): number {
    return localTimestamp + this._offset;
  }

  get offset(): number { return this._offset; }
  get rtt(): number { return this._rtt; }
}
