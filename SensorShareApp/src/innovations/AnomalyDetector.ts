/**
 * Innovation #1 — AI Anomaly Detector
 * Uses rolling z-score to flag statistical outliers across all sensor fields.
 * Severity bands: |z| > 2 = low, > 3 = medium, > 4 = high, > 6 = critical.
 */

import { SensorBundle, AnomalyEvent } from '../types';
import { ANOMALY_WINDOW_SIZE } from '../constants';

interface RollingStats {
  values: number[];
  sum: number;
  sumSq: number;
}

const stats: Record<string, RollingStats> = {};

function push(key: string, value: number): void {
  if (!stats[key]) stats[key] = { values: [], sum: 0, sumSq: 0 };
  const s = stats[key];
  s.values.push(value);
  s.sum += value;
  s.sumSq += value * value;
  if (s.values.length > ANOMALY_WINDOW_SIZE) {
    const old = s.values.shift()!;
    s.sum -= old;
    s.sumSq -= old * old;
  }
}

function zScore(key: string, value: number): number {
  const s = stats[key];
  if (!s || s.values.length < 10) return 0;
  const n = s.values.length;
  const mean = s.sum / n;
  const variance = s.sumSq / n - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));
  if (std < 1e-9) return 0;
  return Math.abs((value - mean) / std);
}

function severity(z: number): AnomalyEvent['severity'] {
  if (z >= 6) return 'critical';
  if (z >= 4) return 'high';
  if (z >= 3) return 'medium';
  return 'low';
}

const WATCHED: { sensor: keyof SensorBundle; field: string; extract: (b: SensorBundle) => number | undefined }[] = [
  { sensor: 'accelerometer', field: 'x', extract: b => b.accelerometer?.x },
  { sensor: 'accelerometer', field: 'y', extract: b => b.accelerometer?.y },
  { sensor: 'accelerometer', field: 'z', extract: b => b.accelerometer?.z },
  { sensor: 'gyroscope', field: 'x', extract: b => b.gyroscope?.x },
  { sensor: 'gyroscope', field: 'y', extract: b => b.gyroscope?.y },
  { sensor: 'gyroscope', field: 'z', extract: b => b.gyroscope?.z },
  { sensor: 'magnetometer', field: 'x', extract: b => b.magnetometer?.x },
  { sensor: 'barometer', field: 'pressure', extract: b => b.barometer?.pressure },
  { sensor: 'location', field: 'speed', extract: b => b.location?.speed ?? 0 },
  { sensor: 'battery', field: 'level', extract: b => b.battery?.level },
];

export function detectAnomalies(bundle: SensorBundle): AnomalyEvent[] {
  const events: AnomalyEvent[] = [];

  for (const w of WATCHED) {
    const value = w.extract(bundle);
    if (value === undefined || value === null) continue;
    const key = `${w.sensor}.${w.field}`;
    push(key, value);
    const z = zScore(key, value);
    if (z >= 2) {
      events.push({
        sensor: w.sensor as string,
        field: w.field,
        value,
        zScore: Math.round(z * 100) / 100,
        severity: severity(z),
        timestamp: bundle.timestamp,
        description: `${w.sensor}.${w.field} = ${value.toFixed(3)} (z=${z.toFixed(2)})`,
      });
    }
  }

  return events;
}

export function resetStats(): void {
  for (const key of Object.keys(stats)) delete stats[key];
}
