/**
 * Innovation #3 — Environmental Health Scorer
 * Produces a composite 0–100 score from multiple sensor inputs.
 * Used both on the slave (self-score) and by the master for remote health.
 */

import { SensorBundle } from '../types';

interface ScoreComponent {
  name: string;
  weight: number;
  compute: (b: SensorBundle) => number; // returns 0–100
}

const components: ScoreComponent[] = [
  {
    name: 'Motion Stability',
    weight: 0.20,
    compute: (b) => {
      if (!b.accelerometer) return 50;
      const mag = Math.sqrt(b.accelerometer.x ** 2 + b.accelerometer.y ** 2 + b.accelerometer.z ** 2);
      // 9.8 = still, higher = shaking
      const deviation = Math.abs(mag - 9.8);
      return Math.max(0, 100 - deviation * 10);
    },
  },
  {
    name: 'Battery Health',
    weight: 0.20,
    compute: (b) => {
      if (!b.battery) return 50;
      let score = b.battery.level * 100;
      if (b.battery.lowPowerMode) score *= 0.8;
      if (b.battery.state === 'charging') score = Math.min(100, score + 5);
      return score;
    },
  },
  {
    name: 'Network Quality',
    weight: 0.20,
    compute: (b) => {
      if (!b.network) return 50;
      if (!b.network.isConnected) return 0;
      if (b.network.type === 'wifi') return 90;
      if (b.network.type === 'cellular') return 70;
      return 40;
    },
  },
  {
    name: 'Location Accuracy',
    weight: 0.15,
    compute: (b) => {
      if (!b.location) return 50;
      const acc = b.location.accuracy;
      if (acc <= 5) return 100;
      if (acc <= 15) return 85;
      if (acc <= 50) return 60;
      if (acc <= 100) return 40;
      return 20;
    },
  },
  {
    name: 'Gyro Stability',
    weight: 0.15,
    compute: (b) => {
      if (!b.gyroscope) return 50;
      const mag = Math.sqrt(b.gyroscope.x ** 2 + b.gyroscope.y ** 2 + b.gyroscope.z ** 2);
      return Math.max(0, 100 - mag * 20);
    },
  },
  {
    name: 'Magnetic Field Clarity',
    weight: 0.10,
    compute: (b) => {
      if (!b.magnetometer) return 50;
      const mag = Math.sqrt(b.magnetometer.x ** 2 + b.magnetometer.y ** 2 + b.magnetometer.z ** 2);
      // Earth's field is typically 25–65 μT
      if (mag >= 25 && mag <= 65) return 100;
      if (mag < 25) return Math.max(0, (mag / 25) * 80);
      return Math.max(0, 100 - ((mag - 65) / 65) * 100);
    },
  },
  {
    name: 'Latency',
    weight: 0.10,
    compute: (b) => {
      const lat = b.latencyMs ?? 0;
      if (lat === 0) return 80;
      if (lat < 50) return 100;
      if (lat < 150) return 80;
      if (lat < 300) return 60;
      if (lat < 500) return 40;
      return 10;
    },
  },
];

export function computeHealthScore(bundle: SensorBundle): number {
  const totalWeight = components.reduce((s, c) => s + c.weight, 0);
  const weightedSum = components.reduce((s, c) => s + c.weight * c.compute(bundle), 0);
  return Math.round(weightedSum / totalWeight);
}

export function getScoreColor(score: number): string {
  if (score >= 80) return '#2ECC71';
  if (score >= 60) return '#F39C12';
  if (score >= 40) return '#E67E22';
  return '#E74C3C';
}

export function getScoreLabel(score: number): string {
  if (score >= 80) return 'Excellent';
  if (score >= 60) return 'Good';
  if (score >= 40) return 'Fair';
  return 'Poor';
}

export function getComponentBreakdown(bundle: SensorBundle): { name: string; score: number; weight: number }[] {
  return components.map(c => ({
    name: c.name,
    score: Math.round(c.compute(bundle)),
    weight: c.weight,
  }));
}
