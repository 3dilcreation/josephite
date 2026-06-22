/**
 * Innovation #9 — Voice Alert System
 * Speaks critical notifications using expo-speech.
 * Debounces repeated alerts and respects severity levels.
 */

import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import { AnomalyEvent, SensorBundle } from '../types';

interface AlertRule {
  id: string;
  check: (bundle: SensorBundle) => string | null; // returns message or null
  cooldownMs: number;
}

const lastAlertTime: Record<string, number> = {};

const RULES: AlertRule[] = [
  {
    id: 'low_battery',
    cooldownMs: 60_000,
    check: (b) => {
      if (b.battery && b.battery.level < 0.10) {
        return `Warning: slave device battery is at ${Math.round(b.battery.level * 100)} percent.`;
      }
      return null;
    },
  },
  {
    id: 'free_fall',
    cooldownMs: 3_000,
    check: (b) => {
      if (!b.accelerometer) return null;
      const mag = Math.sqrt(b.accelerometer.x ** 2 + b.accelerometer.y ** 2 + b.accelerometer.z ** 2);
      if (mag < 1.0) return 'Alert: free fall detected on slave device!';
      return null;
    },
  },
  {
    id: 'high_speed',
    cooldownMs: 10_000,
    check: (b) => {
      const speed = b.location?.speed;
      if (speed !== null && speed !== undefined && speed > 33) { // > 120 km/h
        return `Alert: slave device speed exceeds ${Math.round(speed * 3.6)} kilometres per hour.`;
      }
      return null;
    },
  },
  {
    id: 'disconnection',
    cooldownMs: 5_000,
    check: (b) => {
      if (!b.network?.isConnected) return 'Warning: slave device has lost network connectivity.';
      return null;
    },
  },
  {
    id: 'strong_vibration',
    cooldownMs: 5_000,
    check: (b) => {
      if (!b.accelerometer) return null;
      const mag = Math.sqrt(b.accelerometer.x ** 2 + b.accelerometer.y ** 2 + b.accelerometer.z ** 2);
      if (mag > 25) return 'Alert: extreme motion detected on slave device.';
      return null;
    },
  },
];

export async function evaluateAndAlert(bundle: SensorBundle): Promise<void> {
  const now = Date.now();
  for (const rule of RULES) {
    const last = lastAlertTime[rule.id] ?? 0;
    if (now - last < rule.cooldownMs) continue;
    const message = rule.check(bundle);
    if (message) {
      lastAlertTime[rule.id] = now;
      await speak(message);
    }
  }
}

export async function alertAnomaly(event: AnomalyEvent): Promise<void> {
  if (event.severity !== 'critical' && event.severity !== 'high') return;
  const key = `anomaly_${event.sensor}_${event.field}`;
  const now = Date.now();
  if (now - (lastAlertTime[key] ?? 0) < 8_000) return;
  lastAlertTime[key] = now;
  const msg = `${event.severity} anomaly detected in ${event.sensor} ${event.field}. Value: ${event.value.toFixed(2)}.`;
  await speak(msg);
}

async function speak(message: string): Promise<void> {
  try {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    Speech.speak(message, { rate: 1.0, pitch: 1.0, language: 'en-US' });
  } catch (e) {
    console.warn('[VoiceAlert] Speech error', e);
  }
}

export function stopAllAlerts(): void {
  Speech.stop();
}
