/**
 * Innovation #8 — AR Sensor Overlay
 * Generates an SVG/Canvas overlay spec that the CameraFeed component
 * renders on top of the live camera view.
 * Returns a list of overlay elements (text, bars, circles) with positions.
 */

import { SensorBundle } from '../types';
import { COLORS } from '../constants';

export interface OverlayElement {
  type: 'text' | 'bar' | 'circle' | 'crosshair';
  x: number; // 0-1 relative
  y: number; // 0-1 relative
  value?: string;
  color?: string;
  width?: number;  // 0-1 for bar fill
  radius?: number; // 0-1 for circle
  fontSize?: number;
  label?: string;
}

export function buildAROverlay(bundle: SensorBundle, frameW: number, frameH: number): OverlayElement[] {
  const elements: OverlayElement[] = [];

  // ── Top-left: Accelerometer vector ────────────
  if (bundle.accelerometer) {
    const a = bundle.accelerometer;
    const mag = Math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2);
    elements.push({
      type: 'text', x: 0.02, y: 0.04,
      value: `⚡ ${mag.toFixed(2)} m/s²`,
      color: COLORS.cyan, fontSize: 14,
    });
    elements.push({
      type: 'bar', x: 0.02, y: 0.09, width: Math.min(1, mag / 20),
      color: COLORS.cyan, label: 'ACC',
    });
  }

  // ── Top-right: Battery ────────────────────────
  if (bundle.battery) {
    const pct = Math.round(bundle.battery.level * 100);
    const color = pct > 40 ? COLORS.success : pct > 20 ? COLORS.warning : COLORS.danger;
    elements.push({
      type: 'text', x: 0.75, y: 0.04,
      value: `🔋 ${pct}%`,
      color, fontSize: 14,
    });
  }

  // ── Bottom-left: GPS ──────────────────────────
  if (bundle.location) {
    const { latitude, longitude, speed } = bundle.location;
    elements.push({
      type: 'text', x: 0.02, y: 0.88,
      value: `📍 ${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
      color: COLORS.text, fontSize: 12,
    });
    if (speed !== null && speed !== undefined) {
      const kmh = (speed * 3.6).toFixed(1);
      elements.push({
        type: 'text', x: 0.02, y: 0.93,
        value: `🚀 ${kmh} km/h`,
        color: COLORS.primary, fontSize: 12,
      });
    }
  }

  // ── Center: Health score circle ───────────────
  if (bundle.healthScore !== undefined) {
    const color = bundle.healthScore >= 70
      ? COLORS.success : bundle.healthScore >= 40
        ? COLORS.warning : COLORS.danger;
    elements.push({
      type: 'circle', x: 0.87, y: 0.20,
      radius: 0.06, color,
      value: `${bundle.healthScore}`,
      label: 'Health',
    });
  }

  // ── Bottom-right: Network + latency ───────────
  if (bundle.network || bundle.latencyMs) {
    elements.push({
      type: 'text', x: 0.68, y: 0.93,
      value: `📶 ${bundle.network?.type ?? '-'} ${bundle.latencyMs ?? 0}ms`,
      color: COLORS.textMuted, fontSize: 11,
    });
  }

  // ── Crosshair overlay (always) ────────────────
  elements.push({ type: 'crosshair', x: 0.5, y: 0.5, color: 'rgba(255,255,255,0.4)' });

  return elements;
}
