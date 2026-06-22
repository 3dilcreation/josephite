/**
 * Innovation #2 — Gesture Command Interface
 * Recognises shake, tilt, double-tap, and free-fall from the master phone's
 * accelerometer and dispatches remote commands to the active slave.
 */

import { AccelerometerData, GestureEvent } from '../types';
import { GESTURE_WINDOW_MS } from '../constants';

const SHAKE_THRESHOLD = 2.5;       // m/s² delta magnitude
const TILT_ANGLE_DEG = 40;
const FREE_FALL_THRESHOLD = 0.3;   // near-zero g vector
const TAP_THRESHOLD = 1.8;

interface Sample { x: number; y: number; z: number; t: number }

const window: Sample[] = [];
let lastTapTime = 0;
let doubleTapCallback: (() => void) | null = null;

export type GestureCallback = (event: GestureEvent) => void;

let _cb: GestureCallback | null = null;

export function setGestureCallback(cb: GestureCallback): void {
  _cb = cb;
}

function magnitude(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

function angleDeg(a: number, total: number): number {
  return Math.asin(Math.min(1, Math.abs(a) / Math.max(total, 0.01))) * (180 / Math.PI);
}

export function feedAccelerometer(data: AccelerometerData): void {
  const now = Date.now();
  const sample: Sample = { ...data, t: now };
  window.push(sample);
  // Keep only last GESTURE_WINDOW_MS of samples
  while (window.length > 0 && now - window[0].t > GESTURE_WINDOW_MS) window.shift();

  const mag = magnitude(data.x, data.y, data.z);

  // ── Free-fall ─────────────────────────────────
  if (mag < FREE_FALL_THRESHOLD) {
    emit({ type: 'free_fall', confidence: 1 - mag / FREE_FALL_THRESHOLD, timestamp: now });
    return;
  }

  // ── Double tap ────────────────────────────────
  if (mag > TAP_THRESHOLD && Math.abs(data.z) > TAP_THRESHOLD * 0.6) {
    const delta = now - lastTapTime;
    if (delta < 400 && delta > 80) {
      emit({ type: 'double_tap', confidence: 0.9, timestamp: now });
      lastTapTime = 0;
    } else {
      lastTapTime = now;
    }
  }

  // ── Shake ─────────────────────────────────────
  if (window.length >= 3) {
    const prev = window[window.length - 3];
    const dx = Math.abs(data.x - prev.x);
    const dy = Math.abs(data.y - prev.y);
    const dz = Math.abs(data.z - prev.z);
    if (dx + dy + dz > SHAKE_THRESHOLD * 3) {
      emit({ type: 'shake', confidence: Math.min(1, (dx + dy + dz) / (SHAKE_THRESHOLD * 6)), timestamp: now });
    }
  }

  // ── Tilt ──────────────────────────────────────
  const pitch = angleDeg(data.y, mag);
  const roll  = angleDeg(data.x, mag);

  if (pitch > TILT_ANGLE_DEG) {
    emit({ type: data.y > 0 ? 'tilt_forward' : 'tilt_back', confidence: pitch / 90, timestamp: now });
  } else if (roll > TILT_ANGLE_DEG) {
    emit({ type: data.x > 0 ? 'tilt_right' : 'tilt_left', confidence: roll / 90, timestamp: now });
  }
}

let _lastEmitType: string = '';
let _lastEmitTime = 0;

function emit(event: GestureEvent): void {
  const now = Date.now();
  // Debounce same gesture type for 300 ms
  if (event.type === _lastEmitType && now - _lastEmitTime < 300) return;
  _lastEmitType = event.type;
  _lastEmitTime = now;
  _cb?.(event);
}
