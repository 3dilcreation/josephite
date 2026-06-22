/**
 * Innovation #5 — Adaptive Compression Engine
 * Dynamically adjusts camera JPEG quality and sensor emission rate
 * based on measured throughput and connection latency.
 */

import { SENSOR_INTERVAL_MS, CAMERA_FRAME_INTERVAL_MS } from '../constants';

interface BandwidthSample { bytes: number; ms: number; timestamp: number }

const SAMPLE_WINDOW_MS = 5000;
const samples: BandwidthSample[] = [];

export class AdaptiveCompressor {
  private currentJpegQuality = 0.5;
  private currentSensorIntervalMs = SENSOR_INTERVAL_MS;
  private currentCameraIntervalMs = CAMERA_FRAME_INTERVAL_MS;
  private latencyMs = 50;

  recordTransmission(bytes: number, durationMs: number): void {
    samples.push({ bytes, ms: durationMs, timestamp: Date.now() });
    // Prune old samples
    const cutoff = Date.now() - SAMPLE_WINDOW_MS;
    while (samples.length > 0 && samples[0].timestamp < cutoff) samples.shift();
    this.adapt();
  }

  updateLatency(latencyMs: number): void {
    this.latencyMs = latencyMs;
    this.adapt();
  }

  private estimateBandwidthKbps(): number {
    if (samples.length < 2) return 500; // assume 500 Kbps default
    const totalBytes = samples.reduce((s, x) => s + x.bytes, 0);
    const totalMs = samples.reduce((s, x) => s + x.ms, 0);
    if (totalMs === 0) return 500;
    return (totalBytes / totalMs) * 8; // kbps
  }

  private adapt(): void {
    const bw = this.estimateBandwidthKbps();
    const lat = this.latencyMs;

    // Tier-based adjustment
    if (bw > 2000 && lat < 80) {
      // Excellent — max quality
      this.currentJpegQuality = 0.85;
      this.currentSensorIntervalMs = 50;
      this.currentCameraIntervalMs = 100; // 10 FPS
    } else if (bw > 800 && lat < 200) {
      // Good
      this.currentJpegQuality = 0.65;
      this.currentSensorIntervalMs = 100;
      this.currentCameraIntervalMs = 200; // 5 FPS
    } else if (bw > 300 && lat < 400) {
      // Fair
      this.currentJpegQuality = 0.45;
      this.currentSensorIntervalMs = 200;
      this.currentCameraIntervalMs = 500; // 2 FPS
    } else {
      // Poor — minimum viable
      this.currentJpegQuality = 0.25;
      this.currentSensorIntervalMs = 500;
      this.currentCameraIntervalMs = 1000; // 1 FPS
    }
  }

  get jpegQuality(): number { return this.currentJpegQuality; }
  get sensorIntervalMs(): number { return this.currentSensorIntervalMs; }
  get cameraIntervalMs(): number { return this.currentCameraIntervalMs; }

  get compressionRatio(): number {
    // Ratio relative to max quality baseline
    return Math.round((1 - this.currentJpegQuality) * 100);
  }

  describe(): string {
    return `BW:${Math.round(this.estimateBandwidthKbps())}Kbps Lat:${this.latencyMs}ms Q:${Math.round(this.currentJpegQuality * 100)}%`;
  }
}
