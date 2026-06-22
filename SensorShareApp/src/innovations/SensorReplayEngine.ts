/**
 * Innovation #7 — Sensor Replay Engine
 * Records a live sensor session and plays it back at original speed or
 * at adjustable multipliers, frame by frame.
 */

import { SensorBundle, ReplaySession } from '../types';
import { v4 as uuidv4 } from 'uuid';

type FrameCallback = (bundle: SensorBundle, index: number, total: number) => void;
type DoneCallback = () => void;

export class SensorReplayEngine {
  private recording: SensorBundle[] = [];
  private isRecording = false;
  private playbackTimer: ReturnType<typeof setTimeout> | null = null;
  private sessions: ReplaySession[] = [];
  private recordStartTime = 0;

  // ── Recording ──────────────────────────────────
  startRecording(): void {
    this.recording = [];
    this.isRecording = true;
    this.recordStartTime = Date.now();
  }

  record(bundle: SensorBundle): void {
    if (!this.isRecording) return;
    this.recording.push({ ...bundle });
  }

  stopRecording(): ReplaySession {
    this.isRecording = false;
    const session: ReplaySession = {
      id: uuidv4(),
      userId: this.recording[0]?.userId ?? 'unknown',
      startTime: this.recordStartTime,
      endTime: Date.now(),
      frames: this.recording,
      cameraFrameCount: 0,
    };
    this.sessions.push(session);
    return session;
  }

  // ── Playback ───────────────────────────────────
  play(
    session: ReplaySession,
    onFrame: FrameCallback,
    onDone: DoneCallback,
    speedMultiplier = 1.0,
  ): void {
    this.stop();
    const frames = session.frames;
    if (frames.length === 0) { onDone(); return; }

    let index = 0;

    const step = () => {
      if (index >= frames.length) { onDone(); return; }
      const current = frames[index];
      const next = frames[index + 1];
      onFrame(current, index, frames.length);
      index++;

      if (next) {
        const delay = Math.max(16, (next.timestamp - current.timestamp) / speedMultiplier);
        this.playbackTimer = setTimeout(step, delay);
      } else {
        onDone();
      }
    };

    step();
  }

  stop(): void {
    if (this.playbackTimer) {
      clearTimeout(this.playbackTimer);
      this.playbackTimer = null;
    }
  }

  get sessions_(): ReplaySession[] { return this.sessions; }
  get frameCount(): number { return this.recording.length; }
  get durationMs(): number {
    if (this.recording.length < 2) return 0;
    return this.recording[this.recording.length - 1].timestamp - this.recording[0].timestamp;
  }
}
