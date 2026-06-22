import { create } from 'zustand';
import {
  SensorBundle, CameraFrame, SlaveRecord, ConnectionState,
  ConnectionMode, AnomalyEvent, GestureEvent, ReplaySession,
} from '../types';

// ── Slave Store ───────────────────────────────

interface SlaveState {
  userId: string;
  connectionState: ConnectionState;
  connectionMode: ConnectionMode;
  isCameraActive: boolean;
  isTransmitting: boolean;
  currentBundle: SensorBundle | null;
  sentFrameCount: number;
  latencyMs: number;
  healthScore: number;
  gesture: GestureEvent | null;
  setUserId: (id: string) => void;
  setConnectionState: (s: ConnectionState) => void;
  setConnectionMode: (m: ConnectionMode) => void;
  setCameraActive: (v: boolean) => void;
  setTransmitting: (v: boolean) => void;
  setCurrentBundle: (b: SensorBundle) => void;
  incrementSentFrames: () => void;
  setLatency: (ms: number) => void;
  setHealthScore: (s: number) => void;
  setGesture: (g: GestureEvent | null) => void;
  reset: () => void;
}

export const useSlaveStore = create<SlaveState>((set) => ({
  userId: '',
  connectionState: 'idle',
  connectionMode: 'relay',
  isCameraActive: false,
  isTransmitting: false,
  currentBundle: null,
  sentFrameCount: 0,
  latencyMs: 0,
  healthScore: 100,
  gesture: null,
  setUserId: (id) => set({ userId: id }),
  setConnectionState: (s) => set({ connectionState: s }),
  setConnectionMode: (m) => set({ connectionMode: m }),
  setCameraActive: (v) => set({ isCameraActive: v }),
  setTransmitting: (v) => set({ isTransmitting: v }),
  setCurrentBundle: (b) => set({ currentBundle: b }),
  incrementSentFrames: () => set((s) => ({ sentFrameCount: s.sentFrameCount + 1 })),
  setLatency: (ms) => set({ latencyMs: ms }),
  setHealthScore: (s) => set({ healthScore: s }),
  setGesture: (g) => set({ gesture: g }),
  reset: () => set({
    connectionState: 'idle', isTransmitting: false, isCameraActive: false,
    sentFrameCount: 0, currentBundle: null, gesture: null,
  }),
}));

// ── Master Store ──────────────────────────────

interface MasterState {
  userId: string;
  targetSlaveId: string;
  connectionState: ConnectionState;
  connectionMode: ConnectionMode;
  connectedSlaves: SlaveRecord[];
  activeSlaveId: string | null;
  latestBundle: SensorBundle | null;
  latestFrame: CameraFrame | null;
  sensorHistory: SensorBundle[];
  anomalies: AnomalyEvent[];
  replaySessions: ReplaySession[];
  isRecording: boolean;
  isReplaying: boolean;
  replayIndex: number;
  showAROverlay: boolean;
  setUserId: (id: string) => void;
  setTargetSlaveId: (id: string) => void;
  setConnectionState: (s: ConnectionState) => void;
  setConnectionMode: (m: ConnectionMode) => void;
  addSlave: (slave: SlaveRecord) => void;
  removeSlave: (userId: string) => void;
  updateSlave: (userId: string, patch: Partial<SlaveRecord>) => void;
  setActiveSlave: (id: string | null) => void;
  setLatestBundle: (b: SensorBundle) => void;
  setLatestFrame: (f: CameraFrame) => void;
  pushHistory: (b: SensorBundle) => void;
  addAnomaly: (a: AnomalyEvent) => void;
  clearAnomalies: () => void;
  addReplaySession: (r: ReplaySession) => void;
  setRecording: (v: boolean) => void;
  setReplaying: (v: boolean) => void;
  setReplayIndex: (i: number) => void;
  toggleAROverlay: () => void;
  reset: () => void;
}

const MAX_HISTORY = 300;

export const useMasterStore = create<MasterState>((set) => ({
  userId: '',
  targetSlaveId: '',
  connectionState: 'idle',
  connectionMode: 'relay',
  connectedSlaves: [],
  activeSlaveId: null,
  latestBundle: null,
  latestFrame: null,
  sensorHistory: [],
  anomalies: [],
  replaySessions: [],
  isRecording: false,
  isReplaying: false,
  replayIndex: 0,
  showAROverlay: false,
  setUserId: (id) => set({ userId: id }),
  setTargetSlaveId: (id) => set({ targetSlaveId: id }),
  setConnectionState: (s) => set({ connectionState: s }),
  setConnectionMode: (m) => set({ connectionMode: m }),
  addSlave: (slave) => set((st) => ({
    connectedSlaves: [...st.connectedSlaves.filter(s => s.userId !== slave.userId), slave],
    activeSlaveId: st.activeSlaveId ?? slave.userId,
  })),
  removeSlave: (uid) => set((st) => ({
    connectedSlaves: st.connectedSlaves.filter(s => s.userId !== uid),
    activeSlaveId: st.activeSlaveId === uid ? null : st.activeSlaveId,
  })),
  updateSlave: (uid, patch) => set((st) => ({
    connectedSlaves: st.connectedSlaves.map(s => s.userId === uid ? { ...s, ...patch } : s),
  })),
  setActiveSlave: (id) => set({ activeSlaveId: id }),
  setLatestBundle: (b) => set({ latestBundle: b }),
  setLatestFrame: (f) => set({ latestFrame: f }),
  pushHistory: (b) => set((st) => {
    const next = [...st.sensorHistory, b];
    return { sensorHistory: next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next };
  }),
  addAnomaly: (a) => set((st) => ({ anomalies: [a, ...st.anomalies].slice(0, 50) })),
  clearAnomalies: () => set({ anomalies: [] }),
  addReplaySession: (r) => set((st) => ({ replaySessions: [r, ...st.replaySessions] })),
  setRecording: (v) => set({ isRecording: v }),
  setReplaying: (v) => set({ isReplaying: v }),
  setReplayIndex: (i) => set({ replayIndex: i }),
  toggleAROverlay: () => set((st) => ({ showAROverlay: !st.showAROverlay })),
  reset: () => set({
    connectionState: 'idle', connectedSlaves: [], activeSlaveId: null,
    latestBundle: null, latestFrame: null, sensorHistory: [], anomalies: [],
    isRecording: false, isReplaying: false, replayIndex: 0,
  }),
}));
