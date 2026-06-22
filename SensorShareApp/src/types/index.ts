// ─────────────────────────────────────────────
//  Core data types for SensorShare
// ─────────────────────────────────────────────

export type ConnectionMode = 'wifi' | 'bluetooth' | 'relay';
export type DeviceRole = 'master' | 'slave';
export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error' | 'disconnected';

// ── Sensor Payloads ──────────────────────────

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface AccelerometerData extends Vector3 {}
export interface GyroscopeData extends Vector3 {}
export interface MagnetometerData extends Vector3 {}

export interface LocationData {
  latitude: number;
  longitude: number;
  altitude: number | null;
  speed: number | null;
  heading: number | null;
  accuracy: number;
  timestamp: number;
}

export interface BarometerData {
  pressure: number; // hPa
  relativeAltitude?: number; // metres
}

export interface BatteryData {
  level: number;       // 0-1
  state: string;       // 'charging' | 'full' | 'discharging' | 'unknown'
  lowPowerMode: boolean;
}

export interface NetworkData {
  type: string;       // 'wifi' | 'cellular' | 'none'
  isConnected: boolean;
  ip?: string;
  strength?: number;  // 0-100 estimated
}

export interface LightData {
  illuminance: number; // lux
}

export interface PedometerData {
  steps: number;
  distance?: number; // metres
}

export interface DeviceInfo {
  brand: string;
  modelName: string;
  osName: string;
  osVersion: string;
  deviceType: string;
}

// ── Full Sensor Bundle sent from Slave → Master ──

export interface SensorBundle {
  userId: string;
  deviceInfo: DeviceInfo;
  timestamp: number;
  serverTimestamp?: number;   // set by relay for time-sync
  latencyMs?: number;         // round-trip latency (TimeSyncProtocol)
  accelerometer?: AccelerometerData;
  gyroscope?: GyroscopeData;
  magnetometer?: MagnetometerData;
  barometer?: BarometerData;
  location?: LocationData;
  battery?: BatteryData;
  network?: NetworkData;
  light?: LightData;
  pedometer?: PedometerData;
  healthScore?: number;       // HealthScorer output 0-100
  anomalies?: AnomalyEvent[];
  gesture?: GestureEvent | null;
  compressionRatio?: number;  // AdaptiveCompressor
}

// ── Camera ───────────────────────────────────

export interface CameraFrame {
  userId: string;
  timestamp: number;
  frame: string;         // base64 JPEG
  width: number;
  height: number;
  quality: number;       // 0-1 JPEG quality used
  frameIndex: number;
}

// ── WebSocket Message Protocol ───────────────

export type WSMessageType =
  | 'HANDSHAKE'
  | 'HANDSHAKE_ACK'
  | 'SENSOR_DATA'
  | 'CAMERA_FRAME'
  | 'HEARTBEAT'
  | 'HEARTBEAT_ACK'
  | 'TIME_SYNC_REQUEST'
  | 'TIME_SYNC_RESPONSE'
  | 'COMMAND'
  | 'ALERT'
  | 'SLAVE_LIST'
  | 'REPLAY_CHUNK'
  | 'SUBSCRIBE'
  | 'UNSUBSCRIBE'
  | 'ERROR';

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  payload: T;
  timestamp: number;
  senderId: string;
}

export interface HandshakePayload {
  userId: string;
  role: DeviceRole;
  deviceInfo: DeviceInfo;
  capabilities: string[];
}

export interface CommandPayload {
  command: 'SWITCH_CAMERA' | 'START_STREAM' | 'STOP_STREAM' | 'SET_INTERVAL' | 'FLASH_ON' | 'FLASH_OFF' | 'START_REPLAY' | 'STOP_REPLAY';
  value?: unknown;
  targetUserId?: string;
}

// ── Innovations ───────────────────────────────

export interface AnomalyEvent {
  sensor: string;
  field: string;
  value: number;
  zScore: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: number;
  description: string;
}

export interface GestureEvent {
  type: 'shake' | 'tilt_left' | 'tilt_right' | 'tilt_forward' | 'tilt_back' | 'double_tap' | 'free_fall';
  confidence: number; // 0-1
  timestamp: number;
}

export interface TimeSyncResult {
  offset: number;     // ms offset to apply
  roundTripMs: number;
  serverTime: number;
}

export interface SlaveRecord {
  userId: string;
  deviceInfo: DeviceInfo;
  connectedAt: number;
  lastSeen: number;
  latencyMs: number;
  healthScore: number;
  isStreaming: boolean;
}

export interface ReplaySession {
  id: string;
  userId: string;
  startTime: number;
  endTime: number;
  frames: SensorBundle[];
  cameraFrameCount: number;
}
