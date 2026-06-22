export const RELAY_SERVER_URL = 'wss://sensorshare-relay.example.com'; // Replace with real server
export const LOCAL_WS_PORT = 8765;
export const HEARTBEAT_INTERVAL_MS = 3000;
export const SENSOR_INTERVAL_MS = 100;          // 10 Hz default
export const CAMERA_FRAME_INTERVAL_MS = 200;    // 5 FPS default
export const ANOMALY_WINDOW_SIZE = 50;          // rolling window for z-score
export const GESTURE_WINDOW_MS = 500;
export const HEALTH_SCORE_INTERVAL_MS = 2000;

export const COLORS = {
  bg: '#0A0E1A',
  surface: '#141829',
  border: '#1E2640',
  primary: '#4F8EF7',
  success: '#2ECC71',
  warning: '#F39C12',
  danger: '#E74C3C',
  text: '#E8EAF6',
  textMuted: '#8892B0',
  accent: '#7B2FBE',
  cyan: '#00D4FF',
  chartLine: '#4F8EF7',
  chartFill: 'rgba(79,142,247,0.15)',
};

export const SENSOR_LABELS: Record<string, string> = {
  accelerometer: 'Accelerometer',
  gyroscope: 'Gyroscope',
  magnetometer: 'Magnetometer',
  barometer: 'Barometer',
  location: 'GPS / Location',
  battery: 'Battery',
  network: 'Network',
  light: 'Ambient Light',
  pedometer: 'Pedometer',
};

export const SENSOR_UNITS: Record<string, string> = {
  accelerometer: 'm/s²',
  gyroscope: 'rad/s',
  magnetometer: 'μT',
  barometer: 'hPa',
  location: '°',
  battery: '%',
  network: '',
  light: 'lux',
  pedometer: 'steps',
};

export const INNOVATIONS = [
  {
    id: 1,
    name: 'AI Anomaly Detector',
    description: 'Real-time z-score based anomaly detection across all sensor streams with severity grading.',
    icon: '🧠',
  },
  {
    id: 2,
    name: 'Gesture Command Interface',
    description: 'Control the slave device remotely by shaking, tilting, or double-tapping the master phone.',
    icon: '👋',
  },
  {
    id: 3,
    name: 'Environmental Health Scorer',
    description: 'Composite 0–100 score combining motion stability, GPS accuracy, battery health, and network quality.',
    icon: '❤️',
  },
  {
    id: 4,
    name: 'Precision Time-Sync Protocol',
    description: 'NTP-inspired clock synchronisation between master and slave for microsecond-accurate timestamps.',
    icon: '⏱️',
  },
  {
    id: 5,
    name: 'Adaptive Compression Engine',
    description: 'Dynamically adjusts camera JPEG quality and sensor sampling rate based on available bandwidth.',
    icon: '📦',
  },
  {
    id: 6,
    name: 'Multi-Slave Orchestrator',
    description: 'Connect up to 8 slave devices simultaneously; switch between them or view split-screen data.',
    icon: '🌐',
  },
  {
    id: 7,
    name: 'Sensor Replay Engine',
    description: 'Record a full sensor session and replay it later frame-by-frame for analysis or training.',
    icon: '⏪',
  },
  {
    id: 8,
    name: 'AR Sensor Overlay',
    description: 'Overlay live sensor numbers and charts on the camera feed using Canvas compositing.',
    icon: '🔮',
  },
  {
    id: 9,
    name: 'Voice Alert System',
    description: 'Text-to-speech announcements when critical thresholds are breached (fall detection, low battery, etc.).',
    icon: '🔊',
  },
  {
    id: 10,
    name: 'One-Click Data Exporter',
    description: 'Export any session as CSV, JSON, or PDF report and share via email, cloud, or AirDrop.',
    icon: '📤',
  },
];
