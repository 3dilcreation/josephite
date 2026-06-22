# SensorShare — Use Cases & Applications

## What Is SensorShare?

SensorShare is a real-time mobile sensor bridge that streams **all device sensors**
(accelerometer, gyroscope, magnetometer, barometer, GPS, battery, network, ambient
light, pedometer) plus a **live camera feed** from a **Slave** phone to a **Master**
phone over WiFi, Bluetooth, or a cloud relay — identified by a simple User ID.

---

## 10 Built-in Innovations

| # | Innovation | What It Does |
|---|---|---|
| 1 | **AI Anomaly Detector** | Rolling z-score detection flags statistical outliers across all sensor streams in real time with severity levels (low/medium/high/critical). |
| 2 | **Gesture Command Interface** | Shake, tilt, or double-tap the Master phone to remotely control the Slave (flip camera, start/stop stream, change sampling rate). |
| 3 | **Environmental Health Scorer** | Composite 0–100 score combining motion stability, battery health, network quality, GPS accuracy, latency, gyro stability, and magnetic field clarity. |
| 4 | **Precision Time-Sync Protocol** | NTP-inspired clock synchronisation eliminates timestamp drift between devices for microsecond-accurate data correlation. |
| 5 | **Adaptive Compression Engine** | Dynamically adjusts camera JPEG quality (25–85%) and sensor sampling rate (50–1000ms) based on real-time bandwidth and latency measurements. |
| 6 | **Multi-Slave Orchestrator** | Connect up to 8 Slave devices simultaneously; fan-out commands, aggregate health scores, and switch active slave from a single Master dashboard. |
| 7 | **Sensor Replay Engine** | Record any live session and play it back frame-by-frame at variable speed (0.25×–4×) for detailed forensic analysis. |
| 8 | **AR Sensor Overlay** | Live SVG layer drawn on top of the camera feed showing accelerometer vector, battery, GPS, health score, latency, and a targeting crosshair. |
| 9 | **Voice Alert System** | Text-to-speech announcements for critical events: free-fall, low battery (<10%), speed > 120 km/h, network loss, extreme vibration — with cooldown timers. |
| 10 | **One-Click Data Exporter** | Export complete sensor sessions as CSV (spreadsheet-ready), JSON (developer), or TXT summary and share via native share sheet (email, AirDrop, Drive). |

---

## Real-World Use Cases

### 1. Personal Safety & Elder Care
**Scenario:** An elderly parent carries a Slave phone. A caregiver monitors from a Master.
- Free-fall detection triggers voice alert + camera livestream instantly.
- GPS shows exact location if the person wanders.
- Battery alerts prevent the phone going dead unnoticed.
- Data exported for health reports shared with doctors.

---

### 2. Sports & Athletic Performance
**Scenario:** An athlete wears a Slave phone; a coach watches on Master courtside.
- Accelerometer + gyroscope analyse running cadence, jump force, and rotational speed.
- HealthScorer grades overall device/session quality.
- Replay Engine lets the coach review a specific sprint frame by frame.
- Anomaly Detector flags sudden deceleration (potential injury risk).

---

### 3. Construction & Industrial Equipment Monitoring
**Scenario:** Slave phone strapped to machinery; engineer monitors remotely.
- Barometer detects pressure changes in sealed chambers.
- Vibration anomalies from accelerometer flag bearing failure before breakdown.
- Magnetometer detects metal stress or external EM fields.
- Voice alert fires when vibration z-score exceeds critical threshold.
- Multi-Slave lets one engineer monitor multiple machines at once.

---

### 4. Field Research & Environmental Science
**Scenario:** Researchers deploy multiple Slave phones across a terrain.
- GPS logs precise coordinates + altitude for geospatial mapping.
- Barometer tracks weather pressure changes over time.
- Sensor Replay Engine replays and compares data from different sites.
- CSV export feeds directly into GIS or analysis tools (QGIS, Python, R).
- Multi-Slave Orchestrator manages the entire sensor array from one device.

---

### 5. Remote Vehicle & Drone Telemetry
**Scenario:** Slave phone mounted in a vehicle or drone; pilot monitors on Master.
- GPS speed, heading, and altitude stream in real time.
- Gyroscope tracks roll/pitch/yaw.
- Adaptive Compression maintains camera feed quality over cellular.
- AR Overlay shows telemetry data on the camera view.
- Gesture Interface: tilt Master phone to send control commands to Slave.

---

### 6. Security & Surveillance
**Scenario:** Slave phone acts as a covert sensor node in a monitored area.
- Camera streams to Master via cloud relay (no local network needed).
- Motion detected via accelerometer changes (vibration, tampering).
- Anomaly Detector flags unusual magnetic fields (metal intrusion).
- AR Overlay on camera feed shows movement vectors and timestamps.
- Replay Engine for post-incident forensic review.

---

### 7. Child & Pet Monitoring
**Scenario:** Slave phone in a child's backpack or pet collar attachment.
- GPS tracks real-time location and speed.
- Accelerometer distinguishes running, sitting, and sleeping patterns.
- Free-fall detection alerts parent if device (and child) falls.
- Two-way: Master sends commands to toggle Slave camera (check-in view).
- Voice Alert announces if child's GPS leaves a predefined speed corridor.

---

### 8. Medical & Rehabilitation
**Scenario:** Patient holds Slave phone during physiotherapy exercises.
- Gyroscope tracks joint rotation range of motion.
- Accelerometer quantifies repetition count and force.
- Health Scorer gives clinician a summary quality index per session.
- Replay Engine lets the therapist review exact movement kinematics.
- Exported data integrates with EHR systems via JSON format.

---

### 9. Smart Home & IoT Bridging
**Scenario:** Old smartphones repurposed as smart home sensor nodes.
- Slave phones placed around the house send temperature-proxy (barometer), light levels, and motion data.
- Master aggregates data from 8 rooms via Multi-Slave Orchestrator.
- Anomaly Detector triggers alerts for unusual magnetic fields (appliance faults) or unexpected motion.
- Data Exporter feeds into home automation platforms (Home Assistant, IFTTT).

---

### 10. Live Event Production & Broadcasting
**Scenario:** Camera operators carry Slave phones as additional angles.
- Master director sees all camera feeds simultaneously (multi-slave).
- Gesture commands switch between camera angles without radio earpieces.
- AR Overlay shows production metadata on each feed.
- Time-Sync Protocol ensures all camera timestamps align for edit sync.
- Adaptive Compression maintains quality even on congested venue WiFi.

---

## Technical Summary

| Component | Technology |
|---|---|
| Mobile Framework | React Native + Expo |
| Navigation | Expo Router (file-based) |
| Sensors | expo-sensors, expo-location, expo-battery, expo-network |
| Camera | expo-camera (CameraView) |
| Transport (WiFi/Net) | WebSocket relay server (Node.js + ws) |
| Transport (Bluetooth) | react-native-ble-plx (GATT peripheral/central) |
| State Management | Zustand |
| AR Overlay | react-native-svg (SVG compositing on camera) |
| Voice Alerts | expo-speech + expo-haptics |
| Data Export | expo-file-system + expo-sharing |
| Time Sync | Custom NTP-style protocol |
| Anomaly Detection | Rolling z-score (no ML dependency) |
