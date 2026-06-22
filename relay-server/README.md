# SensorShare Relay Server

Node.js WebSocket relay server that routes sensor and camera data between Slave and Master devices.

## Quick Start

```bash
npm install
npm start          # production
npm run dev        # development with auto-reload (requires nodemon)
```

Server listens on `ws://0.0.0.0:8080` by default.
Set `PORT` environment variable to change it.

## Deployment Options

### Railway / Render / Fly.io
```bash
# Set env var PORT (auto-assigned) and deploy.
# Update RELAY_SERVER_URL in SensorShareApp/src/constants/index.ts to wss://<your-domain>
```

### Docker
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json .
RUN npm ci --production
COPY server.js .
EXPOSE 8080
CMD ["node", "server.js"]
```

## Protocol

All messages are JSON: `{ type, payload, timestamp, senderId }`.

| Message | Direction | Description |
|---|---|---|
| `HANDSHAKE` | Client → Relay | Register as slave or master |
| `HANDSHAKE_ACK` | Relay → Client | Confirm registration |
| `SUBSCRIBE` | Master → Relay | Subscribe to a slave's data by userId |
| `SENSOR_DATA` | Slave → Masters | Full sensor bundle |
| `CAMERA_FRAME` | Slave → Masters | Base64 JPEG camera frame |
| `COMMAND` | Master → Slave | Remote control command |
| `HEARTBEAT` | Client → Relay | Keep-alive ping |
| `HEARTBEAT_ACK` | Relay → Client | Pong response |
| `TIME_SYNC_REQUEST` | Master → Relay | NTP-style sync request |
| `TIME_SYNC_RESPONSE` | Relay → Master | Sync response with server timestamp |
| `ALERT` | Slave → Masters | Critical event notification |
