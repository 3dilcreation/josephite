/**
 * SensorShare Relay Server
 * ========================
 * WebSocket relay that connects slave devices to master devices.
 *
 * Protocol:
 *   Slave  → JOIN with role:'slave'   → room  `slave:<userId>`
 *   Master → JOIN with role:'master'  → subscribes to `slave:<targetUserId>`
 *
 * All subsequent messages from slave are forwarded to all subscribed masters.
 * Commands from master are forwarded to the target slave.
 */

const { WebSocketServer, WebSocket } = require('ws');
const { v4: uuidv4 } = require('uuid');

const PORT = process.env.PORT || 8080;
const wss = new WebSocketServer({ port: PORT });

// userId → { ws, role, roomId, subscriptions: Set<userId> }
const clients = new Map();
// slaveUserId → Set of master ws clients
const slaveSubscribers = new Map();
// slaveUserId → slave ws client
const slaves = new Map();

console.log(`[Relay] SensorShare relay server started on ws://0.0.0.0:${PORT}`);

wss.on('connection', (ws) => {
  const connectionId = uuidv4();
  let userId = null;
  let role = null;

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    // ── HANDSHAKE ─────────────────────────────────
    if (msg.type === 'HANDSHAKE') {
      userId = msg.payload.userId;
      role   = msg.payload.role;

      clients.set(connectionId, { ws, userId, role, connectionId });

      if (role === 'slave') {
        slaves.set(userId, ws);
        if (!slaveSubscribers.has(userId)) slaveSubscribers.set(userId, new Set());

        // Notify existing masters that this slave connected
        const subs = slaveSubscribers.get(userId);
        subs?.forEach((masterWs) => {
          safeSend(masterWs, {
            type: 'HANDSHAKE',
            payload: msg.payload,
            timestamp: Date.now(),
            senderId: userId,
          });
        });

        console.log(`[Relay] Slave connected: ${userId}`);
        safeSend(ws, { type: 'HANDSHAKE_ACK', payload: { userId, role }, timestamp: Date.now(), senderId: 'relay' });
      }

      if (role === 'master') {
        console.log(`[Relay] Master connected: ${userId}`);
        safeSend(ws, { type: 'HANDSHAKE_ACK', payload: { userId, role }, timestamp: Date.now(), senderId: 'relay' });
      }
      return;
    }

    // ── SUBSCRIBE (master subscribes to a slave) ──
    if (msg.type === 'SUBSCRIBE') {
      const targetId = msg.payload?.targetUserId;
      if (!targetId) return;
      if (!slaveSubscribers.has(targetId)) slaveSubscribers.set(targetId, new Set());
      slaveSubscribers.get(targetId).add(ws);

      // If slave already connected, send its handshake to master
      const slaveWs = slaves.get(targetId);
      if (slaveWs) {
        safeSend(ws, {
          type: 'HANDSHAKE',
          payload: { userId: targetId, role: 'slave', deviceInfo: {}, capabilities: [] },
          timestamp: Date.now(),
          senderId: 'relay',
        });
      }
      console.log(`[Relay] Master ${userId} subscribed to slave ${targetId}`);
      return;
    }

    // ── HEARTBEAT ─────────────────────────────────
    if (msg.type === 'HEARTBEAT') {
      safeSend(ws, { type: 'HEARTBEAT_ACK', payload: { ts: msg.payload.ts }, timestamp: Date.now(), senderId: 'relay' });
      return;
    }

    // ── TIME_SYNC_REQUEST (master → relay → slave) ─
    if (msg.type === 'TIME_SYNC_REQUEST') {
      const targetId = msg.payload?.targetUserId ?? null;
      const slaveWs = targetId ? slaves.get(targetId) : null;
      if (slaveWs) {
        forwardToSlave(slaveWs, msg, userId);
      } else {
        // Relay itself responds (approximate)
        const t2 = Date.now();
        safeSend(ws, {
          type: 'TIME_SYNC_RESPONSE',
          payload: { requestId: msg.payload.requestId, t1: msg.payload.t1, t2 },
          timestamp: t2,
          senderId: 'relay',
        });
      }
      return;
    }

    // ── SENSOR_DATA / CAMERA_FRAME (slave → masters) ─
    if (msg.type === 'SENSOR_DATA' || msg.type === 'CAMERA_FRAME' || msg.type === 'REPLAY_CHUNK') {
      if (role !== 'slave') return;
      const tagged = { ...msg, serverTimestamp: Date.now() };
      const subs = slaveSubscribers.get(userId) ?? new Set();
      subs.forEach((masterWs) => forwardToMaster(masterWs, tagged));
      return;
    }

    // ── COMMAND (master → slave) ──────────────────
    if (msg.type === 'COMMAND') {
      const targetId = msg.payload?.targetUserId;
      const slaveWs = targetId ? slaves.get(targetId) : null;
      if (slaveWs) forwardToSlave(slaveWs, msg, userId);
      return;
    }

    // ── ALERT (slave → masters) ───────────────────
    if (msg.type === 'ALERT') {
      if (role !== 'slave') return;
      const subs = slaveSubscribers.get(userId) ?? new Set();
      subs.forEach((masterWs) => forwardToMaster(masterWs, msg));
    }
  });

  ws.on('close', () => {
    if (role === 'slave' && userId) {
      slaves.delete(userId);
      // Notify masters
      const subs = slaveSubscribers.get(userId) ?? new Set();
      subs.forEach((masterWs) => {
        safeSend(masterWs, {
          type: 'ERROR',
          payload: { message: `Slave ${userId} disconnected` },
          timestamp: Date.now(),
          senderId: 'relay',
        });
      });
      console.log(`[Relay] Slave disconnected: ${userId}`);
    }
    if (role === 'master' && userId) {
      // Remove from all slave subscriber sets
      for (const subs of slaveSubscribers.values()) subs.delete(ws);
      console.log(`[Relay] Master disconnected: ${userId}`);
    }
    clients.delete(connectionId);
  });

  ws.on('error', (err) => console.error(`[Relay] WS error for ${userId}:`, err.message));
});

function safeSend(ws, data) {
  if (ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify(data)); } catch { /* ignore */ }
  }
}

function forwardToMaster(masterWs, msg) {
  safeSend(masterWs, msg);
}

function forwardToSlave(slaveWs, msg, fromUserId) {
  safeSend(slaveWs, { ...msg, fromMaster: fromUserId });
}

// Stats endpoint (HTTP upgrade fallback)
wss.on('listening', () => {
  console.log(`[Relay] Ready — ${slaves.size} slaves, ${clients.size} total connections`);
});

setInterval(() => {
  const deadClients = [];
  clients.forEach((c, id) => {
    if (c.ws.readyState !== WebSocket.OPEN) deadClients.push(id);
  });
  deadClients.forEach(id => clients.delete(id));
  if (deadClients.length > 0) console.log(`[Relay] Cleaned ${deadClients.length} dead connections`);
}, 30_000);
