// Introduction service for the mesh, plus a static host for the PWA itself.
//
// Runs on any machine sharing the Wi-Fi or hotspot -- a laptop, a spare phone in
// Termux, a Pi in a backpack. It never sees message content: once two browsers
// have exchanged SDP they talk directly, and killing this process leaves running
// conversations intact.

import { createServer } from 'node:https';
import { createServer as createHttpServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { networkInterfaces } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const here = fileURLToPath(new URL('.', import.meta.url));
const PUBLIC_DIR = resolve(here, '..', 'public');
const CERT_DIR = join(here, 'certs');
const PORT = Number(process.env.PORT || 8443);
const INSECURE = process.env.INSECURE === '1';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

async function handle(req, res) {
  const url = new URL(req.url, 'https://placeholder');
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';

  // normalize() collapses any ../ before we join, so a crafted path cannot
  // escape the public directory.
  const target = join(PUBLIC_DIR, normalize(pathname));
  if (!target.startsWith(PUBLIC_DIR)) {
    res.writeHead(403).end('forbidden');
    return;
  }

  try {
    const info = await stat(target);
    if (!info.isFile()) throw new Error('not a file');
    const body = await readFile(target);
    res.writeHead(200, {
      'content-type': MIME[extname(target)] || 'application/octet-stream',
      'cache-control': 'no-cache',
      // The PWA is entirely self-hosted apart from the optional voice pack.
      'service-worker-allowed': '/',
    });
    res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
  }
}

function buildServer() {
  if (INSECURE) return createHttpServer(handle);
  const key = join(CERT_DIR, 'key.pem');
  const cert = join(CERT_DIR, 'cert.pem');
  if (!existsSync(key) || !existsSync(cert)) {
    console.error('No certificate found. Run: npm run certs');
    console.error('(or start with INSECURE=1 for a plain-HTTP run on localhost only)');
    process.exit(1);
  }
  return createServer({ key: readFileSync(key), cert: readFileSync(cert) }, handle);
}

const server = buildServer();
const wss = new WebSocketServer({ server, path: '/signal' });

// channel -> Map(peerId -> { socket, name })
const channels = new Map();

wss.on('connection', (socket) => {
  let peerId = null;
  let channel = null;

  socket.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.kind === 'join') {
      channel = String(msg.channel || 'main').slice(0, 64);
      const room = channels.get(channel) || new Map();
      channels.set(channel, room);

      // Two clients can genuinely arrive with the same id (a restored backup, a
      // second tab on one phone). Displacing the first one silently would look
      // like a random disconnect, so the newcomer is given a distinct id and
      // told what it is.
      peerId = String(msg.id).slice(0, 64);
      while (room.has(peerId)) peerId = `${String(msg.id).slice(0, 56)}-${Math.random().toString(36).slice(2, 6)}`;
      send(socket, { kind: 'joined', id: peerId });

      // Tell the newcomer who is already here; they open the offers, which
      // keeps exactly one side dialling and avoids offer glare.
      send(socket, {
        kind: 'peers',
        peers: [...room.entries()].map(([id, p]) => ({ id, name: p.name })),
      });
      for (const [, peer] of room) {
        send(peer.socket, { kind: 'peer-joined', peer: { id: peerId, name: msg.name } });
      }
      room.set(peerId, { socket, name: msg.name });
      log(`${msg.name} (${peerId.slice(0, 6)}) joined #${channel} — ${room.size} on channel`);
      return;
    }

    // Everything else is a routed handshake message.
    if (!channel || !msg.to) return;
    const room = channels.get(channel);
    const target = room?.get(String(msg.to));
    if (target) send(target.socket, { ...msg, from: peerId });
  });

  socket.on('close', () => {
    const room = channels.get(channel);
    if (!room || !peerId) return;
    room.delete(peerId);
    for (const [, peer] of room) send(peer.socket, { kind: 'peer-left', id: peerId });
    if (!room.size) channels.delete(channel);
    log(`${peerId.slice(0, 6)} left #${channel}`);
  });
});

function send(socket, obj) {
  if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(obj));
}

function log(line) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${line}`);
}

server.listen(PORT, '0.0.0.0', () => {
  const scheme = INSECURE ? 'http' : 'https';
  console.log('\n  Nearby PTT signalling server\n');
  for (const [name, list] of Object.entries(networkInterfaces())) {
    for (const ni of list || []) {
      if (ni.family === 'IPv4') console.log(`    ${scheme}://${ni.address}:${PORT}   (${name})`);
    }
  }
  if (!INSECURE) {
    console.log('\n  The certificate is self-signed: each phone has to tap through');
    console.log('  the browser warning once before the microphone will work.\n');
  }
});
