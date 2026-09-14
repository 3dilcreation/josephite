// Boots the real server on a throwaway port and opens browser peers against it.
//
// http://127.0.0.1 is a secure context as far as the browser is concerned, so
// getUserMedia, MediaRecorder and WebRTC all behave normally without dragging
// certificates into the tests.

import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { installSpeechFakes } from './fakes.js';

const SERVER = fileURLToPath(new URL('../../server/index.js', import.meta.url));
const CHROME = process.env.CHROME_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';

export async function freePort() {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

export async function startServer() {
  const port = await freePort();
  const child = spawn(process.execPath, [SERVER], {
    env: { ...process.env, PORT: String(port), INSECURE: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('server did not start in time')), 10000);
    child.stdout.on('data', (buf) => {
      if (buf.toString().includes(`:${port}`)) { clearTimeout(timer); resolve(); }
    });
    child.once('error', reject);
  });

  return {
    origin: `http://127.0.0.1:${port}`,
    stop: () => new Promise((resolve) => { child.once('exit', resolve); child.kill('SIGTERM'); }),
  };
}

export async function launchBrowser() {
  return chromium.launch({
    executablePath: CHROME,
    args: [
      '--use-fake-device-for-media-stream',
      '--use-fake-ui-for-media-stream',
      '--autoplay-policy=no-user-gesture-required',
    ],
  });
}

// Each peer gets its own context: the app derives a stable identity from
// localStorage, so sharing one would put two "devices" under a single id.
export async function openPeer(browser, { name, channel, transcript = 'nothing', query = '', origin }) {
  const context = await browser.newContext({ permissions: ['microphone'] });
  await context.addInitScript(installSpeechFakes({ transcript }));
  const page = await context.newPage();
  page.on('pageerror', (err) => { page.__error = err; });
  await page.goto(origin + query, { waitUntil: 'domcontentloaded' });
  // Channel first: each change restarts the mesh, and settling on the final
  // channel before announcing the name keeps the join log readable.
  if (channel) {
    await page.fill('#channel', channel);
    await page.press('#channel', 'Enter');
  }
  await page.fill('#name', name);
  await page.press('#name', 'Enter');
  page.__close = () => context.close();
  return page;
}

// Waiting on a count is not enough when peers from an earlier test are still
// connected -- the count is already satisfied and the hold starts before the
// pair we care about has finished its handshake. Wait for the name.
export async function waitForPeer(page, name, timeout = 20000) {
  await page.waitForFunction(
    (n) => [...document.querySelectorAll('#peers li')].some((li) => li.textContent.startsWith(n)),
    name,
    { timeout },
  );
}

export async function hold(page, ms, during) {
  const box = await page.locator('#ptt').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  if (during) await during();
  await page.waitForTimeout(ms);
  await page.mouse.up();
}

export function entries(page) {
  return page.$$eval('#log .entry', (nodes) => nodes.map((n) => ({
    kind: n.dataset.kind,
    mine: n.dataset.mine === 'true',
    who: n.querySelector('b').textContent,
    badge: n.querySelector('.badge').textContent,
    body: n.querySelector('.entry-body').textContent,
  })));
}
