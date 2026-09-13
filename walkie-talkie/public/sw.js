// Cache the whole app shell so the PWA opens with no server reachable at all.
// Network-first for same-origin requests keeps development sane; the cache is
// what makes the app usable once the infrastructure is gone.

const CACHE = 'nearby-ptt-v1';
const SHELL = [
  '/',
  '/index.html',
  '/styles.css',
  '/manifest.webmanifest',
  '/src/app.js',
  '/src/mesh.js',
  '/src/ptt.js',
  '/src/player.js',
  '/src/protocol.js',
  '/src/link-quality.js',
  '/src/stt.js',
  '/src/tts.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match('/index.html'))),
  );
});
