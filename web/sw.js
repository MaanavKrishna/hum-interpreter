// Offline support. After one visit, Hum works with no connection at all:
// the model, the app, and the sample recordings are served from this cache.
const CACHE = 'hum-v4';
const SHELL = ['./', 'index.html', 'styles.css', 'favicon.svg', 'js/app.js', 'js/engine.js', 'js/audio.js', 'data/bundle.json', 'model/hum.onnx', 'model/types.onnx', 'manifest.webmanifest'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await cache.addAll(SHELL);
      // Sample recordings too, so the demo voices play offline. Best effort.
      try {
        const bundle = await (await fetch('data/bundle.json')).json();
        await cache.addAll(bundle.people.flatMap((p) => p.clips.map((c) => c.file)));
      } catch {
        /* clips will be cached as they are played */
      }
      await self.skipWaiting();
    })()
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const sameOrigin = new URL(e.request.url).origin === self.location.origin;
  const save = (res) => {
    if (res.ok || res.type === 'opaque') caches.open(CACHE).then((c) => c.put(e.request, res.clone()));
    return res;
  };
  if (sameOrigin) {
    // Network first so a new deploy shows up at once; cache when offline.
    e.respondWith(fetch(e.request).then(save).catch(async () => (await caches.match(e.request, { ignoreSearch: true })) ?? Response.error()));
  } else {
    // Runtime and fonts from CDNs are versioned: cache first.
    e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then(save)));
  }
});
