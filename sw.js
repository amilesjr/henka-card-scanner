// Keeps Henka Crop working offline and opening instantly from the Home
// Screen. Each release has its own cache; old ones are removed.
const CACHE = "henka-crop-202610270225";
const FILES = [
  "./",
  "index.html",
  "styles.css?v=202610270225",
  "app.js?v=202610270225",
  "zip.js?v=202610270225",
  "worker.js?v=202610270225",
  "engine.js?v=202610270225",
  "manifest.webmanifest",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/apple-touch-icon.png",
  "fonts/shippori-mincho-latin-700.woff2",
  "fonts/zen-kaku-gothic-new-latin-400.woff2",
  "fonts/zen-kaku-gothic-new-latin-700.woff2",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("henka-crop-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  // The page itself: the newest version when online, the saved one offline.
  if (req.mode === "navigate") {
    e.respondWith(fetch(req).catch(() => caches.match("index.html")));
    return;
  }
  e.respondWith(caches.match(req).then((hit) => hit || fetch(req)));
});
