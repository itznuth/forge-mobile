/* F.O.R.G.E. Mobile service worker: makes the app load offline after the first visit.
 * - App files: network first (so updates show up), cache as fallback.
 * - Tools model: cache first, refreshed in the background.
 * - CDN libraries and COCO-SSD weights: cache first (their URLs are versioned).
 * Bump VERSION when the list below changes.
 */
const VERSION = "forge-v2";
const SHELL = ["./", "index.html", "forge.js", "manifest.json", "icon-192.png", "icon-512.png", "apple-touch-icon.png", "model/classes.json"];
// fetched before this worker exists on the very first visit, so download them up front (best effort)
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const EXTRA = [
  "model/best.onnx",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js",
  ORT + "ort.min.js", ORT + "ort-wasm-simd-threaded.mjs", ORT + "ort-wasm-simd-threaded.wasm",
];
const CDN_HOSTS = ["cdn.jsdelivr.net", "storage.googleapis.com", "tfhub.dev", "www.kaggle.com"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then(async (c) => {
    await c.addAll(SHELL);
    await Promise.all(EXTRA.map((u) => c.add(new Request(u, { mode: "cors" })).catch(() => {})));
  }).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

const put = (req, res) => { if (res && res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); } return res; };

async function networkFirst(req) {
  try {
    const res = await Promise.race([fetch(req), new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 5000))]);
    return put(req, res);
  } catch (err) {
    const hit = await caches.match(req, { ignoreSearch: true }) || (req.mode === "navigate" && await caches.match("index.html"));
    if (hit) return hit;
    throw err;
  }
}

async function cacheFirst(req, refresh) {
  const hit = await caches.match(req);
  if (hit) {
    if (refresh) fetch(req).then((res) => put(req, res)).catch(() => {});
    return hit;
  }
  return put(req, await fetch(req));
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    e.respondWith(url.pathname.endsWith(".onnx") ? cacheFirst(req, true) : networkFirst(req));
  } else if (CDN_HOSTS.includes(url.hostname)) {
    e.respondWith(cacheFirst(req, false));
  }
});
