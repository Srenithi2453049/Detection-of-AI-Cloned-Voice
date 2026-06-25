self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("df-detector-v1").then((cache) =>
      cache.addAll([
        "/",
        "/script.js",
        "/styles.css",
        "/manifest.json"
      ]),
    ),
  );
});
self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});
self.addEventListener("fetch", (event) => {
  const { request } = event;
  // Only cache GET requests for static assets; always go to network for API
  if (request.method !== "GET") return;
  if (request.url.includes("/predict") || request.url.includes("/health")) return;
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        const responseClone = response.clone();
        caches.open("df-detector-v1").then((cache) => cache.put(request, responseClone));
        return response;
      });
    }),
  );
});