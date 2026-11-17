// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 vehicle-plate-ocr-jp-v1 contributors.
self.addEventListener('install',event=>event.waitUntil(self.skipWaiting()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=>{
  event.respondWith((async()=>{
    const res=await fetch(event.request);
    if(res.status===0||!res.url.startsWith(self.location.origin)) return res;
    const headers=new Headers(res.headers);
    headers.set('Cross-Origin-Opener-Policy','same-origin');
    headers.set('Cross-Origin-Embedder-Policy','credentialless');
    headers.set('Cross-Origin-Resource-Policy','same-origin');
    return new Response(res.body,{status:res.status,statusText:res.statusText,headers});
  })());
});
