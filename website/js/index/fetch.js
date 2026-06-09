import { DATA_ENDPOINTS, ENDPOINT_CACHE_KEY, LLMS_URLS, SWEBENCH_LITE_URLS } from "./config.js";

let DATA_URL = null;
export let llmsTextCache = null;
let SWEBENCH_LITE_CACHE = null;

export function loadCachedEndpoint() {
  try {
    const raw = localStorage.getItem(ENDPOINT_CACHE_KEY);
    if (!raw) return null;
    const o = JSON.parse(raw);
    if (Date.now() - o.ts > 3600 * 1000) return null;
    return o.url;
  } catch { return null; }
}

function saveCachedEndpoint(url) {
  try { localStorage.setItem(ENDPOINT_CACHE_KEY, JSON.stringify({ url, ts: Date.now() })); } catch {}
}

export async function fetchDashboard(buster) {
  const seen = new Set();
  const urls = [];
  const add = u => { if (!seen.has(u)) { seen.add(u); urls.push(u); } };
  add("../dashboard.json");
  if (DATA_URL) add(DATA_URL);
  DATA_ENDPOINTS.forEach(add);
  for (const url of urls) {
    try {
      const r = await fetch(url + "?t=" + buster, { cache: "no-store" });
      if (!r.ok) continue;
      const data = await r.json();
      if (url !== "../dashboard.json") {
        DATA_URL = url;
        saveCachedEndpoint(url);
      }
      return data;
    } catch {}
  }
  return null;
}

export async function fetchLlmsText() {
  if (llmsTextCache) return llmsTextCache;
  for (const url of LLMS_URLS) {
    try {
      const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) continue;
      llmsTextCache = await r.text();
      return llmsTextCache;
    } catch {}
  }
  return null;
}

export async function fetchSwebenchLite(buster) {
  for (const url of SWEBENCH_LITE_URLS) {
    try {
      const r = await fetch(url + "?t=" + buster, { cache: "no-store" });
      if (!r.ok) continue;
      SWEBENCH_LITE_CACHE = await r.json();
      return SWEBENCH_LITE_CACHE;
    } catch {}
  }
  return SWEBENCH_LITE_CACHE;
}

export function initEndpointCache() {
  DATA_URL = loadCachedEndpoint();
}
