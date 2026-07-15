import { DATA_ENDPOINTS, STATE_ENDPOINTS, BENCHMARK_ENDPOINTS, MANIFEST_ENDPOINTS, LLMS_URLS, REGISTRATION_ENDPOINTS } from "./config.js";

let llmsTextCache = null;
const registrationCacheKey = "albedo.registrationHistory.v2";

async function fetchFirstJson(endpoints) {
  const buster = Date.now();
  for (const url of endpoints) {
    try {
      const r = await fetch(url + "?t=" + buster, { cache: "no-store" });
      if (!r.ok) continue;
      return await r.json();
    } catch {}
  }
  return null;
}

export async function fetchDashboard() {
  return fetchFirstJson(DATA_ENDPOINTS);
}

export async function fetchState() {
  return fetchFirstJson(STATE_ENDPOINTS);
}

export async function fetchBenchmarks() {
  return fetchFirstJson(BENCHMARK_ENDPOINTS);
}

export async function fetchManifest() {
  return fetchFirstJson(MANIFEST_ENDPOINTS);
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

export async function fetchRegistrationHistory() {
  for (const url of REGISTRATION_ENDPOINTS) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      const data = await r.json();
      try { localStorage.setItem(registrationCacheKey, JSON.stringify(data)); } catch {}
      return data;
    } catch {}
  }
  try { return JSON.parse(localStorage.getItem(registrationCacheKey)); } catch { return null; }
}

export async function fetchText(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.text();
  } catch { return null; }
}

export async function fetchJson(url) {
  const t = await fetchText(url);
  if (t == null) return null;
  try { return JSON.parse(t); } catch { return null; }
}
