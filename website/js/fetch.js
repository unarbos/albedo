import { DATA_ENDPOINTS, STATE_ENDPOINTS, LLMS_URLS } from "./config.js";

let llmsTextCache = null;

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
