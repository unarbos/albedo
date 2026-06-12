import { DATA_ENDPOINTS } from "./config.js";

export async function fetchDashboard() {
  const buster = Date.now();
  for (const url of DATA_ENDPOINTS) {
    try {
      const r = await fetch(url + "?t=" + buster, { cache: "no-store" });
      if (!r.ok) continue;
      return await r.json();
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
