import { DATA_ENDPOINTS, STATE_ENDPOINTS, BENCHMARK_ENDPOINTS, MODEL_SCORE_ENDPOINTS, MANIFEST_ENDPOINTS, LLMS_URLS, REGISTRATION_ENDPOINTS, PREDS_ENDPOINTS } from "./config.js";

let llmsTextCache = null;
const registrationCacheKey = "albedo.registrationHistory.v2";

async function fetchFirstJson(endpoints, { revalidate = false } = {}) {
  const suffix = revalidate ? "" : "?t=" + Date.now();
  for (const url of endpoints) {
    try {
      const r = await fetch(url + suffix, { cache: revalidate ? "no-cache" : "no-store" });
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
  return fetchFirstJson(BENCHMARK_ENDPOINTS, { revalidate: true });
}

export async function fetchModelScores() {
  return fetchFirstJson(MODEL_SCORE_ENDPOINTS, { revalidate: true });
}

export async function fetchBenchmarkRun(run) {
  if (!run?.detail_path) return null;
  for (const endpoint of BENCHMARK_ENDPOINTS) {
    const base = endpoint.slice(0, endpoint.lastIndexOf("/") + 1);
    try {
      const r = await fetch(base + run.detail_path, { cache: "no-cache" });
      if (!r.ok) continue;
      const payload = await r.json();
      return payload?.run || payload;
    } catch {}
  }
  return null;
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

const PRED_MARKER = '"instance_id":';
const predsState = new Map();

function countMarkers(text) {
  let n = 0;
  for (let i = text.indexOf(PRED_MARKER); i !== -1; i = text.indexOf(PRED_MARKER, i + PRED_MARKER.length)) n++;
  return n;
}

async function readPredsTail(url, offset) {
  const from = Math.max(0, offset - PRED_MARKER.length + 1);
  const r = await fetch(url, { cache: "no-store", headers: from ? { Range: `bytes=${from}-` } : {} });
  if (!r.ok) return null;
  const text = await r.text();
  const end = Number(r.headers.get("content-range")?.split("/")[0]?.split("-")[1]);
  return {
    found: countMarkers(text),
    partial: from > 0 && r.status === 206,
    offset: Number.isFinite(end) ? end + 1 : from + text.length,
  };
}

export async function fetchPredsProgress(runId) {
  if (!runId) return null;
  for (const base of PREDS_ENDPOINTS) {
    const url = `${base}/${runId}/preds.json`;
    try {
      const head = await fetch(url, { method: "HEAD", cache: "no-store" });
      if (!head.ok) continue;
      const size = Number(head.headers.get("content-length"));
      if (!Number.isFinite(size) || size <= 0) continue;
      const prev = predsState.get(url);
      const state = prev && size >= prev.offset ? prev : { offset: 0, count: 0, first: null };
      if (size > state.offset) {
        const tail = await readPredsTail(url, state.offset);
        if (!tail) continue;
        state.count = tail.partial ? state.count + tail.found : tail.found;
        state.offset = tail.offset;
      }
      state.first ||= { at: Date.now(), count: state.count };
      predsState.set(url, state);
      const elapsed = Date.now() - state.first.at;
      const gained = state.count - state.first.count;
      return {
        runId,
        count: state.count,
        updatedAt: head.headers.get("last-modified") || null,
        rate: elapsed > 30000 && gained > 0 ? gained / elapsed : null,
      };
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
