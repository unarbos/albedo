import { POLL_MS, BUILD_ID, HTML_ENDPOINTS } from "./config.js";
import { fetchDashboard, fetchLlmsText, fetchSwebenchLite, initEndpointCache } from "./fetch.js";
import { render } from "./render.js";

async function poll() {
  const buster = Date.now();
  const [data, swebenchLite] = await Promise.all([
    fetchDashboard(buster),
    fetchSwebenchLite(buster),
  ]);
  if (data) render(data, swebenchLite);
}

async function checkVersion() {
  if (!BUILD_ID || BUILD_ID.indexOf("__") === 0) return;
  for (const url of HTML_ENDPOINTS) {
    try {
      const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) continue;
      const html = await r.text();
      const m = html.match(/<meta name="build" content="([^"]+)"/);
      if (m && m[1] && m[1].indexOf("__") !== 0 && m[1] !== BUILD_ID) {
        if (document.visibilityState === "visible") location.reload();
      }
      return;
    } catch {}
  }
}

async function copyLlmsTxt(e) {
  e.preventDefault();
  const btn = document.getElementById("hero-llms-btn");
  const label = btn?.querySelector(".hero-llms-label");
  if (!btn || !label) return;
  const orig = label.textContent;
  btn.disabled = true;
  try {
    const text = await fetchLlmsText();
    if (!text) throw new Error("could not load llms.txt");
    await navigator.clipboard.writeText(text);
    label.textContent = "copied";
    btn.classList.add("copied");
  } catch {
    label.textContent = "copy failed";
  }
  setTimeout(() => {
    label.textContent = orig;
    btn.classList.remove("copied");
    btn.disabled = false;
  }, 1600);
}

document.getElementById("hero-llms-btn")?.addEventListener("click", copyLlmsTxt);

initEndpointCache();
poll();
setInterval(poll, POLL_MS);
setInterval(checkVersion, 60_000);
