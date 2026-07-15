import { POLL_MS } from "../config.js";
import { fetchDashboard, fetchState, fetchBenchmarks, fetchManifest, fetchLlmsText, fetchRegistrationHistory } from "../fetch.js";
import { normalize } from "../data.js";
import { el, mount } from "../dom.js";
import { fmtRelative } from "../format.js";
import { kingTitleName, hubRepoUrl, modelRepo } from "../model.js";
import { renderReign } from "../render/reign.js";
import { renderBenchmarks } from "../render/benchmarks.js";
import { renderPipeline } from "../render/pipeline.js";
import { renderHistory, renderFails } from "../render/history.js";
import { renderDatasets } from "../render/datasets.js";
import { renderHeroChart } from "../render/heroChart.js";
import { renderRegistrationChart } from "../render/registrationChart.js";

const $ = id => document.getElementById(id);

let state = null;
let filter = "";
let netuid = null;
let registrations = null;

function matches(x, q) {
  if (!q) return true;
  const hay = `${x.model_uri || ""} ${x.hotkey || ""} ${x.uid ?? ""} ${x.fault_code || ""}`.toLowerCase();
  return hay.includes(q);
}

function renderHero(d) {
  const king = d.reign.members?.[0];
  const repoUrl = king && hubRepoUrl(king.model_uri);
  mount($("hero-king"),
    king
      ? (repoUrl ? el("a", { href: repoUrl, target: "_blank", rel: "noopener" }, kingTitleName(king.king_version))
                 : kingTitleName(king.king_version))
      : "ALBEDO");
  mount($("hero-sub"), king ? modelRepo(king.model_uri) : "");
}

function renderStats(d) {
  const s = d.stats || {};
  const chip = (k, v) => el("span", { class: "stat-chip" }, el("span", { class: "k" }, k), el("b", {}, String(v ?? "—")));
  mount($("hero-stats"), chip("evaluated models", s.evaluated));
}

function renderTables(d) {
  const netuid = d.chain.netuid;
  const histRows = d.history.filter(x => matches(x, filter));
  const failRows = d.fails.filter(x => matches(x, filter));

  renderHistory($("history-wrap"), histRows, d.chain.judge_models, netuid, d.reign.members?.[0]?.eval_run_id);
  renderFails($("fails-wrap"), failRows, netuid);
  $("history-meta").textContent = `${histRows.length} shown`;
  $("fails-meta").textContent = `${failRows.length} shown`;
}

function render(d) {
  netuid = d.chain.netuid;
  renderHero(d);
  renderStats(d);
  renderHeroChart($("hero-chart"), d.history);
  renderReign($("reign-wrap"), d.reign, netuid);
  renderTables(d);
  if (d.updatedAt) $("updated").textContent = "updated " + fmtRelative(d.updatedAt);
}

let lastSig = null;
let benchmarkSig = null;
async function tick() {
  const raw = await fetchDashboard();
  if (!raw) return;
  const sig = JSON.stringify(raw);
  if (sig === lastSig) return;
  lastSig = sig;
  state = normalize(raw);
  render(state);
}

async function tickBenchmarks() {
  const data = await fetchBenchmarks();
  if (!data) return;
  const sig = JSON.stringify(data);
  if (sig === benchmarkSig) return;
  benchmarkSig = sig;
  renderBenchmarks($("benchmarks-wrap"), $("benchmarks-meta"), data);
}

async function loadDatasets() {
  const manifest = await fetchManifest();
  if (!manifest) return;
  renderDatasets($("datasets-wrap"), $("datasets-meta"), manifest);
}

async function loadRegistrations() {
  registrations = await fetchRegistrationHistory();
  renderRegistrationChart($("registration-chart"), registrations);
}

async function tickPipeline() {
  const st = await fetchState();
  if (!st) return;
  renderPipeline($("pipeline-wrap"), st, netuid ?? 97);
  const c = st.counts || {};
  const total = Object.values(c).reduce((sum, x) => sum + (Number(x.running) || 0) + (Number(x.queued) || 0), 0);
  $("pipeline-meta").textContent = total ? `${total} in queue` : "queue idle";
}

async function writeClipboard(text) {
  // navigator.clipboard exists only in a secure context (https or localhost).
  // Fall back to execCommand so copy still works over a LAN IP / plain http.
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {}
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

async function copyLlmsTxt(e) {
  e.preventDefault();
  const btn = $("hero-llms-btn");
  const label = btn?.querySelector(".hero-llms-label");
  if (!btn || !label) return;
  const orig = label.textContent;
  btn.disabled = true;
  try {
    const text = await fetchLlmsText();
    if (!text) throw new Error("could not load llms.txt");
    if (!(await writeClipboard(text))) throw new Error("clipboard write failed");
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

function wireFilter() {
  const input = $("filter-input");
  if (!input) return;
  input.addEventListener("input", () => {
    filter = input.value.trim().toLowerCase();
    if (state) renderTables(state);
  });
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state) renderHeroChart($("hero-chart"), state.history);
    if (registrations) renderRegistrationChart($("registration-chart"), registrations);
  }, 150);
});

wireFilter();
$("hero-llms-btn")?.addEventListener("click", copyLlmsTxt);
tick();
tickPipeline();
tickBenchmarks();
loadDatasets();
loadRegistrations();
setInterval(tick, POLL_MS);
setInterval(tickPipeline, POLL_MS);
setInterval(tickBenchmarks, POLL_MS);
