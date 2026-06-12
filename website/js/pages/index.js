import { POLL_MS } from "../config.js";
import { fetchDashboard } from "../fetch.js";
import { normalize } from "../data.js";
import { el, mount } from "../dom.js";
import { fmtRelative } from "../format.js";
import { kingTitleName, hubRepoUrl, modelRepo } from "../model.js";
import { renderReign } from "../render/reign.js";
import { renderChart } from "../render/chart.js";
import { renderQueue } from "../render/queue.js";
import { renderHistory, renderFails } from "../render/history.js";

const $ = id => document.getElementById(id);

let state = null;
let filter = "";

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
  const netuid = d.chain.netuid;
  renderHero(d);
  renderStats(d);
  renderReign($("reign-wrap"), d.reign, netuid);
  renderChart($("chart-wrap"), d.crownings);
  renderQueue($("queue-wrap"), d.queue, d.currentEval, netuid);
  $("queue-meta").textContent = `${d.queue.length}${d.currentEval ? " + 1 live" : ""} pending`;
  renderTables(d);
  if (d.updatedAt) $("updated").textContent = "updated " + fmtRelative(d.updatedAt);
}

let lastSig = null;
async function tick() {
  const raw = await fetchDashboard();
  if (!raw) return;
  const sig = JSON.stringify(raw);
  if (sig === lastSig) return;
  lastSig = sig;
  state = normalize(raw);
  render(state);
}

function wireFilter() {
  const input = $("filter-input");
  if (!input) return;
  input.addEventListener("input", () => {
    filter = input.value.trim().toLowerCase();
    if (state) renderTables(state);
  });
}

wireFilter();
tick();
setInterval(tick, POLL_MS);
