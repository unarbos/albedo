import { el, mount } from "../dom.js";
import { pct, fmtDateTime, fmtRelative, toRoman } from "../format.js";
import { modelRepo } from "../model.js";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  swe_rebench_2026_03: "SWE-rebench",
};

const BENCHMARK_ORDER = ["tau2_airline", "tau2_retail", "tau2_telecom"];

const PAGE_SIZES = [5, 10, 25, 50];
const ACTIVE_STATES = new Set(["QUEUED", "CLAIMED", "LOADING_MODEL", "RUNNING", "SCORING"]);

let historyOpen = localStorage.getItem("benchPanelHistoryOpen") !== "0";
let historyPage = Math.max(1, Number(localStorage.getItem("benchPanelHistoryPage")) || 1);
let historyPageSize = Number(localStorage.getItem("benchPanelHistoryPageSize")) || 10;
if (!PAGE_SIZES.includes(historyPageSize)) historyPageSize = 10;

function benchmarkLabel(suite) {
  return BENCHMARK_LABELS[suite] || suite || "—";
}

function modelName(model) {
  return model?.model_repo || modelRepo(model?.model_uri) || model?.model_uri || "—";
}

const ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };

function romanToInt(value) {
  let total = 0;
  let previous = 0;
  for (const char of value.toUpperCase().split("").reverse()) {
    const current = ROMAN_VALUES[char] || 0;
    total += current < previous ? -current : current;
    previous = Math.max(previous, current);
  }
  return total;
}

// benchmarks.json king labels lag the chain numbering by one (its "King LXXIV" is chain king LXXV).
function modelLabel(model) {
  const label = model?.label || "—";
  const match = /^King\s+([IVXLCDM]+)$/i.exec(label);
  if (!match) return label;
  return `King ${toRoman(romanToInt(match[1]) + 1)}`;
}

function hfRepoUrl(model) {
  return model?.model_repo ? `https://huggingface.co/${model.model_repo}` : null;
}

function completedRuns(model) {
  return (model?.runs || []).filter(run => run.score != null || Number(run.task_count || 0) > 0 || run.finished_at);
}

function progressKey(modelRepo, suite) {
  return `${modelRepo || ""}\n${suite || ""}`;
}

function activeState(item) {
  return String(item?.phase || item?.state || "").toUpperCase();
}

function isActiveProgress(item) {
  return ACTIVE_STATES.has(activeState(item));
}

function activeProgressByModelSuite(data) {
  const out = new Map();
  for (const source of [...(data?.jobs || []), ...(data?.workers || [])]) {
    if (!source?.model_repo || !source?.suite || !isActiveProgress(source)) continue;
    out.set(progressKey(source.model_repo, source.suite), source);
  }
  return out;
}

function hasActiveProgress(model, activeProgress) {
  return BENCHMARK_ORDER.some(suite => activeProgress.has(progressKey(model?.model_repo || model?.id, suite)));
}

function progressLabel(progress) {
  const done = Number(progress?.progress_done);
  const total = Number(progress?.progress_total);
  if (Number.isFinite(done) && Number.isFinite(total) && total > 0) {
    return `${Math.max(0, Math.min(100, Math.round((done / total) * 100)))}%`;
  }
  const state = activeState(progress).toLowerCase().replaceAll("_", " ");
  return state || "active";
}

function progressMeta(progress) {
  const seconds = Number(progress?.seconds_since_last_progress);
  if (Number.isFinite(seconds)) return `${Math.max(0, Math.round(seconds))}s since progress`;
  if (progress?.updated_at) return fmtRelative(progress.updated_at);
  return progress?.worker_id || "active";
}

function latestRun(model) {
  return completedRuns(model).sort((a, b) => {
    const at = new Date(a.finished_at || a.started_at || "").getTime();
    const bt = new Date(b.finished_at || b.started_at || "").getTime();
    if (Number.isFinite(bt - at) && bt !== at) return bt - at;
    return Number(b.run_attempt || 0) - Number(a.run_attempt || 0);
  })[0] || null;
}

function latestRunTime(model) {
  const run = latestRun(model);
  return run?.finished_at || run?.started_at || model?.activated_at || model?.discovered_at || "";
}

function sortModels(models) {
  return [...(models || [])].sort((a, b) => {
    if (isGenesis(a) !== isGenesis(b)) return isGenesis(a) ? 1 : -1;
    const orderDelta = Number(a.model_order ?? 999999) - Number(b.model_order ?? 999999);
    if (orderDelta) return orderDelta;
    const timeDelta = new Date(latestRunTime(b)).getTime() - new Date(latestRunTime(a)).getTime();
    return Number.isFinite(timeDelta) ? timeDelta : 0;
  });
}

function isGenesis(model) {
  const identity = `${model?.label || ""} ${model?.model_repo || ""}`.toLowerCase();
  return identity.includes("genesis") || identity.includes("qwen/qwen3.6-35b-a3b");
}

function detailHref(model, runId = null) {
  const qs = new URLSearchParams();
  if (model?.id) qs.set("model_id", model.id);
  if (runId) qs.set("run_id", runId);
  return `./benchmark.html?${qs.toString()}`;
}

function suiteScores(model) {
  return model?.latest_scores || {};
}

function latestScoreDate(model) {
  const dates = Object.values(suiteScores(model)).map(entry => entry?.finished_at).filter(Boolean).sort();
  return dates[dates.length - 1] || null;
}

function hasPanelScores(model) {
  const scores = suiteScores(model);
  return BENCHMARK_ORDER.some(suite => scores[suite]?.score != null);
}

function panelScore(value) {
  return `${pct(value, 1)}%`;
}

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function renderTile(model, suite, progress) {
  const entry = suiteScores(model)[suite];
  const scored = entry?.score != null;
  if (!scored && progress) {
    return el("div", { class: "bench-tile", "data-status": "running" },
      el("div", { class: "bench-tile-name" }, benchmarkLabel(suite)),
      el("div", { class: "bench-tile-score" }, progressLabel(progress)),
      el("div", { class: "bench-tile-status" },
        el("span", { class: "live" }, activeState(progress).toLowerCase().replaceAll("_", " ") || "active"),
        el("span", {}, progressMeta(progress))));
  }
  const href = entry?.run_id ? detailHref(model, entry.run_id) : null;
  return el(href ? "a" : "div", { class: "bench-tile", "data-status": scored ? "completed" : "missing", href },
    el("div", { class: "bench-tile-name" }, benchmarkLabel(suite)),
    el("div", { class: "bench-tile-score" }, scored ? panelScore(entry.score) : "missing"),
    el("div", { class: "bench-tile-status" },
      scored
        ? [el("span", { class: "ok" }, "succeeded"),
           el("span", {}, `${entry.passed_count ?? "—"}/${entry.task_count ?? "—"}`)]
        : [el("span", {}, "—"), el("span", {}, "no run")]));
}

function renderSparks(sorted) {
  const chrono = [...sorted].reverse();
  return el("div", { class: "bench-spark-grid" }, BENCHMARK_ORDER.map(suite => {
    const points = chrono
      .map(model => ({ label: modelLabel(model), score: suiteScores(model)[suite]?.score }))
      .filter(point => point.score != null);
    const latest = points[points.length - 1];
    const svg = svgEl("svg", { viewBox: "0 0 360 34", preserveAspectRatio: "none", role: "img" });
    svg.append(svgEl("line", { x1: 6, y1: 26, x2: 354, y2: 26, stroke: "currentColor", "stroke-width": 1, opacity: 0.15 }));
    if (!points.length) {
      svg.append(svgEl("text", { x: 180, y: 20, "text-anchor": "middle", "font-size": 8, fill: "currentColor", opacity: 0.45 }, "no score"));
    } else {
      let min = Math.min(...points.map(point => point.score));
      let max = Math.max(...points.map(point => point.score));
      if (min === max) { min -= 0.005; max += 0.005; }
      const span = max - min;
      const coords = points.map((point, i) => ({
        x: points.length === 1 ? 180 : 6 + (i / (points.length - 1)) * 348,
        y: 26 - ((point.score - min) / span) * 20,
        point,
      }));
      if (coords.length > 1) {
        svg.append(svgEl("polyline", {
          points: coords.map(c => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" "),
          fill: "none", stroke: "currentColor", "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round",
        }));
      }
      coords.forEach((c, i) => {
        const last = i === coords.length - 1;
        svg.append(svgEl("circle", {
          cx: c.x.toFixed(1), cy: c.y.toFixed(1), r: 2.4,
          fill: "currentColor", opacity: last ? 1 : 0.45,
          class: last ? "spark-dot-last" : null,
        }, svgEl("title", {}, `${c.point.label} · ${panelScore(c.point.score)}`)));
      });
    }
    return el("div", { class: "bench-spark" },
      el("div", { class: "bench-spark-head" },
        el("strong", {}, benchmarkLabel(suite)),
        el("span", {}, latest ? panelScore(latest.score) : "waiting")),
      svg);
  }));
}

function renderHistoryPanel(sorted, selectedModel, rerender) {
  const pages = Math.max(1, Math.ceil(sorted.length / historyPageSize));
  historyPage = Math.min(Math.max(1, historyPage), pages);
  const shown = sorted.slice((historyPage - 1) * historyPageSize, historyPage * historyPageSize);

  const setPage = page => {
    historyPage = page;
    localStorage.setItem("benchPanelHistoryPage", String(historyPage));
    rerender();
  };
  const pager = el("div", { class: "bench-history-pager" },
    el("div", { class: "bench-history-pager-left" },
      el("button", { type: "button", disabled: historyPage <= 1, onClick: () => setPage(historyPage - 1) }, "prev"),
      el("span", {}, `page ${historyPage} / ${pages} · ${sorted.length} kings`),
      el("button", { type: "button", disabled: historyPage >= pages, onClick: () => setPage(historyPage + 1) }, "next")),
    el("label", { class: "bench-history-pager-right" }, "rows",
      el("select", { onChange: e => {
        historyPageSize = Number(e.target.value);
        localStorage.setItem("benchPanelHistoryPageSize", String(historyPageSize));
        setPage(1);
      } }, PAGE_SIZES.map(size => el("option", { value: size, selected: size === historyPageSize }, String(size))))));

  const rows = shown.map(model => {
    const scores = suiteScores(model);
    const repoUrl = hfRepoUrl(model);
    return el("tr", {
      class: model.id === selectedModel?.id ? "clickable crowned-now" : "clickable",
      onClick: e => { if (!e.target.closest("a")) location.href = detailHref(model); },
    },
      el("td", { class: "bench-king-col" },
        el("a", { href: detailHref(model) }, modelLabel(model)),
        " ", el("span", { class: "muted" }, fmtDateTime(latestScoreDate(model)))),
      el("td", { class: "model" }, repoUrl
        ? el("a", { href: repoUrl, target: "_blank", rel: "noopener" }, modelName(model))
        : el("span", { class: "model-cell" }, modelName(model))),
      BENCHMARK_ORDER.map(suite => {
        const entry = scores[suite];
        if (entry?.score == null) return el("td", { class: "r" }, el("span", { class: "muted-dash" }, "—"));
        return el("td", { class: "r" }, panelScore(entry.score));
      }));
  });

  return el("div", { class: "bench-history" },
    renderSparks(sorted),
    pager,
    sorted.length
      ? el("div", { class: "data-table-wrap" },
          el("table", { class: "data-table" },
            el("thead", {}, el("tr", {},
              el("th", {}, "king"), el("th", {}, "model"),
              BENCHMARK_ORDER.map(suite => el("th", { class: "r" }, benchmarkLabel(suite))))),
            el("tbody", {}, rows)))
      : el("div", { class: "bench-history-empty" }, "no benchmark history yet"));
}

export function renderBenchmarks(container, metaNode, data) {
  const activeProgress = activeProgressByModelSuite(data);
  const models = (data?.models || []).filter(model => completedRuns(model).length || hasActiveProgress(model, activeProgress));
  if (!models.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark data yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const sorted = sortModels(models).filter(hasPanelScores);
  if (!sorted.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark scores yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const selected = sorted.find(model => !isGenesis(model)) || sorted[0];
  const rerender = () => renderBenchmarks(container, metaNode, data);
  const scores = suiteScores(selected);
  const done = BENCHMARK_ORDER.filter(suite => scores[suite]?.score != null).length;

  mount(container,
    el("section", { class: "bench-panel" },
      el("div", { class: "bench-panel-head" },
        el("span", {}, "benchmark panel"),
        el("div", { class: "bench-panel-tools" },
          el("button", { class: "bench-history-toggle", type: "button", onClick: () => {
            historyOpen = !historyOpen;
            localStorage.setItem("benchPanelHistoryOpen", historyOpen ? "1" : "0");
            rerender();
          } }, historyOpen ? "hide history" : "history"),
          el("span", { class: "bench-panel-meta" },
            `${done}/${BENCHMARK_ORDER.length} scores · ${modelLabel(selected)}`))),
      el("div", { class: "bench-tile-grid" }, BENCHMARK_ORDER.map(suite =>
        renderTile(selected, suite, activeProgress.get(progressKey(selected.model_repo || selected.id, suite))))),
      historyOpen ? renderHistoryPanel(sorted, selected, rerender) : null));
  if (metaNode) metaNode.textContent = `${models.length} models · ${data.counts?.runs ?? 0} benchmark runs · updated ${fmtRelative(data.generated_at)}`;
}
