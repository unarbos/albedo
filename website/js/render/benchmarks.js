import { el, mount } from "../dom.js";
import { pct, fmtDateTime, fmtRelative, fmtDuration } from "../format.js";
import { modelRepo, kingTitleName } from "../model.js";
import { PREDS_TOTAL_FALLBACK, PREDS_STALE_MS } from "../config.js";

const MODEL_SCORE_SUITE = "model_score";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  swe_rebench_2026_03: "SWE-rebench",
  model_score: "SWE-bench Verified"
};

// const BENCHMARK_ORDER = ["tau2_airline", "tau2_retail", "tau2_telecom", "swe_rebench_2026_03", MODEL_SCORE_SUITE];
const BENCHMARK_ORDER = ["swe_rebench_2026_03", MODEL_SCORE_SUITE];

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

// benchmarks.json labels ("King <N>") map 1:1 to chain reigns — each king-<N> repo's
// albedo.md names the hippius repo/hotkey of chain king N. Display the reign name
// (ALBEDO-<roman>) used everywhere else on the site.
function modelLabel(model) {
  const label = model?.label || "—";
  if (/^genesis$/i.test(label)) return kingTitleName(0);
  const match = /^King\s+([IVXLCDM]+)$/i.exec(label);
  if (!match) return label;
  return kingTitleName(romanToInt(match[1]));
}

function modelKingNumber(model) {
  const labelMatch = /^King\s+([IVXLCDM]+)$/i.exec(model?.label || "");
  const repoMatch = /-king-([IVXLCDM]+)$/i.exec(model?.model_repo || "");
  const numeral = labelMatch?.[1] || repoMatch?.[1];
  return numeral ? romanToInt(numeral) : null;
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

export function sortModels(models) {
  return [...(models || [])].sort((a, b) => {
    if (isGenesis(a) !== isGenesis(b)) return isGenesis(a) ? 1 : -1;
    const aKing = modelKingNumber(a);
    const bKing = modelKingNumber(b);
    if (aKing != null || bKing != null) {
      if (aKing == null) return 1;
      if (bKing == null) return -1;
      if (aKing !== bKing) return bKing - aKing;
    }
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

export function mergeModelScores(data, rows) {
  if (!Array.isArray(rows)) return data;
  const scores = new Map(rows.filter(row => row?.run_id).map(row => [String(row.run_id).toLowerCase(), row]));
  return {
    ...data,
    models: (data?.models || []).map(model => {
      const number = modelKingNumber(model);
      const key = isGenesis(model) ? "king-genesis" : number == null ? null : `king-${modelLabel(model).split("-").pop()}`.toLowerCase();
      const row = scores.get(key);
      const score = Number(row?.score);
      if (!Number.isFinite(score)) return model;
      return {
        ...model,
        runs: [
          ...(model.runs || []).filter(run => run.suite !== MODEL_SCORE_SUITE),
          {
            id: `model-score:${row.run_id}`,
            suite: MODEL_SCORE_SUITE,
            score: score / 100,
            state: "SUCCEEDED",
            task_count: row.total,
            passed_count: row.resolved,
            score_meta: `${row.resolved ?? "—"}/${row.total ?? "—"} resolved`,
            no_detail: true,
          },
        ],
      };
    }),
  };
}

export function modelScoreRunId(model) {
  if (!model) return null;
  if (isGenesis(model)) return "king-genesis";
  const numeral = modelLabel(model).split("-").pop();
  return /^[IVXLCDM]+$/.test(numeral) ? `king-${numeral}` : null;
}

function panelModels(data) {
  const activeProgress = activeProgressByModelSuite(data);
  const models = (data?.models || []).filter(model => completedRuns(model).length || hasActiveProgress(model, activeProgress));
  const sorted = sortModels(models).filter(hasPanelScores);
  return { models, sorted, selected: sorted.find(model => !isGenesis(model)) || sorted[0] || null };
}

export function liveScoreRunId(data, modelScores) {
  const { selected } = panelModels(mergeModelScores(data, modelScores));
  if (!selected || suiteScores(selected)[MODEL_SCORE_SUITE]?.score != null) return null;
  return modelScoreRunId(selected);
}

function scoreTotal(modelScores) {
  const totals = (modelScores || []).map(row => Number(row?.total)).filter(n => Number.isFinite(n) && n > 0);
  return totals.length ? Math.max(...totals) : PREDS_TOTAL_FALLBACK;
}

function livePreds(live, modelScores) {
  if (!live?.count) return null;
  const total = scoreTotal(modelScores);
  const left = Math.max(0, total - live.count);
  const updated = live.updatedAt ? new Date(live.updatedAt).getTime() : NaN;
  const ratio = Math.min(1, live.count / total);
  const fresh = Number.isFinite(updated) ? Date.now() - updated < PREDS_STALE_MS : true;
  return {
    count: live.count,
    total,
    ratio,
    fresh,
    scoring: !fresh && ratio >= 0.95,
    eta: live.rate && left ? fmtDuration(left / live.rate) : null,
    updatedAt: live.updatedAt,
  };
}

function detailHref(model, runId = null) {
  const qs = new URLSearchParams();
  if (model?.id) qs.set("model_id", model.id);
  if (runId) qs.set("run_id", runId);
  return `./benchmark.html?${qs.toString()}`;
}

function runTime(run) {
  return new Date(run?.finished_at || run?.started_at || "").getTime() || 0;
}

export function suiteScores(model) {
  const scores = { ...(model?.latest_scores || {}) };
  const passes = {};
  for (const run of model?.runs || []) {
    if (!run?.suite || run.score == null) continue;
    (passes[run.suite] ||= []).push(run);
  }
  for (const [suite, runs] of Object.entries(passes)) {
    const latest = runs.reduce((a, b) => runTime(b) > runTime(a) ? b : a);
    scores[suite] = {
      ...latest,
      score: runs.reduce((sum, run) => sum + Number(run.score), 0) / runs.length,
      pass_count: runs.length,
      run_id: latest.run_id || latest.id,
    };
  }
  return scores;
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

function baselineComparison(entry, baseline) {
  if (baseline?.score == null) return { label: "genesis —", delta: "—", cls: "" };
  if (entry?.score == null) return { label: `genesis ${panelScore(baseline.score)}`, delta: "—", cls: "" };
  const delta = (Number(entry.score) - Number(baseline.score)) * 100;
  return {
    label: `genesis ${panelScore(baseline.score)}`,
    delta: `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`,
    cls: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
  };
}

function previousComparison(entry, sorted, selected, suite) {
  if (entry?.score == null) return { delta: "—", cls: "flat" };
  const previous = sorted
    .slice(sorted.indexOf(selected) + 1)
    .find(model => !isGenesis(model) && suiteScores(model)[suite]?.score != null);
  if (!previous) return { delta: "—", cls: "flat" };
  const delta = (Number(entry.score) - Number(suiteScores(previous)[suite].score)) * 100;
  return {
    delta: `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`,
    cls: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
  };
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

function renderSpark(sorted, suite, width = 360) {
  const points = [...sorted].reverse()
    .map(model => ({ label: modelLabel(model), score: suiteScores(model)[suite]?.score }))
    .filter(point => point.score != null);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} 44`, preserveAspectRatio: "xMidYMid", role: "img" });

  // line below graph
  svg.append(svgEl("line", { x1: 6, y1: 36, x2: width - 6, y2: 36, stroke: "currentColor", "stroke-width": 1, opacity: 0.15 }));

  if (!points.length) {
    svg.append(svgEl("text", { x: width / 2, y: 27, "text-anchor": "middle", "font-size": 8, fill: "currentColor", opacity: 0.45 }, "no score"));
    return svg;
  }
  let min = Math.min(...points.map(point => point.score));
  let max = Math.max(...points.map(point => point.score));
  if (min === max) { min -= 0.005; max += 0.005; }
  const coords = points.map((point, i) => ({
    x: points.length === 1 ? width / 2 : 6 + (i / (points.length - 1)) * (width - 12),
    y: 36 - ((point.score - min) / (max - min)) * 28,
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
      cx: c.x.toFixed(1), cy: c.y.toFixed(1), r: last ? 3 : 2.4,
      fill: "currentColor", opacity: last ? 1 : 0.45,
      class: last ? "spark-dot-last" : null,
    }, svgEl("title", {}, `${c.point.label} · ${panelScore(c.point.score)}`)));
  });
  return svg;
}

function progressLabel(preds) {
  if (preds.fresh) return "running";
  return preds.scoring ? "scoring" : "stalled";
}

function renderProgress(preds, label) {
  const percent = (preds.ratio * 100).toFixed(1);
  const state = preds.fresh
    ? [`${percent}%`, preds.eta ? `eta ${preds.eta}` : "generating"]
    : preds.scoring
      ? [`${percent}%`, "awaiting score"]
      : [`${percent}%`, `idle ${fmtRelative(preds.updatedAt)}`];
  return el("div", { class: "bench-tile-progress" },
    el("div", { class: preds.fresh ? "bench-tile-progress-bar live" : "bench-tile-progress-bar" },
      el("i", { style: `width:${percent}%` })),
    el("div", { class: "bench-tile-progress-note" }, [label, ...state].filter(Boolean).join(" · ")));
}

function renderTile(model, suite, sorted, baseline, activity, preds) {
  const entry = suiteScores(model)[suite];
  const scored = entry?.score != null;
  const progress = scored ? null : preds;
  const genesis = baselineComparison(entry, baseline);
  const previous = previousComparison(entry, sorted, model, suite);
  const running = activity?.running;
  const queued = activity?.queued || [];
  const href = entry?.run_id && !entry.no_detail ? detailHref(model, entry.run_id) : null;
  const runNote = running
    ? [runningLabel(running, activity.labelByRepo), progressNote(running)].filter(Boolean).join(" · ")
    : queued.length ? `${queued.length} pending` : "";
  const live = Boolean(running) || Boolean(progress?.fresh);

  const chartSvgElement = el("div", { class: "bench-tile-chart" }, renderSpark(sorted, suite));
  let chartWidth = 0;
  const chartObserver = new ResizeObserver(entries => {
    const w = Math.round(entries[0].contentRect.width);
    if (!w || w === chartWidth) return;
    chartWidth = w;
    chartSvgElement.replaceChildren(renderSpark(sorted, suite, w));
  });
  chartObserver.observe(chartSvgElement);

  return el("article", {
    class: "bench-tile",
    "data-status": scored ? "completed" : progress ? "progress" : "missing",
    "data-activity": live ? "running" : "idle",
  },
    el("div", { class: "bench-tile-head" },
      el("div", { class: "bench-tile-name" }, benchmarkLabel(suite)),
      el("span", { class: live ? "bench-tile-activity live" : "bench-tile-activity" }, live ? "running" : "idle")),
    el("div", { class: "bench-tile-main" },
      el("div", { class: "bench-tile-score-wrap" },
        el(href ? "a" : "span", { class: "bench-tile-score", href },
          scored ? panelScore(entry.score) : progress ? progressLabel(progress) : "missing"),
        scored
          ? el("span", { class: "bench-tile-pass-count" },
              entry.score_meta || `avg · ${entry.pass_count || 1} ${entry.pass_count === 1 ? "pass" : "passes"}`)
          : progress
            ? el("span", { class: "bench-tile-pass-count" }, `${progress.count} / ${progress.total} predictions`)
            : null),
      el("div", { class: `bench-tile-change ${previous.cls}` },
        el("strong", {}, previous.delta),
        el("span", {}, "since last"))),
    chartSvgElement,
    el("div", { class: "bench-tile-status" },
      el("span", {}, genesis.label),
      el("span", { class: `bench-delta ${genesis.cls}`, title: "delta vs genesis" }, genesis.delta)),
    progress ? renderProgress(progress, modelLabel(model)) : runNote ? el("div", { class: "bench-tile-run-note" }, runNote) : null);
}

function runningLabel(item, labelByRepo) {
  if (labelByRepo?.has(item?.model_repo)) return labelByRepo.get(item.model_repo);
  if (item?.label) return modelLabel({ label: item.label });
  return (item?.model_repo || "").split("/").pop() || "—";
}

function progressNote(item) {
  const done = Number(item?.progress_done);
  const total = Number(item?.progress_total);
  if (Number.isFinite(done) && Number.isFinite(total) && total > 0) return `${done}/${total}`;
  return null;
}

function suiteActivity(data) {
  const models = data?.models || [];
  const labelByRepo = new Map(models.filter(m => m.model_repo).map(m => [m.model_repo, modelLabel(m)]));
  const orderByRepo = new Map(models.filter(m => m.model_repo).map(m => [m.model_repo, Number(m.model_order ?? 999999)]));

  const runningBySuite = new Map();
  for (const worker of data?.workers || []) {
    if (worker?.suite && worker?.model_repo && isActiveProgress(worker)) runningBySuite.set(worker.suite, worker);
  }
  const queuedBySuite = new Map(BENCHMARK_ORDER.map(suite => [suite, []]));
  for (const job of data?.jobs || []) {
    if (!queuedBySuite.has(job?.suite) || !isActiveProgress(job)) continue;
    if (activeState(job) === "QUEUED") queuedBySuite.get(job.suite).push(job);
    else if (!runningBySuite.has(job.suite)) runningBySuite.set(job.suite, job);
  }
  return new Map(BENCHMARK_ORDER.map(suite => {
    const queued = [...(queuedBySuite.get(suite) || [])].sort((a, b) =>
      (orderByRepo.get(a.model_repo) ?? 999999) - (orderByRepo.get(b.model_repo) ?? 999999));
    return [suite, { running: runningBySuite.get(suite), queued, labelByRepo }];
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
        return el("td", { class: "r", title: entry.score_meta || `${entry.pass_count || 1} pass average` }, panelScore(entry.score));
      }));
  });

  return el("div", { class: "bench-history" },
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

export function renderBenchmarks(container, metaNode, data, modelScores = null, live = null) {
  data = mergeModelScores(data, modelScores);
  const { models, sorted, selected } = panelModels(data);
  if (!models.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark data yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  if (!sorted.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark scores yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const baselineScores = suiteScores((data?.models || []).find(isGenesis));
  const activity = suiteActivity(data);
  const preds = live?.runId === modelScoreRunId(selected) ? livePreds(live, modelScores) : null;
  const rerender = () => renderBenchmarks(container, metaNode, data, modelScores, live);
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
        renderTile(selected, suite, sorted, baselineScores[suite], activity.get(suite),
          suite === MODEL_SCORE_SUITE ? preds : null))),
      historyOpen ? renderHistoryPanel(sorted, selected, rerender) : null));
  if (metaNode) metaNode.textContent = `${models.length} models · ${data.counts?.runs ?? 0} benchmark runs · updated ${fmtRelative(data.generated_at)}`;
}
