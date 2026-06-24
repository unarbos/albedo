import { el, mount } from "../dom.js";
import { fmt, fmtRelative, shortDigest } from "../format.js";
import { kingTitleName, modelRepo } from "../model.js";

const BENCHMARK_LABELS = {
  terminal_bench_2: "Terminal-Bench 2.0",
  swe_bench_pro_100: "SWE-bench Pro",
  tau2_airline: "Tau2 Airline",
};

const BENCHMARK_ORDER = ["terminal_bench_2", "swe_bench_pro_100", "tau2_airline"];

const PAGE_SIZES = [5, 10, 25, 50];

let pageSize = 10;
let page = 1;

function benchmarkLabel(suite) {
  return BENCHMARK_LABELS[suite] || suite || "—";
}

function modelName(model) {
  return modelRepo(model?.model_uri) || model?.model_uri || "—";
}

function benchmarkKingTitle(model) {
  const n = Number(model?.king_version);
  return kingTitleName(Number.isFinite(n) ? n - 40 : null);
}

function completedRuns(model) {
  return (model?.runs || []).filter(run => run.score != null || Number(run.task_count || 0) > 0 || run.finished_at);
}

function benchmarkCount(model) {
  return new Set(completedRuns(model).map(run => run.suite).filter(Boolean)).size;
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
    const kingDelta = Number(b.king_version || 0) - Number(a.king_version || 0);
    if (kingDelta) return kingDelta;
    const timeDelta = new Date(latestRunTime(b)).getTime() - new Date(latestRunTime(a)).getTime();
    return Number.isFinite(timeDelta) ? timeDelta : 0;
  });
}

function latestFeaturedModel(models) {
  return sortModels(models).find(model => completedRuns(model).length) || sortModels(models)[0] || null;
}

function latestRunsByBenchmark(model) {
  const map = new Map();
  for (const run of completedRuns(model)) {
    const current = map.get(run.suite);
    const key = `${run.finished_at || run.started_at || ""}:${run.run_attempt || 0}`;
    const curKey = current ? `${current.finished_at || current.started_at || ""}:${current.run_attempt || 0}` : "";
    if (!current || key >= curKey) map.set(run.suite, run);
  }
  return [...map.values()].sort((a, b) => String(a.suite).localeCompare(String(b.suite)));
}

function runStateClass(run) {
  const state = String(run?.state || "").toLowerCase();
  if (state === "succeeded") return "ok";
  if (state.includes("fail") || state.includes("missing")) return "bad";
  return "live";
}

function isGenesis(model) {
  const uri = `${model?.model_uri || ""} ${model?.artifact_uri || ""}`.toLowerCase();
  return uri.includes("genesis") || String(model?.artifact_sha256 || "") === "efd5b8d0a1c1f472be56ff919419cdd0561bdecd9013d5c2a96dd0e23e89c165";
}

function genesisScores(models) {
  const genesis = (models || []).find(isGenesis);
  const out = new Map();
  if (!genesis) return out;
  for (const run of latestRunsByBenchmark(genesis)) {
    if (run.score != null) out.set(run.suite, Number(run.score));
  }
  return out;
}

function scoreDelta(model, run, baseline) {
  if (!run || run.score == null || isGenesis(model)) return null;
  const base = baseline?.get(run.suite);
  if (base == null) return null;
  return Number(run.score) - Number(base);
}

function deltaCell(delta) {
  if (delta == null || !Number.isFinite(delta)) return null;
  const cls = delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const sign = delta > 0 ? "+" : "";
  return el("span", { class: `bench-delta ${cls}`, title: "delta vs genesis" }, `${sign}${fmt(delta, 3)}`);
}

function scoreCell(run, delta = null) {
  if (!run) return el("span", { class: "muted-dash" }, "—");
  return el("span", { class: "bench-score-wrap" },
    el("span", { class: `bench-score ${runStateClass(run)}` }, fmt(run.score, 3)),
    deltaCell(delta));
}

function taskSummary(run) {
  if (!run) return "—";
  const passed = run.passed_count ?? "—";
  const total = run.task_count ?? "—";
  return `${passed}/${total}`;
}

function detailHref(model, runId = null) {
  const qs = new URLSearchParams();
  if (model?.id) qs.set("model_id", model.id);
  if (runId) qs.set("run_id", runId);
  return `./benchmark.html?${qs.toString()}`;
}

function renderFeatured(model, baseline) {
  if (!model) return el("div", { class: "empty" }, "no benchmark models yet.");
  const run = latestRun(model);
  const benchmarkRuns = new Map(latestRunsByBenchmark(model).map(item => [item.suite, item]));
  const scoreItems = BENCHMARK_ORDER.map(suite => {
    const item = benchmarkRuns.get(suite);
    return el("div", { class: item ? "bench-summary-score" : "bench-summary-score pending" },
      el("span", { class: "k" }, benchmarkLabel(suite)),
      item ? scoreCell(item, scoreDelta(model, item, baseline)) : el("span", { class: "bench-score pending" }, "pending"),
      el("span", { class: "meta" }, item ? taskSummary(item) : "—"));
  });
  return el("div", { class: "bench-feature" },
    el("div", { class: "bench-feature-main" },
      el("div", { class: "bench-feature-eyebrow" }, "latest benchmark"),
      el("div", { class: "bench-feature-title" }, `${benchmarkKingTitle(model)} · ${modelName(model)}`),
      el("div", { class: "bench-feature-meta" },
        el("span", {}, shortDigest(model.model_hash)),
        el("span", {}, run ? benchmarkLabel(run.suite) : "pending"),
        el("span", {}, latestRunTime(model) ? fmtRelative(latestRunTime(model)) : "pending")),
      el("a", { class: "bench-feature-details", href: detailHref(model, run?.id || null) }, "details")),
    el("div", { class: "bench-summary-grid" }, scoreItems));
}

function modelBestScores(model, baseline) {
  const runs = latestRunsByBenchmark(model);
  if (!runs.length) return el("span", { class: "muted-dash" }, "pending");
  return el("div", { class: "bench-suite-list" }, runs.map(run =>
    el("span", { class: `bench-suite-chip ${runStateClass(run)}` },
      el("span", { class: "suite" }, benchmarkLabel(run.suite)),
      el("b", {}, fmt(run.score, 3)),
      deltaCell(scoreDelta(model, run, baseline)))));
}

function renderTableRows(models, baseline) {
  return models.map(model => {
    const latest = latestRun(model);
    return el("tr", {},
      el("td", { class: "bench-king-col" }, benchmarkKingTitle(model)),
      el("td", { class: "model" }, el("span", { class: "model-cell", title: model.model_uri }, modelName(model))),
      el("td", {}, modelBestScores(model, baseline)),
      el("td", { class: "r" }, String(benchmarkCount(model))),
      el("td", { class: "when" }, latestRunTime(model) ? fmtRelative(latestRunTime(model)) : "—"),
      el("td", { class: "r" }, el("a", { class: "bench-details-btn", href: detailHref(model, latest?.id || null) }, "details")));
  });
}

export function renderBenchmarks(container, metaNode, data) {
  const models = data?.models || [];
  if (!models.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark data yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const featured = latestFeaturedModel(models);
  const baseline = genesisScores(models);
  const rest = sortModels(models).filter(model => model !== featured);
  const pages = Math.max(1, Math.ceil(rest.length / pageSize));
  page = Math.min(page, pages);
  const start = (page - 1) * pageSize;
  const shown = rest.slice(start, start + pageSize);

  const rerender = () => renderBenchmarks(container, metaNode, data);
  const pageControls = el("div", { class: "bench-controls" },
    el("label", {}, "rows", el("select", { onChange: e => { pageSize = Number(e.target.value); page = 1; rerender(); } },
      PAGE_SIZES.map(size => el("option", { value: size, selected: size === pageSize }, String(size))))),
    el("button", { type: "button", disabled: page <= 1, onClick: () => { page -= 1; rerender(); } }, "prev"),
    el("span", { class: "meta" }, `${page}/${pages}`),
    el("button", { type: "button", disabled: page >= pages, onClick: () => { page += 1; rerender(); } }, "next"));

  mount(container,
    renderFeatured(featured, baseline),
    el("div", { class: "bench-table-head" },
      el("div", { class: "label" }, "other benchmarked models"),
      pageControls),
    el("div", { class: "data-table-wrap bench-table-wrap" },
      el("table", { class: "data-table" },
        el("thead", {}, el("tr", {},
          el("th", { class: "bench-king-col" }, "king"), el("th", {}, "model"), el("th", {}, "benchmarks"),
          el("th", { class: "r" }, "benchmarks"), el("th", {}, "latest"), el("th", { class: "r" }, ""))),
        el("tbody", {}, shown.length ? renderTableRows(shown, baseline) : el("tr", {}, el("td", { colspan: "6" }, "no other models yet."))))));
  if (metaNode) metaNode.textContent = `${models.length} models · ${data.counts?.runs ?? 0} benchmark runs · updated ${fmtRelative(data.generated_at)}`;
}
