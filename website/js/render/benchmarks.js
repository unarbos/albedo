import { el, mount } from "../dom.js";
import { fmt, fmtRelative } from "../format.js";
import { modelRepo } from "../model.js";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  tau2_banking_knowledge: "Tau2 Banking",
  swe_rebench_2026_03: "SWE-rebench",
};

const BENCHMARK_ORDER = ["tau2_airline", "tau2_retail", "tau2_telecom", "tau2_banking_knowledge", "swe_rebench_2026_03"];

const PAGE_SIZES = [5, 10, 25, 50];
const ACTIVE_STATES = new Set(["QUEUED", "CLAIMED", "LOADING_MODEL", "RUNNING", "SCORING"]);

let pageSize = 10;
let page = 1;

function benchmarkLabel(suite) {
  return BENCHMARK_LABELS[suite] || suite || "—";
}

function modelName(model) {
  return model?.model_repo || modelRepo(model?.model_uri) || model?.model_uri || "—";
}

function modelLabel(model) {
  return model?.label || "—";
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

function latestFeaturedModel(models) {
  return sortModels(models).find(model => completedRuns(model).length) || sortModels(models)[0] || null;
}

function highestKingModel(models) {
  return sortModels(models).find(model => !isGenesis(model)) || null;
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
  const identity = `${model?.label || ""} ${model?.model_repo || ""}`.toLowerCase();
  return identity.includes("genesis") || identity.includes("qwen/qwen3.6-35b-a3b");
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
      el("span", { class: "meta" }, item?.state || "queued"));
  });
  return el("div", { class: "bench-feature" },
    el("div", { class: "bench-feature-main" },
      el("div", { class: "bench-feature-eyebrow" }, "benchmark panel"),
      el("div", { class: "bench-feature-title" }, `${modelLabel(model)} · ${modelName(model)}`),
      el("div", { class: "bench-feature-meta" },
        el("span", {}, model.model_repo || "—"),
        el("span", {}, run ? benchmarkLabel(run.suite) : "pending"),
        el("span", {}, latestRunTime(model) ? fmtRelative(latestRunTime(model)) : "pending")),
      el("a", { class: "bench-feature-details", href: detailHref(model, run?.id || null) }, "details")),
    el("div", { class: "bench-summary-grid" }, scoreItems));
}

function tableScoreCell(model, run, baseline, boxed = false, progress = null) {
  const cls = boxed ? "bench-table-score" : "bench-table-score plain";
  if (!run) {
    if (progress) {
      return el("span", { class: `${cls} pending live` },
        el("span", { class: "bench-card-name" }, activeState(progress).toLowerCase().replaceAll("_", " ") || "active"),
        el("span", { class: "bench-score pending" }, progressLabel(progress)),
        el("span", { class: "bench-card-meta" }, progressMeta(progress)));
    }
    return el("span", { class: `${cls} pending` },
      el("span", { class: "bench-card-name" }, "pending"),
      el("span", { class: "bench-score pending" }, "—"));
  }
  return el("a", { class: `${cls} ${runStateClass(run)}`, href: detailHref(model, run.id) },
    el("span", { class: "bench-card-name" }, run.state || "done"),
    scoreCell(run, scoreDelta(model, run, baseline)),
    el("span", { class: "bench-card-meta" }, run.finished_at ? fmtRelative(run.finished_at) : "running"));
}

function renderTableRows(models, baseline, highlightedModel, activeProgress) {
  return models.map(model => {
    const runs = new Map(latestRunsByBenchmark(model).map(run => [run.suite, run]));
    const boxed = highlightedModel?.id === model.id;
    const repoUrl = hfRepoUrl(model);
    return el("tr", { class: boxed ? "bench-feature-row" : "" },
      el("td", { class: "bench-label-col" }, repoUrl
        ? el("a", { class: "bench-label-link", href: repoUrl, target: "_blank", rel: "noopener" }, modelLabel(model))
        : modelLabel(model)),
      BENCHMARK_ORDER.map(suite => el("td", { class: "bench-score-col" },
        tableScoreCell(model, runs.get(suite), baseline, boxed, activeProgress.get(progressKey(model.model_repo || model.id, suite))))));
  });
}

export function renderBenchmarks(container, metaNode, data) {
  const activeProgress = activeProgressByModelSuite(data);
  const models = (data?.models || []).filter(model => completedRuns(model).length || hasActiveProgress(model, activeProgress));
  if (!models.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark data yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const baseline = genesisScores(models);
  const sortedModels = sortModels(models);
  const highlightedModel = highestKingModel(sortedModels);
  const pages = Math.max(1, Math.ceil(sortedModels.length / pageSize));
  page = Math.min(page, pages);
  const start = (page - 1) * pageSize;
  const shown = sortedModels.slice(start, start + pageSize);

  const rerender = () => renderBenchmarks(container, metaNode, data);
  const pageControls = el("div", { class: "bench-controls" },
    el("label", {}, "rows", el("select", { onChange: e => { pageSize = Number(e.target.value); page = 1; rerender(); } },
      PAGE_SIZES.map(size => el("option", { value: size, selected: size === pageSize }, String(size))))),
    el("button", { type: "button", disabled: page <= 1, onClick: () => { page -= 1; rerender(); } }, "prev"),
    el("span", { class: "meta" }, `${page}/${pages}`),
    el("button", { type: "button", disabled: page >= pages, onClick: () => { page += 1; rerender(); } }, "next"));

  mount(container,
    el("div", { class: "bench-table-head" },
      el("div", { class: "label" }, "benchmark panel history"),
      pageControls),
    el("div", { class: "data-table-wrap bench-table-wrap" },
      el("table", { class: "data-table" },
        el("thead", {}, el("tr", {},
          el("th", { class: "bench-label-col" }, "label"),
          BENCHMARK_ORDER.map(suite => el("th", { class: "bench-score-col" }, benchmarkLabel(suite))))),
        el("tbody", {}, shown.length ? renderTableRows(shown, baseline, highlightedModel, activeProgress) : el("tr", {}, el("td", { colspan: String(BENCHMARK_ORDER.length + 1) }, "no models yet."))))));
  if (metaNode) metaNode.textContent = `${models.length} models · ${data.counts?.runs ?? 0} benchmark runs · updated ${fmtRelative(data.generated_at)}`;
}
