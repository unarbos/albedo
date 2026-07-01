import { fetchBenchmarks } from "../fetch.js";
import { el, mount } from "../dom.js";
import { fmt, fmtDateTime, shortDigest } from "../format.js";
import { hubRepoUrl, kingTitleName, modelRepo } from "../model.js";

const BENCHMARK_LABELS = {
  terminal_bench_2: "Terminal-Bench 2.0",
  swe_bench_pro_100: "SWE-bench Pro",
  tau2_airline: "Tau2 Airline",
};

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const modelId = params.get("model_id");
const runId = params.get("run_id");

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

function latestRun(model) {
  return completedRuns(model).sort((a, b) => {
    const at = new Date(a.finished_at || a.started_at || "").getTime();
    const bt = new Date(b.finished_at || b.started_at || "").getTime();
    if (Number.isFinite(bt - at) && bt !== at) return bt - at;
    return Number(b.run_attempt || 0) - Number(a.run_attempt || 0);
  })[0] || null;
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
  for (const run of completedRuns(genesis)) {
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

function score(run, delta = null) {
  return el("span", { class: "bench-score-wrap" },
    el("span", { class: `bench-score ${runStateClass(run)}` }, run?.score == null ? "—" : fmt(run.score, 3)),
    deltaCell(delta));
}

function taskSummary(run) {
  const passed = run?.passed_count ?? "—";
  const total = run?.task_count ?? "—";
  return `${passed}/${total}`;
}

function detailHref(model, run) {
  const qs = new URLSearchParams();
  qs.set("model_id", model.id);
  if (run?.id) qs.set("run_id", run.id);
  return `./benchmark.html?${qs.toString()}`;
}

function kv(k, v, cls = "") {
  return el("div", { class: "kv" }, el("span", { class: "k" }, k), el("span", { class: cls ? `v ${cls}` : "v" }, v));
}

function renderTaskTable(run) {
  const rows = (run?.task_results || []).map(task => {
    const info = task.metrics?.exception_info;
    return el("tr", {},
      el("td", { class: "model" }, el("span", { class: "model-cell", title: task.task_name }, task.task_name || "—")),
      el("td", {}, el("span", { class: `bench-state ${String(task.state || "").toLowerCase()}` }, task.state || "—")),
      el("td", { class: "r" }, task.score == null ? "—" : fmt(task.score, 3)),
      el("td", { class: "when" }, fmtDateTime(task.finished_at)),
      el("td", { class: "fail-reason-cell" }, info ? `${info.exception_type || "error"}: ${info.exception_message || ""}` : ""));
  });
  return el("div", { class: "data-table-wrap bench-task-wrap" },
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "task"), el("th", {}, "state"), el("th", { class: "r" }, "score"), el("th", {}, "finished"), el("th", {}, "error"))),
      el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: "5" }, "no task rows.")))));
}

function render(model, selected, baseline) {
  const repo = hubRepoUrl(model.model_uri);
  mount($("b-title"), repo ? el("a", { href: repo, target: "_blank", rel: "noopener" }, `${benchmarkKingTitle(model)} · ${modelName(model)}`) : `${benchmarkKingTitle(model)} · ${modelName(model)}`);
  $("b-sub").textContent = `${model.king_version_id} · ${shortDigest(model.model_hash)}`;

  const runs = completedRuns(model);
  const tabs = runs.length ? el("div", { class: "bench-run-tabs detail-section" }, runs.map(run =>
    el("a", { href: detailHref(model, run), class: run.id === selected?.id ? "active" : "" }, benchmarkLabel(run.suite)))) : null;

  mount($("b-body"),
    tabs,
    selected ? el("div", { class: "kv-grid" },
      kv("benchmark", benchmarkLabel(selected.suite)),
      kv("state", selected.state || "—", runStateClass(selected)),
      kv("score", score(selected, scoreDelta(model, selected, baseline))),
      kv("tasks", taskSummary(selected)),
      kv("worker", selected.worker_id || "—"),
      kv("finished", fmtDateTime(selected.finished_at))) : el("div", { class: "empty" }, "no completed benchmark runs yet."),
    selected ? el("div", { class: "detail-section" }, el("h2", {}, "task results"), renderTaskTable(selected)) : false,
    el("div", { class: "detail-section" }, el("h2", {}, "model"),
      el("div", { class: "kv-grid" },
        kv("artifact", model.artifact_uri || "—"),
        kv("activated", fmtDateTime(model.activated_at)))))
}

async function load() {
  const data = await fetchBenchmarks();
  if (!data) {
    mount($("b-body"), el("div", { class: "empty" }, "could not load benchmark data."));
    return;
  }
  const models = data.models || [];
  const model = models.find(m => m.id === modelId)
    || models.find(m => completedRuns(m).some(r => r.id === runId));
  if (!model) {
    mount($("b-body"), el("div", { class: "empty" }, "benchmark model not found."));
    return;
  }
  const selected = completedRuns(model).find(run => run.id === runId) || latestRun(model);
  render(model, selected, genesisScores(models));
}

load();
