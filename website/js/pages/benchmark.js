import { fetchBenchmarks, fetchJson } from "../fetch.js";
import { el, mount } from "../dom.js";
import { fmt, fmtDateTime, shortDigest } from "../format.js";
import { modelRepo } from "../model.js";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  tau2_banking_knowledge: "Tau2 Banking",
};
const TAU2_BENCH_VERSION = "τ²-bench 1.0.0";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const modelId = params.get("model_id");
const runId = params.get("run_id");

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
  const identity = `${model?.label || ""} ${model?.model_repo || ""}`.toLowerCase();
  return identity.includes("genesis") || identity.includes("qwen/qwen3.6-35b-a3b");
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

function cleanUserSimulator(run) {
  const llm = run?.environment?.user_llm || run?.harness_config?.user_llm || "gpt-5.2";
  return String(llm).includes("gpt-5.2") ? "gpt-5.2" : String(llm).replace(/^openrouter\/openai\//, "");
}

function benchVersion(run) {
  const version = run?.environment?.benchmark_version || run?.harness_config?.bench_version;
  if (version) return version;
  const ref = run?.environment?.benchmark_repo_ref || run?.harness_config?.repo_ref;
  if (ref) return `${TAU2_BENCH_VERSION} @ ${shortDigest(ref)}`;
  return String(run?.suite || "").startsWith("tau2_") ? TAU2_BENCH_VERSION : "—";
}

function methodologyNotes(model, run) {
  const cfg = run?.harness_config || {};
  const env = run?.environment || {};
  const parts = [
    `Evaluated using ${modelName(model)} with reasoning_effort: none.`,
    `User simulator: ${cleanUserSimulator(run)} with reasoning_effort: ${cfg.user_reasoning_effort || "low"}.`,
    `${cfg.num_trials || 1} trials.`,
    `Seed: ${cfg.seed || 300}.`,
    `Domain: ${env.domain || suiteDomain(run?.suite)}.`,
  ];
  if ((env.domain || run?.suite || "").includes("banking")) {
    parts.push("Banking domain evaluated with retrieval_config: qwen_embeddings.");
  }
  return parts.join(" ");
}

function suiteDomain(suite) {
  return String(suite || "").replace(/^tau2_/, "") || "—";
}

function renderMethodology(model, run) {
  return el("div", { class: "detail-section" },
    el("h2", {}, "methodology"),
    el("div", { class: "kv-grid" },
      kv("User Simulator", cleanUserSimulator(run)),
      kv("Evaluation Date", fmtDateTime(run?.finished_at || run?.started_at)),
      kv("Bench Version", benchVersion(run)),
      kv("Notes", methodologyNotes(model, run))));
}

function taskArtifactTasks(run) {
  return (run?.task_results || []).filter(task => task.artifact_uri);
}

function renderTrajectoryViewer(run) {
  const tasks = taskArtifactTasks(run);
  if (!tasks.length) {
    return el("div", { class: "detail-section" }, el("h2", {}, "trajectory"), el("div", { class: "empty" }, "no trajectory artifacts yet."));
  }
  return el("div", { class: "detail-section" },
    el("h2", {}, "trajectory"),
    el("div", { class: "trajectory-shell" },
      el("div", { class: "trajectory-toolbar" },
        el("select", { id: "trajectory-task-select" }, tasks.map((task, index) =>
          el("option", { value: String(index) }, `${task.task_name || "task"} ${task.trial_name || ""}`.trim()))),
        el("a", { id: "trajectory-open", href: tasks[0].artifact_uri, target: "_blank", rel: "noopener" }, "open json")),
      el("div", { class: "trajectory-layout" },
        el("div", { id: "trajectory-messages", class: "trajectory-messages" }, el("div", { class: "empty" }, "loading trajectory…")),
        el("div", { id: "trajectory-meta", class: "trajectory-meta" }))));
}

function renderTrajectoryMessages(payload) {
  const messages = Array.isArray(payload?.messages) ? payload.messages : [];
  if (!messages.length) {
    const error = payload?.error || payload?.info?.error || "no messages recorded.";
    return el("div", { class: "trajectory-error" }, String(error));
  }
  return messages.map((message, index) => {
    const role = message?.role || message?.sender || message?.source || `step ${index + 1}`;
    const content = message?.content ?? message?.message ?? message?.text ?? JSON.stringify(message);
    return el("div", { class: "trajectory-message" },
      el("div", { class: "trajectory-role" }, String(role)),
      el("pre", {}, typeof content === "string" ? content : JSON.stringify(content, null, 2)));
  });
}

function renderTrajectoryMeta(payload, task) {
  const links = [];
  if (task?.artifact_uri) links.push(el("a", { href: task.artifact_uri, target: "_blank", rel: "noopener" }, "trajectory json"));
  const reward = payload?.reward_breakdown || payload?.reward_info?.reward_breakdown;
  return el("div", { class: "kv-grid trajectory-kv" },
    kv("task", payload?.task_id || task?.task_name || "—"),
    kv("score", payload?.score == null ? "—" : fmt(payload.score, 3)),
    kv("state", payload?.state || task?.state || "—"),
    kv("termination", payload?.termination_reason || task?.metrics?.termination_reason || "—"),
    kv("duration", payload?.duration == null ? "—" : `${fmt(payload.duration, 2)}s`),
    kv("agent cost", payload?.agent_cost == null ? "—" : fmt(payload.agent_cost, 4)),
    kv("user cost", payload?.user_cost == null ? "—" : fmt(payload.user_cost, 4)),
    kv("reward", reward ? JSON.stringify(reward) : "—"),
    el("div", { class: "kv" }, el("span", { class: "k" }, "artifacts"), el("span", { class: "v" }, links.length ? links : "—")));
}

function wireTrajectory(run) {
  const tasks = taskArtifactTasks(run);
  const select = $("trajectory-task-select");
  const open = $("trajectory-open");
  const messages = $("trajectory-messages");
  const meta = $("trajectory-meta");
  if (!tasks.length || !select || !open || !messages || !meta) return;

  async function loadTask() {
    const task = tasks[Number(select.value) || 0];
    open.href = task.artifact_uri;
    mount(messages, el("div", { class: "empty" }, "loading trajectory…"));
    mount(meta);
    const payload = await fetchJson(task.artifact_uri);
    if (!payload) {
      mount(messages, el("div", { class: "trajectory-error" }, "could not load trajectory artifact."));
      return;
    }
    mount(messages, renderTrajectoryMessages(payload));
    mount(meta, renderTrajectoryMeta(payload, task));
  }

  select.addEventListener("change", loadTask);
  loadTask();
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
  const repo = hfRepoUrl(model);
  mount($("b-title"), repo ? el("a", { href: repo, target: "_blank", rel: "noopener" }, `${modelLabel(model)} · ${modelName(model)}`) : `${modelLabel(model)} · ${modelName(model)}`);
  $("b-sub").textContent = `${model.model_repo || model.id || "—"} · ${shortDigest(model.artifact_sha256 || model.model_hash)}`;

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
    selected ? renderMethodology(model, selected) : false,
    selected ? renderTrajectoryViewer(selected) : false,
    selected ? el("div", { class: "detail-section" }, el("h2", {}, "task results"), renderTaskTable(selected)) : false,
    el("div", { class: "detail-section" }, el("h2", {}, "model"),
      el("div", { class: "kv-grid" },
        kv("repo", model.model_repo || "—"),
        kv("label", modelLabel(model)),
        kv("artifact", model.artifact_uri || "—"),
        kv("model uri", model.model_uri || "—"),
        kv("activated", fmtDateTime(model.activated_at)))))
  if (selected) wireTrajectory(selected);
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
