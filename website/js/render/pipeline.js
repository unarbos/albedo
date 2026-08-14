import { el, mount, link } from "../dom.js";
import { fmtRelative, fmtDateTime } from "../format.js";
import { hubRepoUrl, modelRepo, modelName, taoMinerUrl } from "../model.js";

const STAGES = [
  { key: "eval", label: "eval" },
  { key: "pre_eval", label: "pre-eval" },
  { key: "hippius_validate", label: "validation" },
];

const STAGE_RANK = Object.fromEntries(STAGES.map((s, i) => [s.key, i]));
const STATUS_RANK = { working: 0, queued: 1 };
const updatedMs = row => new Date(row.item.updated_at || 0).getTime();

const byQueueOrder = (a, b) => {
  const stage = STAGE_RANK[a.stageKey] - STAGE_RANK[b.stageKey];
  if (stage) return stage;
  const status = STATUS_RANK[a.status] - STATUS_RANK[b.status];
  if (status) return status;
  return updatedMs(a) - updatedMs(b);
};

function collectRows(stages) {
  return STAGES.flatMap(stage => {
    const data = stages[stage.key] || { running: [], queued: [] };
    return [
      ...(data.running || []).map(item => ({ stageKey: stage.key, stage: stage.label, status: "working", item })),
      ...(data.queued || []).map(item => ({ stageKey: stage.key, stage: stage.label, status: "queued", item })),
    ];
  }).sort(byQueueOrder);
}

function queueRow(row, netuid) {
  const item = row.item;
  const name = modelName(item);
  const repo = modelRepo(item.model_uri);
  const repoUrl = hubRepoUrl(item.model_uri);
  const tao = taoMinerUrl(netuid, item.hotkey);
  let status = row.status;
  let statusTitle = null;
  if (row.stageKey === "eval" && row.status === "working") {
    const pass = (item.prior_wins ?? 0) >= 1 ? 2 : 1;
    status = `working ${pass}/2`;
    statusTitle = pass === 2
      ? "confirmation eval: won pass 1, must win again on fresh samples to be crowned"
      : "pass 1 of 2: a win here queues a confirmation eval on fresh samples";
  }
  return el("tr", { class: row.status === "working" ? "q-row working" : "q-row" },
    el("td", {}, el("span", { class: "q-stage-badge" }, row.stage)),
    el("td", {}, el("span", { class: row.status === "working" ? "q-status working" : "q-status", title: statusTitle }, status)),
    el("td", { class: "uid" }, tao ? link(tao, String(item.uid ?? "-")) : String(item.uid ?? "-")),
    el("td", { class: "model" }, repoUrl ? link(repoUrl, name, { class: "model-cell", title: repo }) : el("span", { class: "model-cell", title: repo }, name)),
    el("td", { class: "r when", title: fmtDateTime(item.updated_at) }, fmtRelative(item.updated_at)));
}

function idleRow() {
  return el("tr", { class: "q-row" },
    el("td", {}, el("span", { class: "q-stage-badge" }, "-")),
    el("td", {}, el("span", { class: "q-status" }, "idle")),
    el("td", { class: "uid" }, "-"),
    el("td", { class: "model" }, "-"),
    el("td", { class: "r when" }, "-"));
}

export function renderPipeline(container, state, netuid) {
  const rows = collectRows(state?.stages || {});
  mount(container,
    el("section", { class: "queue-panel" },
      el("div", { class: "q-table-wrap" },
        el("table", { class: "data-table q-table" },
          el("thead", {}, el("tr", {},
            el("th", {}, "stage"),
            el("th", {}, "state"),
            el("th", {}, "uid"),
            el("th", {}, "model"),
            el("th", { class: "r" }, "updated"))),
          el("tbody", {}, rows.length ? rows.map(row => queueRow(row, netuid)) : idleRow())))));
}
