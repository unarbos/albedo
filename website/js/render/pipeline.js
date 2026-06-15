import { el, mount, link } from "../dom.js";
import { fmtRelative, fmtDateTime, shortHotkey } from "../format.js";
import { hubRepoUrl, modelRepo, modelName, taoMinerUrl } from "../model.js";

const STAGES = [
  { key: "hippius_validate", label: "hippius validate" },
  { key: "pre_eval", label: "pre-eval" },
  { key: "eval", label: "eval" },
];

function row(item, netuid) {
  const name = modelName(item);
  const repo = modelRepo(item.model_uri);
  const repoUrl = hubRepoUrl(item.model_uri);
  const tao = taoMinerUrl(netuid, item.hotkey);
  return el("tr", {},
    el("td", { class: "uid" }, tao ? link(tao, String(item.uid ?? "—")) : String(item.uid ?? "—")),
    el("td", { class: "model" }, repoUrl ? link(repoUrl, name, { class: "model-cell", title: repo }) : el("span", { class: "model-cell", title: repo }, name)),
    el("td", { class: "r when", title: fmtDateTime(item.updated_at) }, fmtRelative(item.updated_at)));
}

function bucket(title, items, live, netuid) {
  const head = el("div", { class: "pl-bucket-head" },
    live ? el("span", { class: "pl-dot" }) : false,
    el("span", { class: "pl-bucket-label" }, title),
    el("span", { class: "pl-bucket-count" }, String(items.length)));
  const body = items.length
    ? el("table", { class: "data-table" }, el("tbody", {}, items.map(it => row(it, netuid))))
    : el("div", { class: "empty" }, "—");
  return el("div", { class: live ? "pl-bucket running" : "pl-bucket" }, head, body);
}

export function renderPipeline(container, state, netuid) {
  const stages = state?.stages || {};
  const counts = state?.counts || {};
  const cards = STAGES.map(s => {
    const stage = stages[s.key] || { running: [], queued: [] };
    const c = counts[s.key] || { running: (stage.running || []).length, queued: (stage.queued || []).length };
    return el("div", { class: "pl-card" },
      el("div", { class: "pl-card-head" },
        el("span", { class: "pl-card-title" }, s.label),
        el("span", { class: "pl-card-meta" }, `${c.running ?? 0} running · ${c.queued ?? 0} queued`)),
      bucket("running", stage.running || [], true, netuid),
      bucket("queued", stage.queued || [], false, netuid));
  });
  mount(container, el("div", { class: "pipeline-grid" }, cards));
}
