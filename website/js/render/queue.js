import { el, mount, link } from "../dom.js";
import { fmtRelative, fmtDateTime } from "../format.js";
import { hubRepoUrl, modelRepo, taoMinerUrl } from "../model.js";

export function renderQueue(container, queue, currentEval, netuid) {
  const rows = [];
  if (currentEval) {
    const prog = currentEval.sample_count
      ? `${currentEval.generated_sample_count ?? 0}/${currentEval.sample_count}`
      : "";
    rows.push(row(currentEval, `${(currentEval.state || "running").toLowerCase()}${prog ? " " + prog : ""}`, true, currentEval.started_at, netuid));
  }
  queue.forEach(q => rows.push(row(q, statusLabel(q.state), false, q.created_at, netuid)));

  if (!rows.length) {
    mount(container, el("div", { class: "empty" }, "queue empty."));
    return;
  }

  mount(container,
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "status"), el("th", {}, "uid"), el("th", {}, "model"), el("th", { class: "r" }, "queued"))),
      el("tbody", {}, rows)));
}

function statusLabel(state) {
  if (!state) return "queued";
  if (state.startsWith("EVAL_RUNNING")) return "evaluating";
  if (state.startsWith("PRE_EVAL")) return "pre-eval";
  return "queued";
}

function row(item, status, live, when, netuid) {
  const repo = modelRepo(item.model_uri);
  const repoUrl = hubRepoUrl(item.model_uri);
  const tao = taoMinerUrl(netuid, item.hotkey);
  return el("tr", {},
    el("td", {}, el("span", { class: live ? "queue-status evaluating" : "queue-status queued" }, status)),
    el("td", { class: "uid" }, tao ? link(tao, String(item.uid ?? "—")) : String(item.uid ?? "—")),
    el("td", { class: "model" }, repoUrl ? link(repoUrl, repo, { class: "model-cell" }) : el("span", { class: "model-cell" }, repo)),
    el("td", { class: "r when", title: fmtDateTime(when) }, fmtRelative(when)));
}
