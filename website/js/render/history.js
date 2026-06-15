import { el, mount, link } from "../dom.js";
import { pct, fmtRelative, fmtDateTime } from "../format.js";
import { judgeMeta, hubRepoUrl, modelRepo, modelName, taoMinerUrl, kingTitleName } from "../model.js";
import { verdictInfo, faultCategory, faultCodeLabel } from "../data.js";

const stop = e => e.stopPropagation();

function judgeCell(chal) {
  if (chal == null) return el("span", { class: "muted-dash" }, "—");
  return el("span", { class: "judge-scores" },
    pct(chal), el("span", { class: "sep" }, " / "),
    el("span", { class: "king-score" }, pct(1 - chal)));
}

const evalHref = r => `detail.html?eval_run_id=${encodeURIComponent(r.eval_run_id || "")}`;
const failHref = f => f.eval_run_id
  ? `detail.html?eval_run_id=${encodeURIComponent(f.eval_run_id)}`
  : `detail.html?submission_id=${encodeURIComponent(f.submission_id || "")}`;

export function renderHistory(container, rows, judgeModels, netuid, currentKingEvalRunId) {
  if (!rows.length) {
    mount(container, el("div", { class: "empty" }, "no completed duels match."));
    return;
  }
  const judges = judgeModels?.length ? judgeModels : ["z-ai/glm-5.1", "qwen/qwen3.5-397b-a17b", "deepseek/deepseek-v3.2"];

  const head = el("tr", {},
    el("th", {}, "when"),
    el("th", {}, "uid"),
    el("th", {}, "model"),
    el("th", {}, "vs king"),
    ...judges.map(m => el("th", { class: "center", title: m }, judgeMeta(m).letter)),
    el("th", { class: "r" }, "result"));

  const body = rows.map(r => {
    const v = verdictInfo(r);
    const isCurrentKing = currentKingEvalRunId != null && r.eval_run_id === currentKingEvalRunId;
    const bj = r.score_breakdown?.by_judge || {};
    const repo = modelRepo(r.model_uri);
    const repoUrl = hubRepoUrl(r.model_uri);
    const tao = taoMinerUrl(netuid, r.hotkey);
    const king = r.king || {};
    const kingName = kingTitleName(king.king_version);
    const kingUrl = hubRepoUrl(king.model_uri);
    const kingTitle = modelRepo(king.model_uri);
    return el("tr", { class: isCurrentKing ? "clickable crowned-now" : "clickable", onClick: () => { location.href = evalHref(r); } },
      el("td", { class: "when", title: fmtDateTime(r.finished_at) }, fmtRelative(r.finished_at)),
      el("td", { class: "uid" }, tao ? link(tao, String(r.uid ?? "—"), { onClick: stop }) : String(r.uid ?? "—")),
      el("td", { class: "model" }, repoUrl ? link(repoUrl, modelName(r), { class: "model-cell", title: repo, onClick: stop }) : el("span", { class: "model-cell", title: repo }, modelName(r))),
      el("td", { class: "model vs" }, kingUrl ? link(kingUrl, kingName, { class: "model-cell", title: kingTitle, onClick: stop }) : el("span", { class: "model-cell", title: kingTitle }, kingName)),
      ...judges.map(m => el("td", { class: "center" }, judgeCell(bj[m]))),
      el("td", { class: "r" }, el("span", { class: `verdict-badge ${v.badge}` }, v.badge)));
  });

  mount(container, el("table", { class: "data-table" }, el("thead", {}, head), el("tbody", {}, body)));
}

function failReasonCell(f) {
  const text = (f.fault_message || f.fault_code || faultCategory(f).label).toString();
  return el("td", { class: "fail-reason-cell" },
    el("span", { class: "fail-code", title: f.fault_class || "" }, faultCodeLabel(f)),
    el("span", { class: "fail-reason", title: text }, text));
}

export function renderFails(container, rows, netuid) {
  if (!rows.length) {
    mount(container, el("div", { class: "empty" }, "no failures match."));
    return;
  }
  const body = rows.map(f => {
    const repo = modelRepo(f.model_uri);
    const repoUrl = hubRepoUrl(f.model_uri);
    const tao = taoMinerUrl(netuid, f.hotkey);
    return el("tr", { class: "clickable", onClick: () => { location.href = failHref(f); } },
      el("td", { class: "when", title: fmtDateTime(f.updated_at) }, fmtRelative(f.updated_at)),
      el("td", { class: "uid" }, tao ? link(tao, String(f.uid ?? "—"), { onClick: stop }) : String(f.uid ?? "—")),
      el("td", { class: "model" }, repoUrl ? link(repoUrl, modelName(f), { class: "model-cell", title: repo, onClick: stop }) : el("span", { class: "model-cell", title: repo }, modelName(f))),
      failReasonCell(f));
  });
  mount(container,
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "model"), el("th", {}, "reason"))),
      el("tbody", {}, body)));
}
