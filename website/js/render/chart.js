import { el, mount, link } from "../dom.js";
import { fmtDateTime, fmtRelative, pct } from "../format.js";
import { hubRepoUrl, kingTitleName, modelRepo } from "../model.js";

const evalHref = r => `detail.html?eval_run_id=${encodeURIComponent(r.eval_run_id || "")}`;
const stop = e => e.stopPropagation();

function marginText(c) {
  const margin = c.win_margin ?? (
    c.chal_mean != null && c.king_mean != null
      ? Number(c.chal_mean) - Number(c.king_mean)
      : null
  );
  return margin == null ? "—" : `+${pct(margin, 2)} pts`;
}

export function renderKingHistory(container, crownings) {
  if (!crownings.length) {
    mount(container, el("div", { class: "empty" }, "no crownings yet."));
    return;
  }

  const ordered = [...crownings].sort((a, b) => (b.king_version || 0) - (a.king_version || 0));
  const rows = ordered.map((c, i) => {
    const current = i === 0;
    const title = kingTitleName(c.king_version);
    const previous = kingTitleName(c.king?.king_version);
    const repo = modelRepo(c.model_uri);
    const repoUrl = hubRepoUrl(c.model_uri);
    return el("div", {
      class: current ? "king-history-row current" : "king-history-row",
      onClick: () => { location.href = evalHref(c); },
    },
      el("div", { class: "king-history-main" },
        el("div", { class: "king-history-title" }, title),
        el("div", { class: "king-history-model" },
          repoUrl ? link(repoUrl, repo, { title: repo, onClick: stop }) : repo)),
      el("div", { class: "king-history-vs" },
        el("span", {}, "beat "),
        el("b", {}, previous)),
      el("div", { class: "king-history-margin", title: `challenger ${pct(c.chal_mean)} / king ${pct(c.king_mean)}` },
        marginText(c)),
      el("div", { class: "king-history-when", title: fmtDateTime(c.finished_at) },
        fmtRelative(c.finished_at)));
  });

  mount(container, el("div", { class: "king-history-list" }, rows));
}
