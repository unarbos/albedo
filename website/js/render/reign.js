import { el, mount, link } from "../dom.js";
import { pct, shortHotkey } from "../format.js";
import { kingTitleName, hubRepoUrl, modelRepo, taoMinerUrl } from "../model.js";

export function renderReign(container, reign, netuid) {
  const members = reign?.members || [];
  if (!members.length) {
    mount(container, el("div", { class: "empty" }, "no active reign."));
    return;
  }

  const orderedMembers = [...members].sort((a, b) => (Number(b.king_version) || 0) - (Number(a.king_version) || 0));

  // Mark the most recently coronated king (highest king_version), not just slot 0.
  let glowIdx = 0, bestVersion = -Infinity;
  orderedMembers.forEach((m, i) => {
    const v = Number(m.king_version);
    if (Number.isFinite(v) && v > bestVersion) { bestVersion = v; glowIdx = i; }
  });

  const rows = orderedMembers.map((m, i) => {
    const repo = modelRepo(m.model_uri);
    const repoUrl = hubRepoUrl(m.model_uri);
    const tao = taoMinerUrl(netuid, m.hotkey);
    const weightPct = m.weight_bps != null ? (m.weight_bps / 100).toFixed(0) + "%" : "—";

    const score = (m.score_challenger != null && m.score_king != null)
      ? el("span", { class: "rc-score" },
          el("span", { class: "chal" }, pct(m.score_challenger)),
          el("span", { class: "sep" }, " / "),
          el("span", { class: "king" }, pct(m.score_king)))
      : el("span", { class: "rc-score" }, el("span", { class: "none" }, "no duel scores"));

    return el("tr", { class: i === glowIdx ? "reign-current" : "" },
      el("td", { class: "rc-era" }, kingTitleName(m.king_version)),
      el("td", { class: "model rc-model" },
        repoUrl ? link(repoUrl, repo, { title: m.model_hash || repo }) : el("span", { class: "model-cell" }, repo)),
      el("td", { class: "uid" }, tao ? link(tao, String(m.uid ?? "—")) : String(m.uid ?? "—")),
      el("td", { class: "rc-hotkey", title: m.hotkey || "" }, shortHotkey(m.hotkey)),
      el("td", {}, score),
      el("td", { class: "r rc-weight" },
        el("span", {}, weightPct),
        el("span", { class: "rc-bar" }, el("i", { style: `width:${m.weight_bps != null ? m.weight_bps / 100 : 0}%` }))));
  });

  mount(container, el("div", { class: "data-table-wrap reign-table-wrap" },
    el("table", { class: "data-table reign-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "reign"),
        el("th", {}, "model"),
        el("th", {}, "uid"),
        el("th", {}, "hotkey"),
        el("th", {}, "duel"),
        el("th", { class: "r" }, "weight"))),
      el("tbody", {}, rows))));
}
