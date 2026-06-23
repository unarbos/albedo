import { el, mount, link } from "../dom.js";
import { pct, shortHotkey } from "../format.js";
import { kingTitleName, hubRepoUrl, modelRepo, taoMinerUrl } from "../model.js";

export function renderReign(container, reign, netuid) {
  const members = reign?.members || [];
  if (!members.length) {
    mount(container, el("div", { class: "empty" }, "no active reign."));
    return;
  }

  const orderedMembers = [...members].sort((a, b) => (Number(a.king_version) || 0) - (Number(b.king_version) || 0));

  // Mark the most recently coronated king (highest king_version), not just slot 0.
  let glowIdx = 0, bestVersion = -Infinity;
  orderedMembers.forEach((m, i) => {
    const v = Number(m.king_version);
    if (Number.isFinite(v) && v > bestVersion) { bestVersion = v; glowIdx = i; }
  });

  const cards = orderedMembers.map((m, i) => {
    const repo = modelRepo(m.model_uri);
    const repoUrl = hubRepoUrl(m.model_uri);
    const tao = taoMinerUrl(netuid, m.hotkey);
    const weightPct = m.weight_bps != null ? (m.weight_bps / 100).toFixed(0) + "%" : "—";

    const score = (m.score_challenger != null && m.score_king != null)
      ? el("div", { class: "rc-score" },
          el("span", { class: "chal" }, pct(m.score_challenger)),
          el("span", { class: "sep" }, " / "),
          el("span", { class: "king" }, pct(m.score_king)))
      : el("div", { class: "rc-score" }, el("span", { class: "none" }, "no duel scores"));

    return el("div", { class: i === glowIdx ? "reign-card current" : "reign-card" },
      el("div", { class: "rc-top" },
        el("span", { class: "rc-era" }, kingTitleName(m.king_version)),
        el("span", { class: "rc-weight" }, weightPct)),
      el("div", { class: "rc-model" },
        repoUrl ? link(repoUrl, repo, { title: m.model_hash || repo }) : repo),
      el("div", { class: "rc-meta" },
        el("span", {}, "uid ", tao ? link(tao, String(m.uid ?? "—")) : String(m.uid ?? "—")),
        el("span", { title: m.hotkey || "" }, shortHotkey(m.hotkey))),
      score,
      el("div", { class: "rc-bar" }, el("i", { style: `width:${m.weight_bps != null ? m.weight_bps / 100 : 0}%` })));
  });

  mount(container, el("div", { class: "reign-grid" }, cards));
}
