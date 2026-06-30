import { el, mount, link } from "../dom.js";
import { fmtCount } from "../format.js";

const hubDatasetUrl = repo => "https://huggingface.co/datasets/" + repo;

export function renderDatasets(container, metaEl, manifest) {
  const sources = manifest?.sources || [];
  if (!sources.length) {
    mount(container, el("div", { class: "empty" }, "no dataset manifest."));
    if (metaEl) metaEl.textContent = "";
    return;
  }

  const ordered = [...sources].sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0));
  let totalRows = 0, totalShards = 0;

  const rows = ordered.map(s => {
    const rowCount = Number(s.total_rows) || 0;
    const shardCount = (s.shards || []).length;
    totalRows += rowCount;
    totalShards += shardCount;
    const weightPct = s.weight != null ? (s.weight * 100).toFixed(0) + "%" : "—";

    return el("tr", {},
      el("td", { class: "ds-name" }, s.name || "—"),
      el("td", { class: "ds-weight" },
        el("span", {}, weightPct),
        el("span", { class: "ds-bar" }, el("i", { style: `width:${s.weight != null ? s.weight * 100 : 0}%` }))),
      el("td", { class: "r ds-num" }, rowCount.toLocaleString()),
      el("td", { class: "r ds-num" }, shardCount.toLocaleString()),
      el("td", { class: "ds-source" },
        s.repo ? link(hubDatasetUrl(s.repo), s.repo, { title: s.repo }) : "—"));
  });

  if (metaEl) {
    metaEl.textContent = `${ordered.length} datasets · ${fmtCount(totalRows)} trajectories · ${totalShards.toLocaleString()} shards`;
  }

  mount(container, el("div", { class: "data-table-wrap" },
    el("table", { class: "data-table datasets-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "dataset"),
        el("th", {}, "weight"),
        el("th", { class: "r" }, "trajectories"),
        el("th", { class: "r" }, "shards"),
        el("th", {}, "source"))),
      el("tbody", {}, rows))));
}
