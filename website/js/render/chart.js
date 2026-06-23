import { el, mount } from "../dom.js";
import { pct } from "../format.js";
import { kingTitleName } from "../model.js";

const BAR_H = 150;
const DOMAIN_LO = 0.40;
const DOMAIN_HI = 0.60;

export function renderChart(container, crownings) {
  if (!crownings.length) {
    mount(container, el("div", { class: "empty" }, "no crownings yet."));
    return;
  }

  const h = v => {
    if (v == null) return 3;
    const frac = (v - DOMAIN_LO) / (DOMAIN_HI - DOMAIN_LO);
    return Math.max(3, Math.min(BAR_H, Math.round(frac * BAR_H)));
  };

  const ordered = [...crownings].sort((a, b) => (a.king_version || 0) - (b.king_version || 0));
  const cols = ordered.map((c, i) => {
    const current = i === ordered.length - 1;
    const [eraTop, eraBot = ""] = kingTitleName(c.king_version).split(/[-\s]/);
    return el("div", { class: current ? "chart-col current" : "chart-col" },
      el("div", { class: "chart-vals" }, el("span", { class: "chal" }, pct(c.chal_mean, 1)), " / " + pct(c.king_mean, 1)),
      el("div", { class: "chart-bars" },
        el("div", { class: "chart-bar chal", style: `height:${h(c.chal_mean)}px`, title: `challenger ${pct(c.chal_mean)}` }),
        el("div", { class: "chart-bar king", style: `height:${h(c.king_mean)}px`, title: `king ${pct(c.king_mean)}` })),
      el("div", { class: "chart-era" }, el("div", {}, eraTop), el("div", {}, eraBot)));
  });

  const scroller = el("div", { class: "chart-scroll" }, el("div", { class: "chart" }, cols));
  mount(container, scroller);
  requestAnimationFrame(() => { scroller.scrollLeft = scroller.scrollWidth; });
}
