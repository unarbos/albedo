import { el, mount } from "../dom.js";
import { fmtDateTime, shortHotkey } from "../format.js";

const HEIGHT = 128;
const PAD = { top: 10, right: 14, bottom: 22, left: 40 };
const RANGE_MS = { "1M": 30 * 24 * 60 * 60 * 1000, "7D": 7 * 24 * 60 * 60 * 1000, "24H": 24 * 60 * 60 * 1000 };
let activeRange = "1M";
const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function timeMs(value) {
  if (typeof value === "number") return value < 1e12 ? value * 1000 : value;
  return new Date(value).getTime();
}

function labelDate(ms) {
  const d = new Date(ms);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

function tao(n) {
  if (n >= 100) return n.toFixed(0);
  if (n >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

function rowsFrom(data) {
  if (Array.isArray(data)) return data;
  for (const key of ["results", "data", "neurons"]) {
    if (Array.isArray(data?.[key])) return data[key];
  }
  return [];
}

function keyText(value) {
  return value?.ss58 || value || "";
}

function pointsFrom(data) {
  const rows = rowsFrom(data);
  const points = (rows || [])
    .map(r => {
      return {
        hotkey: keyText(r.hotkey),
        time: timeMs(r.timestamp ?? r.registration_block_time),
        rawDate: r.timestamp ?? r.registration_block_time,
        price: Number(r.registration_cost_tao),
      };
    })
    .filter(p => p.hotkey && Number.isFinite(p.time) && Number.isFinite(p.price) && p.price >= 0.5)
    .sort((a, b) => a.time - b.time);
  const latest = points[points.length - 1]?.time;
  return latest ? points.filter(p => p.time >= latest - RANGE_MS[activeRange]) : points;
}

export function renderRegistrationChart(container, rows) {
  const points = pointsFrom(rows);
  if (points.length < 2) {
    mount(container, el("div", { class: "empty" }, rows?.detail || "registration history unavailable."));
    return;
  }

  const width = Math.max(container.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const t0 = points[0].time;
  const t1 = points[points.length - 1].time;
  let pMin = Math.min(...points.map(p => p.price));
  let pMax = Math.max(...points.map(p => p.price));
  const pPad = Math.max((pMax - pMin) * 0.08, 0.01);
  pMin = Math.max(0.5, pMin - pPad);
  pMax += pPad;

  const x = t => PAD.left + ((t - t0) / Math.max(t1 - t0, 1)) * w;
  const y = p => PAD.top + (1 - (p - pMin) / Math.max(pMax - pMin, 0.01)) * h;
  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));

  for (let i = 0; i <= 2; i++) {
    const p = pMin + ((pMax - pMin) * i) / 2;
    svg.append(svgEl("line", { x1: PAD.left, y1: y(p), x2: width - PAD.right, y2: y(p), class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(p), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, tao(p)));
  }

  for (let i = 0; i <= 2; i++) {
    const t = t0 + ((t1 - t0) * i) / 2;
    svg.append(svgEl("text", { x: x(t), y: HEIGHT - 6, class: "tick", "text-anchor": "middle" }, labelDate(t)));
  }

  svg.append(svgEl("polyline", {
    points: points.map(p => `${x(p.time).toFixed(1)},${y(p.price).toFixed(1)}`).join(" "),
    class: "reg-line",
  }));
  points.forEach(p => svg.append(svgEl("circle", { cx: x(p.time).toFixed(1), cy: y(p.price).toFixed(1), r: 2, class: "reg-dot" })));

  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  const active = svgEl("circle", { r: 4, class: "reg-active", visibility: "hidden" });
  svg.append(crosshair, active);

  const tip = el("div", { class: "registration-tip", hidden: true });
  function nearest(px) {
    let best = 0;
    for (let i = 1; i < points.length; i++) {
      if (Math.abs(x(points[i].time) - px) < Math.abs(x(points[best].time) - px)) best = i;
    }
    return best;
  }

  svg.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const pointerX = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * width;
    const p = points[nearest(pointerX)];
    const px = x(p.time);
    const py = y(p.price);
    const cssX = (px / width) * rect.width;
    crosshair.setAttribute("x1", px.toFixed(1));
    crosshair.setAttribute("x2", px.toFixed(1));
    crosshair.removeAttribute("visibility");
    active.setAttribute("cx", px.toFixed(1));
    active.setAttribute("cy", py.toFixed(1));
    active.removeAttribute("visibility");
    mount(tip,
      el("b", {}, `${tao(p.price)} TAO`),
      el("span", { title: p.hotkey }, ` · ${shortHotkey(p.hotkey)}`),
      el("div", { class: "registration-tip-meta", title: fmtDateTime(p.time) }, labelDate(p.time)));
    tip.hidden = false;
    const left = cssX + 12 + tip.offsetWidth > rect.width ? cssX - tip.offsetWidth - 12 : cssX + 12;
    tip.style.left = `${Math.max(left, 0)}px`;
    tip.style.top = `${PAD.top}px`;
  });
  svg.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    active.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });

  mount(container,
    el("div", { class: "registration-chart-head" },
      el("span", {}, "registration price"),
      el("span", { class: "registration-range" },
        Object.keys(RANGE_MS).map(range => el("button", {
          type: "button",
          class: range === activeRange ? "active" : "",
          onClick: () => { activeRange = range; renderRegistrationChart(container, rows); },
        }, range))),
      el("span", {}, `${points.length} hotkeys`)),
    el("div", { class: "registration-plot" }, svg, tip));
}
