import { fmtScore3 } from "./format.js";
import { judgeMeta, kingTitleName, modelLinkHtml } from "./model.js";
import { kingDateShort } from "./format.js";
import { EVALS_BASE } from "./config.js";

let evoJudgeFilter = "all";
let evoRenderCtx = null;

// Older kings' crowning verdicts scroll out of the live history window, so the
// backend-derived `entry.judges` is empty for them. Recover their scores from the
// persistent S3 eval bundle (scores.json has the same `by_judge` percentages).
const _crownCache   = new Map();   // challenge_id -> judges[] | null (null = no bundle)
const _crownPending = new Set();

function _judgesFromScores(s) {
  const bj = s?.by_judge;
  if (!bj) return null;
  const n = s.n_valid ?? 0;
  return Object.entries(bj).map(([model, pct]) => {
    const chal = Number(pct) / 100;
    return { model, chal_mean: chal, king_mean: 1 - chal, n };
  });
}

function _crownScoresUrls(entry) {
  const m = String(entry.challenge_id || "").match(/(\d+)\s*$/);
  const day = String(entry.crowned_at || entry.completed_at || "").slice(0, 10);
  if (!m || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return [];
  const dir = String(parseInt(m[1], 10)).padStart(3, "0");
  // Evals that finished just after midnight are stored under the day they ran,
  // so also try the previous day if the crowning-day path 404s.
  const prev = new Date(day + "T00:00:00Z");
  prev.setUTCDate(prev.getUTCDate() - 1);
  const prevDay = prev.toISOString().slice(0, 10);
  return [`${EVALS_BASE}${day}/${dir}/scores.json`, `${EVALS_BASE}${prevDay}/${dir}/scores.json`];
}

async function _fetchCrownScores(entry, onReady) {
  const cid = entry.challenge_id;
  if (!cid || cid === "seed" || cid === "genesis") return;
  if (_crownCache.has(cid) || _crownPending.has(cid)) return;
  _crownPending.add(cid);
  let judges = null;
  for (const url of _crownScoresUrls(entry)) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      judges = _judgesFromScores(await r.json());
      if (judges) break;
    } catch { /* try next */ }
  }
  _crownCache.set(cid, judges);   // cache null too, so we don't refetch missing bundles
  _crownPending.delete(cid);
  if (judges && typeof onReady === "function") onReady();
}

export function kingBarScores(entry, index, chain, currentEval) {
  if (index === 0 && currentEval?.judges?.length) {
    return currentEval.judges.map(j => ({
      model:      j.model,
      score:      j.king_mean,
      chal_score: j.chal_mean,
      n:          j.n,
      live:       true,
    }));
  }
  // Prefer history-derived judges; fall back to scores recovered from the S3 bundle.
  const judges = (entry.judges?.length ? entry.judges : null)
              || _crownCache.get(entry.challenge_id) || null;
  if (judges?.length) {
    return judges.map(j => ({
      model:      j.model,
      score:      j.king_mean,
      chal_score: j.chal_mean,
      n:          j.n,
      live:       false,
    }));
  }
  const models = chain?.judge_models || [];
  return models.map(m => ({ model: m, score: null, chal_score: null, n: 0, live: false }));
}

function judgeColumns(kc, chain, currentEval) {
  const models = [];
  const seen = new Set();
  const add = m => { if (m && !seen.has(m)) { seen.add(m); models.push(m); } };
  (chain?.judge_models || []).forEach(add);
  if (currentEval?.judges) currentEval.judges.forEach(j => add(j.model));
  kc.forEach((e, i) => kingBarScores(e, i, chain, currentEval).forEach(s => add(s.model)));
  return models;
}

function renderEvolutionTower(s, filter) {
  const meta = judgeMeta(s.model);
  const hidden = filter !== "all" && filter !== s.model ? " hidden" : "";
  const liveCls = s.live ? " live" : "";
  // No verdict data (e.g. the genesis/base model) → ghost placeholder bar.
  if (s.chal_score == null && s.score == null) {
    return `<div class="evo-tower${hidden}" data-judge="${s.model}" title="${s.model}">
      <div class="evo-tower-metrics"><span class="evo-tower-val">—</span></div>
      <div class="evo-bar missing"><div class="evo-bar-baseline"></div></div>
      <span class="evo-tower-letter">${meta.letter}</span>
    </div>`;
  }
  // Each bar is a full-height (100%) head-to-head split of the crowning duel:
  // the current king it dethroned on TOP (blacked gold), the latest king
  // (challenger that won) as solid gold filling from the BOTTOM.
  // chal_score + score == 1, so the two segments fill the bar.
  const latest  = s.chal_score != null ? Number(s.chal_score) : 1 - Number(s.score);
  const current = s.score != null ? Number(s.score) : 1 - latest;
  const latestPct  = (latest  * 100).toFixed(2);
  const currentPct = (current * 100).toFixed(2);
  const tip = s.model
    + (s.n ? ` · n=${s.n}` : "")
    + ` · latest ${fmtScore3(latest)} / current king ${fmtScore3(current)}`;
  return `<div class="evo-tower${hidden}${liveCls}" data-judge="${s.model}" title="${tip}">
    <div class="evo-tower-metrics">
      <span class="evo-tower-val">${fmtScore3(latest)}</span>
      <span class="evo-tower-val evo-tower-king-val">${fmtScore3(current)}</span>
    </div>
    <div class="evo-bar split">
      <div class="evo-seg current" style="height:${currentPct}%"></div>
      <div class="evo-seg latest" style="height:${latestPct}%"></div>
    </div>
    <span class="evo-tower-letter">${meta.letter}</span>
  </div>`;
}

function renderEvolutionFilters(judges) {
  const filters = document.getElementById("evolution-filters");
  if (!judges.length) { filters.hidden = true; return; }
  filters.hidden = false;
  const allActive = evoJudgeFilter === "all" ? " active" : "";
  const btns = [`<button type="button" class="evo-filter${allActive}" data-filter="all">all</button>`];
  judges.forEach(m => {
    const meta = judgeMeta(m);
    const active = evoJudgeFilter === m ? " active" : "";
    btns.push(`<button type="button" class="evo-filter${active}" data-filter="${m}" title="${m}">
      <span class="evo-filter-key">${meta.letter}</span>${meta.label}
    </button>`);
  });
  filters.innerHTML = btns.join("");
  filters.querySelectorAll(".evo-filter").forEach(btn => {
    btn.onclick = () => {
      evoJudgeFilter = btn.dataset.filter;
      if (evoRenderCtx) renderEvolution(evoRenderCtx.kc, evoRenderCtx.chain, evoRenderCtx.currentEval);
    };
  });
}

export function renderEvolution(kc, chain, currentEval) {
  evoRenderCtx = { kc, chain, currentEval };
  const scroll = document.getElementById("evolution-scroll");
  if (!kc || kc.length === 0) {
    scroll.innerHTML = '<div class="empty">no kings yet.</div>';
    document.getElementById("evolution-filters").hidden = true;
    return;
  }
  // Recover crowning scores for older kings whose verdict left the history window,
  // then re-render once each bundle resolves.
  const reRender = () => { if (evoRenderCtx) renderEvolution(evoRenderCtx.kc, evoRenderCtx.chain, evoRenderCtx.currentEval); };
  kc.forEach(e => {
    const hasObjs = Array.isArray(e.judges) && e.judges.length && typeof e.judges[0] === "object";
    if (!hasObjs && e.challenge_id && !_crownCache.has(e.challenge_id)) _fetchCrownScores(e, reRender);
  });
  const judges = judgeColumns(kc, chain, currentEval);
  const ordered = kc.slice().reverse();

  const groups = ordered.map((e, displayIdx) => {
    const dataIdx = kc.length - 1 - displayIdx;
    const scores = kingBarScores(kc[dataIdx], dataIdx, chain, currentEval);
    const byModel = Object.fromEntries(scores.map(s => [s.model, s]));
    const towers = judges.map(m => {
      const s = byModel[m] || { model: m, score: null, chal_score: null, n: 0, live: false };
      return renderEvolutionTower(s, evoJudgeFilter);
    }).join("");
    const dim = e.registered ? "" : " dim";
    const current = dataIdx === 0 ? " is-current" : "";
    const repo = e.model_repo || "";
    const digest = e.king_digest || e.model_digest || "";
    const name = modelLinkHtml(repo, digest, kingTitleName(e.reign_number));

    // Final score: mean chal / king across judges, from the same resolved scores the
    // bars use (history or S3-recovered) — shown for every king, not just the current one.
    const scored = scores.filter(s => s.chal_score != null && s.score != null);
    let finalHtml;
    if (scored.length) {
      const avgChal = scored.reduce((a, s) => a + s.chal_score, 0) / scored.length;
      const avgKing = scored.reduce((a, s) => a + s.score, 0) / scored.length;
      const cPct = (avgChal * 100).toFixed(1);
      const kPct = (avgKing * 100).toFixed(1);
      const winCls  = avgChal > 0.5 ? " win"  : avgChal < 0.5 ? " lose" : "";
      const loseCls = avgKing  > 0.5 ? " win"  : avgKing  < 0.5 ? " lose" : "";
      finalHtml = `<div class="evo-king-final">
        <span class="evo-final-chal${winCls}">${cPct}</span>
        <span class="evo-final-sep"> / </span>
        <span class="evo-final-king${loseCls}">${kPct}</span>
      </div>`;
    } else {
      // Placeholder keeps the column height identical (e.g. base model) so all bars align.
      finalHtml = `<div class="evo-king-final"><span class="evo-final-sep">—</span></div>`;
    }

    return `<div class="evo-king${dim}${current}">
      <div class="evo-towers">${towers}</div>
      ${finalHtml}
      <div class="evo-king-name">${name}</div>
      <div class="evo-king-date">${kingDateShort(e.crowned_at)}</div>
    </div>`;
  }).join("");

  scroll.innerHTML = `<div class="evo-chart">${groups}</div>`;
  renderEvolutionFilters(judges);
  requestAnimationFrame(() => { scroll.scrollLeft = scroll.scrollWidth; });
}
