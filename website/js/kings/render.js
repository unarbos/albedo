import { kingTitle, shortDigest, shortHotkey, fmtDate, fmtRelative, escHtml } from "./format.js";
import { hubUrl, taoUrl, judgeLetter } from "./model.js";
import { buildKingsList, applyDisplayStartBlock } from "./data.js";
import { EVALS_BASE } from "./config.js";

// The king_chain carries `judges` as judge model-name strings (crown_judges), not
// score objects, so per-judge scores must be resolved: from the crowning verdict in
// the live history, or — for older kings whose verdict has scrolled out — from the
// king's persistent S3 eval bundle (scores.json). Each resolved judge is
// {model, king_mean} where king_mean is THIS king's win fraction for that judge.
const _crownCache   = new Map();   // challenge_id -> judges[] | null (null = no bundle)
const _crownPending = new Set();
let _lastData = null;

function _judgesFromByJudge(bj) {
  if (!bj) return null;
  return Object.entries(bj).map(([model, pct]) => ({ model, king_mean: Number(pct) / 100 }));
}

function _crownScoresUrls(k) {
  const m = String(k.challenge_id || "").match(/(\d+)\s*$/);
  const day = String(k.crowned_at || "").slice(0, 10);
  if (!m || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return [];
  const dir = String(parseInt(m[1], 10)).padStart(3, "0");
  const prev = new Date(day + "T00:00:00Z");
  prev.setUTCDate(prev.getUTCDate() - 1);
  const prevDay = prev.toISOString().slice(0, 10);
  return [`${EVALS_BASE}${day}/${dir}/scores.json`, `${EVALS_BASE}${prevDay}/${dir}/scores.json`];
}

async function _fetchCrownScores(k) {
  const cid = k.challenge_id;
  if (!cid || cid === "seed" || cid === "genesis") return;
  if (_crownCache.has(cid) || _crownPending.has(cid)) return;
  _crownPending.add(cid);
  let judges = null;
  for (const url of _crownScoresUrls(k)) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      judges = _judgesFromByJudge((await r.json()).by_judge);
      if (judges) break;
    } catch { /* try next */ }
  }
  _crownCache.set(cid, judges);
  _crownPending.delete(cid);
  if (judges && _lastData) render(_lastData);
}

function resolveKingJudges(k, history) {
  // Score objects already on the entry (e.g. the current king's history record):
  // this king's per-judge score is its winning (challenger) mean, so display chal_mean.
  if (Array.isArray(k.judges) && k.judges.length && typeof k.judges[0] === "object"
      && (k.judges[0].chal_mean != null || k.judges[0].king_mean != null)) {
    return k.judges.map(j => ({ model: j.model, king_mean: j.chal_mean != null ? j.chal_mean : j.king_mean }));
  }
  // From the crowning verdict in the live history.
  const cid = k.challenge_id;
  const h = (history || []).find(x => x.accepted && (x.eval_id === cid || x.challenge_id === cid));
  const fromVerdict = _judgesFromByJudge(h?.verdict?.by_judge);
  if (fromVerdict) return fromVerdict;
  // From the persistent S3 bundle (older kings); fetch + re-render if not cached yet.
  if (cid && _crownCache.has(cid)) return _crownCache.get(cid) || [];
  _fetchCrownScores(k);
  return [];
}

function renderRow(k, isFirst) {
  const isCurrent = isFirst;
  const title   = kingTitle(k.reign_number);
  const repo    = k.model_repo || "";
  const digest  = k.king_digest || k.model_digest || "";
  const hkUrl   = taoUrl(k.hotkey);
  const mdlUrl  = hubUrl(repo);

  const badgeCls   = isCurrent ? "current" : "past";
  const badgeLabel = isCurrent ? "current"  : "past";
  const inferredNote = k._inferred
    ? `<span class="era-inferred" title="reign number inferred from position">~</span>` : "";
  const eraHtml = `
    <td class="col-era">
      <div class="era-name">${inferredNote}${escHtml(title)}</div>
      <span class="era-badge ${badgeCls}">${badgeLabel}</span>
    </td>`;

  const uid = k.uid != null ? k.uid : "—";
  const uidInner = hkUrl
    ? `<a href="${escHtml(hkUrl)}" target="_blank" rel="noopener" title="${escHtml(k.hotkey||"")}">${uid}</a>`
    : uid;
  const hotkeyInner = k.hotkey
    ? (hkUrl
        ? `<a href="${escHtml(hkUrl)}" target="_blank" rel="noopener" title="${escHtml(k.hotkey)}">${shortHotkey(k.hotkey)}</a>`
        : shortHotkey(k.hotkey))
    : "—";
  const uidHtml = `
    <td class="col-uid">
      <div class="uid-val">${uidInner}</div>
      <div class="hotkey-val">${hotkeyInner}</div>
    </td>`;

  const repoHtml = mdlUrl
    ? `<a href="${escHtml(mdlUrl)}" target="_blank" rel="noopener">${escHtml(repo || "—")}</a>`
    : escHtml(repo || "—");
  const digestHtml = mdlUrl
    ? `<a href="${escHtml(mdlUrl)}" target="_blank" rel="noopener" title="${escHtml(digest)}">${shortDigest(digest)}</a>`
    : shortDigest(digest);
  const modelHtml = `
    <td class="col-model">
      <div class="model-repo">${repoHtml}</div>
      <div class="model-digest">${digestHtml}</div>
    </td>`;

  const judgesHtml = `
    <td class="col-judges">
      <div class="judges-row">
        ${(k.judges || []).map(j => {
          const score = j.king_mean != null ? (j.king_mean * 100).toFixed(1) + "%" : "—";
          return `<span class="judge-pill" title="${escHtml(j.model)}">
            <span class="jl">${judgeLetter(j.model)}</span>${score}
          </span>`;
        }).join("")}
      </div>
    </td>`;

  const dateHtml = `
    <td class="col-date">
      <div class="date-abs">${fmtDate(k.crowned_at)}</div>
      <div class="date-rel">${fmtRelative(k.crowned_at)}</div>
    </td>`;

  const rowCls = isCurrent ? " class=\"is-current\"" : "";
  return `<tr${rowCls}>${eraHtml}${uidHtml}${modelHtml}${judgesHtml}${dateHtml}</tr>`;
}

export function render(d) {
  _lastData = d;
  const fd     = applyDisplayStartBlock(d);
  const kings  = buildKingsList(fd);
  // Resolve real per-judge scores (history verdict or S3 bundle) so the judge pills
  // show D/Q/K + score% instead of "? —".
  kings.forEach(k => { k.judges = resolveKingJudges(k, d.history); });
  const empty  = document.getElementById("kings-empty");
  const table  = document.getElementById("kings-table");
  const tbody  = document.getElementById("kings-tbody");
  const meta   = document.getElementById("kings-meta");
  const notice = document.getElementById("kings-notice");

  if (!kings.length) {
    empty.textContent = "no kings yet.";
    empty.hidden = false;
    table.hidden = true;
    return;
  }

  meta.textContent = `${kings.length} king${kings.length !== 1 ? "s" : ""}`;
  tbody.innerHTML = kings.map((k, i) => renderRow(k, i === 0)).join("");
  empty.hidden = true;
  table.hidden = false;

  const inferredCount = kings.filter(k => k._inferred).length;
  notice.textContent = inferredCount
    ? `~ ${inferredCount} older ${inferredCount === 1 ? "entry" : "entries"} before history window — reign numbers estimated`
    : "";
}
