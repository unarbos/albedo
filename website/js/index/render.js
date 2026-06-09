import { setText, fmtWhenCell, fmtAlphaDay, kingDateShort, escHtml } from "./format.js";
import { kingTitleName, challengerDisplayName, modelLinkHtml, taoMinerUrl, evalDirUrl } from "./model.js";
import { isValidEval, verdictBadge, judgeScoreCell, judgeByLetter, failReasonCell } from "./data.js";
import { dlButton, failDlButton, DL_ICON } from "./download.js";
import { renderEvolution } from "./evolution.js";
import { applyDisplayStartBlock, buildIndexKings } from "./data.js";
import { judgeMeta } from "./model.js";

function renderHero(d) {
  const el = document.getElementById("hero-king");
  if (!el) return;
  const king = d.king || {};
  const chain = d.chain || {};
  const repo = king.model_repo || chain.seed_repo || "";
  const digest = king.king_digest || king.model_digest || chain.seed_digest || "";
  let reignNumber = king.reign_number;
  if (reignNumber == null && repo) reignNumber = 0;
  el.innerHTML = modelLinkHtml(repo, digest, kingTitleName(reignNumber));
}

function renderReleases(kc, chain, history) {
  const wrap = document.getElementById("releases-wrap");
  const netuid = chain?.netuid;
  if (!kc || kc.length === 0) {
    wrap.innerHTML = '<div class="empty">no releases yet.</div>';
    return;
  }
  const rows = kc.map((e, i) => {
    const era = kingTitleName(e.reign_number);
    const current = i === 0 ? " current" : "";
    const dim = e.registered ? "" : " dim";
    const uid = e.uid != null ? e.uid : "—";
    const hk = e.hotkey || "";
    const taoUrl = taoMinerUrl(netuid, hk);
    const uidCell = taoUrl
      ? `<a href="${taoUrl}" target="_blank" rel="noopener" title="${hk}">${uid}</a>`
      : uid;
    const repo = e.model_repo || "";
    const digest = e.king_digest || e.model_digest || "";
    const modelCell = modelLinkHtml(repo, digest, kingTitleName(e.reign_number));
    const date = kingDateShort(e.crowned_at);
    const alpha = fmtAlphaDay(e.weight);
    const alphaCls = i === 0 && e.registered ? " earning" : (e.registered ? "" : "");
    // King eval download = the ZIP bundle of the crowning duel that put this king
    // on the throne. Prefer the history entry (exact date); fall back to the king's
    // own challenge_id + crowned_at so ALL kings work, not just the one still in the
    // truncated history. Genesis/base model (no numeric challenge_id) has none.
    const kingEval = (history || []).find(
      x => x.accepted && (x.eval_id === e.challenge_id || x.challenge_id === e.challenge_id)
    );
    const dlDir = evalDirUrl(kingEval)
      || evalDirUrl({ eval_id: e.challenge_id, completed_at: e.crowned_at });
    const dlBtn = dlDir
      ? `<button type="button" class="releases-dl" data-zip-dir="${dlDir.replace(/"/g, "&quot;")}" aria-label="download eval ZIP" title="download eval ZIP">`
      : `<button type="button" class="releases-dl" disabled aria-label="download rollouts">`;
    const dlEnd = `</button>`;
    const dlIcon = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M6 1.5v8M2.5 6L6 9.5 9.5 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M2 11h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`;
    return `<tr class="${dim}">
      <td class="era"><span class="${current.trim() || "past"}">${era}</span></td>
      <td class="uid">${uidCell}</td>
      <td class="model"><div class="model-cell">${modelCell}</div></td>
      <td class="date">${date}</td>
      <td class="alpha${alphaCls}" title="${e.weight != null ? (e.weight * 2960).toFixed(4) + " α/day" : ""}">${alpha}</td>
      <td class="dl">${dlBtn}${dlIcon}${dlEnd}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="releases">
    <thead><tr>
      <th>era</th>
      <th>uid</th>
      <th>model</th>
      <th class="r">date</th>
      <th class="r">ⴷ / day</th>
      <th class="col-dl" aria-label="download"></th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function _swebenchReign(row) {
  if (row.reign_number != null) return row.reign_number;
  const runMatch = String(row.run_id || "").match(/reign-(\d+)/);
  if (runMatch) return Number(runMatch[1]);
  const keyMatch = String(row.king_key || "").match(/reign-(\d+)/);
  if (keyMatch) return Number(keyMatch[1]);
  return null;
}

function _swebenchScore(row) {
  if (row.score != null && Number.isFinite(Number(row.score))) return Number(row.score);
  if (row.resolved != null && row.total) return Number(row.resolved) / Number(row.total);
  return null;
}

function _swebenchScoreCell(row) {
  const score = _swebenchScore(row);
  if (score == null) return `<span class="muted-dash">—</span>`;
  return `<span class="swe-score">${(score * 100).toFixed(1)}%</span>`;
}

function _swebenchCountCell(row, key) {
  const value = row[key];
  if (value == null) return `<span class="muted-dash">—</span>`;
  return escHtml(value);
}

function _swebenchJsonUrl(row) {
  const urls = row.s3_urls || row.s3?.urls || row.s3?.public_urls || {};
  return urls.king || urls.summary || urls.run_summary || urls.result || null;
}

function _swebenchStatusCell(row) {
  const status = String(row.status || "unknown").toLowerCase();
  const cls = status === "complete" || status === "completed" ? "complete"
    : status === "running" ? "running"
    : status === "failed" ? "failed"
    : "unknown";
  return `<span class="swe-status ${cls}">${escHtml(status)}</span>`;
}

function renderSwebenchLite(swebenchLite) {
  const wrap = document.getElementById("swebench-lite-wrap");
  if (!wrap) return;
  const rows = Array.isArray(swebenchLite?.benchmarks) ? swebenchLite.benchmarks : [];
  const completed = rows.filter(r => r.status === "complete" || r.status === "completed").length;
  const running = rows.filter(r => r.status === "running").length;
  const metaBits = [];
  if (completed) metaBits.push(`${completed} scored`);
  if (running) metaBits.push(`${running} running`);
  setText("swebench-lite-meta", metaBits.length ? metaBits.join(" · ") : "0 scored");
  if (!rows.length) {
    wrap.innerHTML = '<div class="empty">no SWE-bench Lite results yet.</div>';
    return;
  }
  const ordered = [...rows].sort((a, b) => {
    const ar = _swebenchReign(a) ?? -1;
    const br = _swebenchReign(b) ?? -1;
    if (br !== ar) return br - ar;
    return String(b.started_at || b.completed_at || "").localeCompare(String(a.started_at || a.completed_at || ""));
  });
  const tableRows = ordered.map(row => {
    const repo = row.repo || "";
    const digest = row.digest || "";
    const reign = _swebenchReign(row);
    const label = kingTitleName(reign);
    const jsonUrl = _swebenchJsonUrl(row);
    const jsonCell = jsonUrl
      ? `<a class="swe-json" href="${escHtml(jsonUrl)}" target="_blank" rel="noopener">JSON</a>`
      : `<span class="muted-dash">—</span>`;
    return `<tr>
      <td class="model pl">${modelLinkHtml(repo, digest, label)}</td>
      <td class="r">${_swebenchScoreCell(row)}</td>
      <td class="r">${_swebenchCountCell(row, "resolved")}</td>
      <td class="r">${_swebenchCountCell(row, "total")}</td>
      <td class="r">${_swebenchCountCell(row, "submitted")}</td>
      <td class="center">${_swebenchStatusCell(row)}</td>
      <td class="when">${fmtWhenCell(row.completed_at || row.started_at)}</td>
      <td class="r">${jsonCell}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="data-table swebench-table">
    <thead><tr>
      <th class="pl">king</th>
      <th class="r">score</th>
      <th class="r">resolved</th>
      <th class="r">total</th>
      <th class="r">submitted</th>
      <th class="center">status</th>
      <th>when</th>
      <th class="r">result</th>
    </tr></thead>
    <tbody>${tableRows}</tbody>
  </table>`;
}

function renderQueue(queue, chain, currentEval) {
  const wrap = document.getElementById("queue-wrap");
  const netuid = chain?.netuid;
  const pendingQueue = queue || [];
  const hasLive = !!(currentEval?.hotkey);
  const totalCount = pendingQueue.length + (hasLive ? 1 : 0);
  setText("queue-meta", `${totalCount} pending`);
  if (totalCount === 0) {
    wrap.innerHTML = '<div class="empty">empty.</div>';
    return;
  }
  const renderRow = (q, status) => {
    const uid = q.uid != null ? q.uid : "—";
    const taoUrl = taoMinerUrl(netuid, q.hotkey);
    const uidCell = taoUrl
      ? `<a href="${taoUrl}" target="_blank" rel="noopener" title="${q.hotkey}">${uid}</a>`
      : uid;
    const repo   = q.model_repo   || q.challenger_repo   || "";
    const digest = q.model_digest || q.challenger_digest || "";
    const modelCell = modelLinkHtml(repo, digest, challengerDisplayName(q.hotkey));
    const when = fmtWhenCell(q.queued_at || q.started_at);
    return `<tr>
      <td class="status">${status}</td>
      <td class="uid">${uidCell}</td>
      <td class="model">${modelCell}</td>
      <td class="when">${when}</td>
    </tr>`;
  };
  const liveRow = hasLive
    ? renderRow(currentEval, `<span class="queue-status evaluating">evaluating</span>`)
    : "";
  const pendingRows = pendingQueue.slice(0, 99).map((q, i) =>
    renderRow(q, `<span class="queue-status queued">queue #${i + 1}</span>`)
  ).join("");
  wrap.innerHTML = `<table class="data-table queue-table">
    <colgroup>
      <col class="col-status">
      <col class="col-uid">
      <col class="col-model">
      <col class="col-when">
    </colgroup>
    <thead><tr>
      <th class="status">status</th>
      <th class="uid">uid</th>
      <th class="model">model</th>
      <th class="when">when</th>
    </tr></thead>
    <tbody>${liveRow}${pendingRows}</tbody>
  </table>`;
}

function _kingCell(h, fallbackKing, kingByRepo) {
  const repo   = h.king_model_repo   || fallbackKing?.model_repo   || "";
  const digest = h.king_model_digest || fallbackKing?.king_digest   || fallbackKing?.model_digest || "";
  let reign;
  if (kingByRepo && repo && kingByRepo.has(repo)) {
    reign = kingByRepo.get(repo).reign_number;
  } else {
    reign = h.king_reign_number;
    if (reign == null && (h.king_hotkey || repo)) reign = (fallbackKing?.reign_number ?? 0);
  }
  return `<span class="champion-link">${modelLinkHtml(repo, digest, kingTitleName(reign))}</span>`;
}

function _verdictLink(h, badge) {
  if (!h.eval_id && !h.hotkey) return badge;
  const qs = new URLSearchParams();
  if (h.eval_id)      qs.set("eval_id",    h.eval_id);
  if (h.hotkey)       qs.set("hotkey",     h.hotkey);
  const _dirUrl = evalDirUrl(h);
  if (_dirUrl) qs.set("dir_url",    _dirUrl);
  if (h.error_code)   qs.set("error_code", h.error_code);
  if (h.error_detail) qs.set("error_detail", String(h.error_detail).slice(0, 300));
  if (h.model_repo)   qs.set("model_repo", h.model_repo);
  return `<a class="verdict-link" href="./detail.html?${qs.toString()}">${badge}</a>`;
}

function renderHistory(history, chain, king, kings) {
  const wrap = document.getElementById("history-wrap");
  const netuid = chain?.netuid;
  const judges = (chain?.judge_models || []).map(m => judgeMeta(m));
  const valid = (history || []).filter(isValidEval);
  setText("history-meta", `${valid.length} entries`);
  if (valid.length === 0) {
    wrap.innerHTML = '<div class="empty">no completed duels yet.</div>';
    return;
  }
  const kingByRepo = new Map((kings || []).filter(k => k.model_repo).map(k => [k.model_repo, k]));
  const judgeHead = judges.map(j => `<th class="r">${j.label}</th>`).join("");
  const ordered = valid;
  const rows = ordered.map(h => {
    const byLetter = judgeByLetter(h.judges);
    const uid = h.uid != null ? h.uid : "—";
    const taoUrl = taoMinerUrl(netuid, h.hotkey);
    const uidCell = taoUrl
      ? `<a href="${taoUrl}" target="_blank" rel="noopener" title="${h.hotkey}">${uid}</a>`
      : uid;
    const repo = h.model_repo || "";
    const digest = h.model_digest || "";
    const modelCell = modelLinkHtml(repo, digest, challengerDisplayName(h.hotkey));
    const judgeCells = judges.map(j => `<td class="r">${judgeScoreCell(byLetter[j.letter])}</td>`).join("");
    const lcbFloat = h.verdict?.lcb;
    const lcbGate  = !!h.lcb;
    const lcbCell  = lcbFloat != null
      ? `<span class="lcb-val ${lcbGate ? "lcb-win" : "lcb-lose"}">${Number(lcbFloat).toFixed(4)}</span>`
      : `<span class="muted-dash">—</span>`;
    const badge = verdictBadge(h);
    return `<tr>
      <td class="when pl">${fmtWhenCell(h.completed_at || h.ts)}</td>
      <td class="uid">${uidCell}</td>
      <td class="model">${modelCell}</td>
      <td>${_kingCell(h, king, kingByRepo)}</td>
      <td class="center">${_verdictLink(h, badge)}</td>
      ${judgeCells}
      <td class="r">${lcbCell}</td>
      <td class="dl">${dlButton(evalDirUrl(h))}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="data-table">
    <thead><tr>
      <th class="pl">when</th>
      <th>uid</th>
      <th>model</th>
      <th>vs. champion</th>
      <th class="center">verdict</th>
      ${judgeHead}
      <th class="r">lcb</th>
      <th class="r" aria-label="download"></th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderFails(history, chain, king, kings) {
  const wrap = document.getElementById("fails-wrap");
  const netuid = chain?.netuid;
  const fails = (history || []).filter(h => !isValidEval(h));
  setText("fails-meta", `${fails.length} entries`);
  if (fails.length === 0) {
    wrap.innerHTML = '<div class="empty">no failures.</div>';
    return;
  }
  const kingByRepo = new Map((kings || []).filter(k => k.model_repo).map(k => [k.model_repo, k]));
  const ordered = fails;
  const rows = ordered.map(h => {
    const uid = h.uid != null ? h.uid : "—";
    const taoUrl = taoMinerUrl(netuid, h.hotkey);
    const uidCell = taoUrl
      ? `<a href="${taoUrl}" target="_blank" rel="noopener" title="${h.hotkey}">${uid}</a>`
      : uid;
    const repo = h.model_repo || "";
    const digest = h.model_digest || "";
    const modelCell = modelLinkHtml(repo, digest, challengerDisplayName(h.hotkey));
    const badge = verdictBadge(h);
    return `<tr>
      <td class="when pl">${fmtWhenCell(h.completed_at || h.ts)}</td>
      <td class="uid">${uidCell}</td>
      <td class="model">${modelCell}</td>
      <td>${_kingCell(h, king, kingByRepo)}</td>
      <td class="center">${_verdictLink(h, badge)}</td>
      <td class="fail-reason-cell">${failReasonCell(h)}</td>
      <td class="dl">${failDlButton(h)}</td>
    </tr>`;
  }).join("");
  wrap.innerHTML = `<table class="data-table">
    <thead><tr>
      <th class="pl">when</th>
      <th>uid</th>
      <th>model</th>
      <th>vs. champion</th>
      <th class="center">verdict</th>
      <th>fail reason</th>
      <th class="r" aria-label="download JSON" title="download fail JSON">JSON</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

export function render(d, swebenchLite = null) {
  if (!d) return;
  const fd = applyDisplayStartBlock(d);
  const kings = buildIndexKings(fd);
  renderHero(fd);
  renderEvolution(kings, fd.chain, fd.current_eval);
  renderSwebenchLite(swebenchLite);
  renderReleases(kings, fd.chain, fd.history);
  renderQueue(fd.queue, fd.chain, fd.current_eval);
  renderHistory(fd.history, fd.chain, fd.king, kings);
  renderFails(fd.history, fd.chain, fd.king, kings);
}
