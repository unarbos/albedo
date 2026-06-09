import { judgeMeta } from "./model.js";
import { escHtml } from "./format.js";
import { BITTENSOR_BLOCK_TIME_S } from "./config.js";

export function isDuplicateEntry(h) {
  // error_code is the canonical field (new records); fall back to old patterns.
  return h.error_code === "duplicate_model"
    || h.is_duplicate
    || (h.error_detail || h.detail || "").startsWith("duplicate_model")
    || (h.error_detail || h.detail || "").startsWith("too similar to");
}

export function isInjectionEntry(h) {
  // identity_mismatch and not_registered are separate categories — don't bucket them
  // as injection so they get their own display treatment.
  return h.error_code === "chal_injection_detected"
    || h.is_injection
    || (h.error_detail || h.detail || "").includes("chal_injection_detected");
}

export function isIdentityInvalid(h) {
  return h.error_code === "identity_mismatch" || h.error_code === "not_registered";
}

export function isValidEval(h) {
  // A record is a "valid eval" (shown in history, not fails) when there is no
  // error_code AND it has actual judge data. failure records always have error_code.
  return !h.error_code && !h.code && (h.n_turns != null ? h.n_turns > 0 : true);
}

export function isMinerFault(h) {
  const code   = h.error_code || h.code || "";
  const detail = h.error_detail || h.detail || "";
  if (code === "no_king") return true;
  if (code === "config_mismatch") {
    if (detail.includes("cannot materialize") || detail.includes("cannot list challenger") ||
        detail.includes("materialize_failed") || detail.includes("404")) return false;
    return true;
  }
  if (detail.includes("chal_vllm_start_failed") || code === "chal_vllm_start_failed") return true;
  if (isInjectionEntry(h)) return true;
  if (isIdentityInvalid(h)) return true;
  if (isDuplicateEntry(h)) return true;
  return false;
}

export function verdictBadge(h) {
  // Check specific invalid categories first (they have their own error_codes).
  if (isIdentityInvalid(h)) return `<span class="verdict-badge lost">invalid</span>`;
  if (isDuplicateEntry(h))  return `<span class="verdict-badge lost">invalid</span>`;
  if (isInjectionEntry(h))  return `<span class="verdict-badge lost">invalid</span>`;
  if ((h.error_code || h.code) === "config_mismatch") return `<span class="verdict-badge lost">invalid</span>`;
  if ((h.error_code || h.code) === "challenger_rejected") return `<span class="verdict-badge lost">invalid</span>`;
  if ((h.error_code || h.code) === "admission_rejected") return `<span class="verdict-badge lost">invalid</span>`;
  if ((h.error_code || h.code) === "eval_infra" || (h.error_code || h.code) === "infra_failure")
    return `<span class="verdict-badge error">error</span>`;
  if (h.error_code || h.code) return `<span class="verdict-badge error">error</span>`;
  if (h.accepted) return `<span class="verdict-badge crowned">crowned</span>`;
  if ((h.n_turns ?? 0) === 0 && !(h.judges || []).length) return `<span class="verdict-badge error">error</span>`;
  return `<span class="verdict-badge lost">lost</span>`;
}

export function judgeScoreCell(j) {
  if (!j || j.chal_mean == null || j.king_mean == null) return `<span class="muted-dash">—</span>`;
  const chal = (Number(j.chal_mean) * 100).toFixed(2);
  const king = (Number(j.king_mean) * 100).toFixed(2);
  return `<span class="judge-scores">${chal}<span class="sep"> / </span><span class="king-score">${king}</span></span>`;
}

export function judgeByLetter(judges) {
  const map = {};
  (judges || []).forEach(j => {
    const letter = judgeMeta(j.model).letter;
    map[letter] = j;
  });
  return map;
}

export function failReasonCell(h) {
  const code   = h.error_code || h.code   || "";
  const detail = h.error_detail || h.detail || "";

  // Duplicate — shown before generic error_code path to avoid it being swallowed.
  if (isDuplicateEntry(h)) {
    let dupOf = h.duplicate_of || "";
    if (!dupOf) {
      const d = detail;
      if (d.startsWith("too similar to")) {
        dupOf = d.replace(/^too similar to\s*/i, "").replace(/\s*\(sim=.*\)$/, "").trim();
      } else if (d.startsWith("duplicate_model:")) {
        dupOf = d.slice("duplicate_model:".length).trim().replace(/^too similar to\s*/i, "");
      } else if (d.startsWith("duplicate_model")) {
        dupOf = d.slice("duplicate_model".length).trim().replace(/^[\s:]+/, "").replace(/^too similar to\s*/i, "");
      }
    }
    const safedup = escHtml(dupOf);
    const simStr  = h.duplicate_sim != null ? ` · sim ${Number(h.duplicate_sim).toFixed(4)}` : "";
    const label   = safedup
      ? `duplicate of <span class="fail-reason" title="${safedup}">${safedup}</span>${simStr}`
      : "duplicate model";
    return `<span class="outcome-lose">⚠ ${label}</span>`;
  }

  // Injection
  if (isInjectionEntry(h)) {
    let display = detail
      .replace(/^chal_injection_detected:\s*/i, "")
      .replace(/^injection_finetuned:\s*/i, "")
      .trim();
    if (!display) display = "injection attempt detected";
    const safe = escHtml(display);
    const label = `injection: <span class="fail-reason" title="${safe}">${safe}</span>`;
    return `<span class="outcome-lose">⚠ ${label}</span>`;
  }

  // Identity / registration invalid
  if (isIdentityInvalid(h)) {
    let display = detail
      .replace(/^identity_mismatch:\s*/i, "")
      .replace(/^not_registered:\s*/i, "")
      .trim();
    if (!display) {
      display = code === "identity_mismatch"
        ? "payload hotkey does not match chain key"
        : "hotkey not registered on metagraph at eval time";
    }
    const safe   = escHtml(display);
    const prefix = code === "identity_mismatch" ? "spoofed identity" : "not registered";
    const label  = `${prefix}: <span class="fail-reason" title="${safe}">${safe}</span>`;
    return `<span class="outcome-lose">⚠ ${label}</span>`;
  }

  // Generic error_code (includes eval_infra, chal_vllm_start_failed, etc.)
  if (code) {
    const safe = escHtml(detail || code);
    return `<span class="fail-code">${escHtml(code)}</span> <span class="fail-reason" title="${safe}">${safe}</span>`;
  }

  // Lost (accepted=false with valid judge data)
  if (!h.accepted && (h.judges || []).length) {
    const parts = h.judges.map(j => {
      const meta    = judgeMeta(j.model);
      const outcome = j.outcome || "tie";
      const cls     = outcome === "win" ? "outcome-win" : outcome === "lose" ? "outcome-lose" : "outcome-tie";
      return `<span class="${cls}">${meta.letter}: ${outcome}</span>`;
    });
    return parts.join(' <span class="sep">·</span> ');
  }

  return '<span class="muted-dash">—</span>';
}

// Build the full king lineage for the releases table + evolution chart. The backend
// keeps the current king in d.king, separate from the on-chain king_chain, so we merge
// it in. Kings that have been evicted from the chain (depth capped at 5) are recovered
// from accepted history entries. Reign numbers are re-derived from chronological order
// using eval_id as a monotonic counter, offset by stats.accepted so the current king's
// reign reflects the true total number of crownings ever.
export function buildIndexKings(d) {
  const chain   = d.king_chain || [];
  const history = d.history || [];

  const evalNum = id => parseInt((id || "").replace(/\D/g, ""), 10) || 0;

  const histById = new Map(history.filter(h => h.accepted && h.eval_id).map(h => [h.eval_id, h]));

  // Chain entries have full metadata; fall back to history's pre-built judges array if needed.
  // Return the original object when no change is needed so reign_number mutations propagate
  // back to d.king (which renderHero reads directly).
  const normChain = (k) => {
    const hasScores = Array.isArray(k.judges) && k.judges.length && typeof k.judges[0] === "object";
    if (!hasScores) {
      const h = histById.get(k.challenge_id);
      if (h?.judges?.length) return { ...k, judges: h.judges };
    }
    return k;
  };

  const byId = new Map();

  // 1. King chain + current king (full metadata: hotkey, uid, coldkey, weight, etc.)
  chain.forEach(k => byId.set(k.challenge_id, normChain(k)));
  if (d.king?.challenge_id && !byId.has(d.king.challenge_id)) {
    byId.set(d.king.challenge_id, normChain(d.king));
  }

  // 2. Supplement from accepted history entries evicted from the chain.
  //    These lack live uid_map metadata (coldkey, registered, weight) but have
  //    model, hotkey, uid, timestamp, and pre-built judge scores.
  histById.forEach((h, id) => {
    if (byId.has(id)) return;
    byId.set(id, {
      challenge_id:  id,
      hotkey:        h.hotkey,
      model_repo:    h.model_repo,
      model_digest:  h.model_digest,
      uid:           h.uid,
      crowned_at:    h.completed_at,
      crowned_block: null,
      registered:    false,   // no longer in live uid_map; dim the row
      weight:        null,
      weight_share:  null,
      judges:        h.judges || [],
    });
  });

  // 3. Sort chronologically by eval_id number (oldest first).
  const sorted = Array.from(byId.values())
    .sort((a, b) => evalNum(a.challenge_id) - evalNum(b.challenge_id));

  // 4. Re-assign reign numbers. stats.accepted is the total count of all crownings ever,
  //    including any that have scrolled out of the history window. Use it as the ceiling
  //    so the current king's reign label matches the true historical count.
  const totalKings = Number(d.stats?.accepted) || sorted.length;
  const offset = Math.max(0, totalKings - sorted.length);
  sorted.forEach((k, i) => { k.reign_number = offset + i + 1; });

  // normChain may have returned a spread copy rather than d.king itself, so the
  // reign_number mutation above may not have reached d.king. Sync it back explicitly
  // so renderHero (which reads d.king.reign_number directly) shows the correct title.
  if (d.king?.challenge_id) {
    const e = byId.get(d.king.challenge_id);
    if (e) d.king.reign_number = e.reign_number;
  }

  const result = sorted.reverse(); // newest first for the UI

  // 5. Append the genesis/base model at the end (bottom of table, oldest).
  //    It is set directly at subnet init — never enters history or king_chain once
  //    evicted — so reconstruct it from chain metadata. Skip if it's somehow still
  //    in king_chain (model_repo match) to avoid a duplicate row.
  const seedRepo = d.chain?.seed_repo;
  const alreadyPresent = seedRepo && result.some(k => k.model_repo === seedRepo);
  if (seedRepo && !alreadyPresent) {
    result.push({
      challenge_id:  null,
      model_repo:    seedRepo,
      model_digest:  d.chain?.seed_digest || "",
      reign_number:  0,    // kingTitleName(0) → "base model"
      crowned_at:    null,
      crowned_block: null,
      registered:    false,
      weight:        null,
      weight_share:  null,
      judges:        [],
      hotkey:        null,
      uid:           null,
    });
  }

  return result;
}

export function applyDisplayStartBlock(d) {
  const startBlock = d.chain?.display_start_block;
  if (!startBlock || startBlock <= 0) return d;

  const ref = d.king || (d.king_chain || [])[0];
  const refBlock = ref?.crowned_block;
  const refAt    = ref?.crowned_at;
  if (refBlock == null || refBlock <= 0 || !refAt) return d;

  const refMs      = new Date(refAt).getTime();
  const diffBlocks = refBlock - startBlock;
  const cutoffMs   = refMs - diffBlocks * BITTENSOR_BLOCK_TIME_S * 1000;
  const cutoffIso  = new Date(cutoffMs).toISOString();

  const afterCutoff     = iso => !iso || new Date(iso).getTime() >= cutoffMs;
  const blockAfterCutoff = block => block == null || block >= startBlock;

  return {
    ...d,
    history:    (d.history    || []).filter(h => afterCutoff(h.completed_at || h.ts)),
    king_chain: (d.king_chain || []).filter(k => blockAfterCutoff(k.crowned_block) || afterCutoff(k.crowned_at)),
    queue:      (d.queue      || []).filter(q => afterCutoff(q.queued_at) || afterCutoff(q.started_at)),
    _cutoff: { block: startBlock, iso: cutoffIso },
  };
}
