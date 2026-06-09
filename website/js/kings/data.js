import { BITTENSOR_BLOCK_TIME_S } from "./config.js";

export function buildKingsList(d) {
  const chain   = d.king_chain || [];
  const history = d.history || [];

  const evalNum = id => parseInt((id || "").replace(/\D/g, ""), 10) || 0;

  const histById = new Map(
    history.filter(h => h.accepted && h.eval_id).map(h => [h.eval_id, h])
  );

  const byId = new Map();

  // 1. King chain + current king (full live metadata: uid, weight, registered, etc.)
  chain.forEach(k => byId.set(k.challenge_id, { ...k, _rich: true }));
  if (d.king?.challenge_id && !byId.has(d.king.challenge_id)) {
    byId.set(d.king.challenge_id, { ...d.king, _rich: true });
  }

  // 2. Supplement with accepted history entries evicted from the chain.
  histById.forEach((h, id) => {
    if (byId.has(id)) return;
    byId.set(id, {
      challenge_id:  id,
      hotkey:        h.hotkey,
      uid:           h.uid ?? null,
      model_repo:    h.model_repo,
      king_digest:   h.model_digest,
      crowned_at:    h.completed_at,
      crowned_block: null,
      weight:        null,
      registered:    null,
      judges:        h.judges || [],
      _rich:         false,
    });
  });

  // 3. Sort chronologically by eval_id (oldest first), assign absolute reign numbers.
  const sorted = Array.from(byId.values())
    .sort((a, b) => evalNum(a.challenge_id) - evalNum(b.challenge_id));

  const totalKings = Number(d.stats?.accepted) || sorted.length;
  const offset = Math.max(0, totalKings - sorted.length);
  sorted.forEach((k, i) => { k.reign_number = offset + i + 1; });

  // Sync back to d.king so the hero section (if any) reads the correct title.
  if (d.king?.challenge_id) {
    const e = byId.get(d.king.challenge_id);
    if (e) d.king.reign_number = e.reign_number;
  }

  const result = sorted.reverse(); // newest first: current king at top

  // 4. Append genesis/base model from chain metadata (never enters history).
  const seedRepo = d.chain?.seed_repo;
  const alreadyPresent = seedRepo && result.some(k => k.model_repo === seedRepo);
  if (seedRepo && !alreadyPresent) {
    result.push({
      challenge_id:  null,
      hotkey:        null,
      uid:           null,
      model_repo:    seedRepo,
      king_digest:   d.chain?.seed_digest || "",
      crowned_at:    null,
      crowned_block: null,
      weight:        null,
      registered:    false,
      judges:        [],
      reign_number:  0,
      _rich:         false,
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
  if (!refBlock || !refAt) return d;
  const refMs    = new Date(refAt).getTime();
  const cutoffMs = refMs - (refBlock - startBlock) * BITTENSOR_BLOCK_TIME_S * 1000;
  return {
    ...d,
    king_chain: (d.king_chain || []).filter(k =>
      (k.crowned_block != null ? k.crowned_block >= startBlock : true) ||
      (k.crowned_at ? new Date(k.crowned_at).getTime() >= cutoffMs : true)
    ),
    history: (d.history || []).filter(h =>
      !h.completed_at || new Date(h.completed_at).getTime() >= cutoffMs
    ),
    _cutoff: { block: startBlock, ms: cutoffMs },
  };
}
