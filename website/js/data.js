export function normalize(d) {
  const runs = d.eval_runs || [];
  return {
    updatedAt: d.updated_at,
    chain: d.chain || {},
    stats: d.stats || {},
    reign: d.reign || { members: [] },
    currentEval: d.current_eval || null,
    queue: d.queue || [],
    history: runs,
    fails: d.fails || [],
    crownings: crownings(runs),
  };
}

export function verdictInfo(r) {
  return {
    won: r.challenger_won === true,
    badge: r.coronated ? "crowned" : r.challenger_won ? "won" : "lost",
    chalMean: r.score_challenger,
    kingMean: r.score_king,
    winMargin: r.win_margin,
  };
}

function crownings(runs) {
  return runs
    .filter(r => r.coronated)
    .map(r => ({
      king_version: r.king_version,
      chal_mean: r.score_challenger,
      king_mean: r.score_king,
      win_margin: r.win_margin,
      model_uri: r.model_uri,
      hotkey: r.hotkey,
      uid: r.uid,
      king: r.king || null,
      eval_run_id: r.eval_run_id,
      finished_at: r.finished_at,
    }))
    .sort((a, b) => (a.king_version || 0) - (b.king_version || 0));
}

const FAULT_CLASS_LABELS = {
  MINER_FAULT: "miner",
  INFRA_FAULT: "infra",
  REMOTE_EVAL_FAULT: "remote eval",
  CHAIN_FAULT: "chain",
  PROVIDER_FAULT: "provider",
  UNKNOWN_FAULT: "unknown",
};

export function faultCategory(f) {
  return { label: FAULT_CLASS_LABELS[f.fault_class] || "error" };
}

// Friendly labels for specific fault codes; falls back to the raw code when unmapped.
const FAULT_CODE_LABELS = {
  hotkey_reused: "hotkey reused",
};

export function faultCodeLabel(f) {
  return FAULT_CODE_LABELS[f.fault_code] || f.fault_code || faultCategory(f).label;
}
