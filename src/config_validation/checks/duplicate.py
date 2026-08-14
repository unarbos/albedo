from __future__ import annotations

import logging

from albedo_config.chain_spec import SIM_THRESHOLD
from config_validation.checks import CheckOutcome
from config_validation.fingerprint import compute_fingerprint, similarity
from config_validation.fingerprint.store import FingerprintStore

log = logging.getLogger(__name__)

NAME = "duplicate"


def check(
    model_dir: str, store: FingerprintStore, *, hotkey: str, threshold: float = SIM_THRESHOLD
) -> CheckOutcome:
    fp = compute_fingerprint(model_dir)

    best_sim = 0.0
    best_key = ""
    for entry in store.candidates():
        if hotkey and entry.get("hotkey") == hotkey:
            continue
        sim = similarity(fp, entry.get("fingerprint", {}))
        if sim > best_sim:
            best_sim, best_key = sim, entry.get("key", "")

    is_dup = best_sim >= threshold
    reason = (
        f"near-duplicate of {best_key} (similarity={best_sim:.4f} >= {threshold})" if is_dup else ""
    )
    return CheckOutcome(
        name=NAME,
        ok=not is_dup,
        reason=reason,
        details={
            "fingerprint": fp,
            "best_similarity": best_sim,
            "duplicate_of": best_key if is_dup else "",
        },
    )
