"""Check #4: the model is not a near-duplicate of one already on the subnet.

Computes a weight fingerprint for the downloaded model and compares it against the
corpus store, skipping the miner's own prior entries (same hotkey). A match at/above
the configured similarity threshold flags the model as a duplicate.
"""
from __future__ import annotations

import logging

from config_validation.checks import CheckOutcome
from config_validation.config import SIM_THRESHOLD
from config_validation.fingerprint import compute_fingerprint, similarity
from config_validation.fingerprint.store import FingerprintStore

log = logging.getLogger(__name__)

NAME = "duplicate"


def check(model_dir: str, store: FingerprintStore, *, hotkey: str,
          threshold: float = SIM_THRESHOLD) -> CheckOutcome:
    """Fingerprint ``model_dir`` and check it against the corpus.

    The computed fingerprint is returned in ``details['fingerprint']`` so the caller
    can record it (publish / index) regardless of the outcome.
    """
    fp = compute_fingerprint(model_dir)

    best_sim = 0.0
    best_key = ""
    for entry in store.candidates():
        if hotkey and entry.get("hotkey") == hotkey:
            continue  # the miner's own prior model is not a duplicate of itself
        sim = similarity(fp, entry.get("fingerprint", {}))
        if sim > best_sim:
            best_sim, best_key = sim, entry.get("key", "")

    is_dup = best_sim >= threshold
    reason = (f"near-duplicate of {best_key} (similarity={best_sim:.4f} >= {threshold})"
              if is_dup else "")
    return CheckOutcome(
        name=NAME,
        ok=not is_dup,
        reason=reason,
        details={"fingerprint": fp, "best_similarity": best_sim,
                 "duplicate_of": best_key if is_dup else ""},
    )
