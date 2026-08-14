from __future__ import annotations

from typing import Any

from albedo_config.chain_spec import ALL_LOCK_KEYS
from config_validation.checks import CheckOutcome

NAME = "architecture"


def _lock_violation(seed_cfg: dict[str, Any], cand_cfg: dict[str, Any]) -> str | None:
    if seed_cfg.get("architectures") != cand_cfg.get("architectures"):
        return (
            f"architectures mismatch: seed={seed_cfg.get('architectures')!r} "
            f"candidate={cand_cfg.get('architectures')!r}"
        )
    for key in ALL_LOCK_KEYS:
        if seed_cfg.get(key) != cand_cfg.get(key):
            return (
                f"lock key {key!r} mismatch: seed={seed_cfg.get(key)!r} "
                f"candidate={cand_cfg.get(key)!r}"
            )
    return None


def check(candidate_cfg: dict[str, Any], seed_cfg: dict[str, Any]) -> CheckOutcome:
    if "auto_map" in candidate_cfg:
        return CheckOutcome(NAME, False, "config.json must not contain 'auto_map'")
    if "quantization_config" in candidate_cfg:
        return CheckOutcome(
            NAME,
            False,
            "config.json must not contain 'quantization_config' (quantized models not allowed)",
        )

    reason = _lock_violation(seed_cfg, candidate_cfg)
    ok = reason is None
    return CheckOutcome(
        name=NAME,
        ok=ok,
        reason=reason or "",
        details={
            "architectures": candidate_cfg.get("architectures"),
            "model_type": candidate_cfg.get("model_type"),
        },
    )
