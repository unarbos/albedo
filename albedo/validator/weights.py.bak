"""albedo.validator.weights — Emission weight management."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albedo.storage import State

log = logging.getLogger(__name__)

BURN_UID: int = int(os.environ.get("ALBEDO_BURN_UID", "0"))
WEIGHT_INTERVAL: int = int(os.environ.get("ALBEDO_WEIGHT_INTERVAL", "300"))


async def maybe_set_weights(
    subtensor: object,
    wallet: object,
    state: "State",
    *,
    netuid: int,
    force: bool = False,
    reason: str = "",
) -> bool:
    """Set on-chain weights; returns True on success.

    uid_map is snapshotted before the first await — TOCTOU fix (metagraph
    may refresh mid-flight). Falls back to BURN_UID when no kings are
    registered. Empty message on failure indicates a silent rate-limit.
    """
    # Snapshot uid_map BEFORE first await — TOCTOU fix
    uid_map: dict[str, int] = dict(state.uid_map)

    current_block: int = getattr(subtensor, "block", 0)

    if not force:
        blocks_since = current_block - state.last_weight_block
        if blocks_since < WEIGHT_INTERVAL:
            log.debug(
                "maybe_set_weights: skipping — %d/%d blocks elapsed",
                blocks_since, WEIGHT_INTERVAL,
            )
            return False

    eligible = state.eligible_hotkeys(uid_map)

    if eligible:
        uids: list[int] = [uid_map[hk] for hk in eligible if hk in uid_map]
        n = len(uids)
        weights: list[float] = [1.0 / n] * n if n else []
    else:
        log.warning("maybe_set_weights: no registered kings — burning to uid %d", BURN_UID)
        uids = [BURN_UID]
        weights = [1.0]

    if not uids:
        log.warning("maybe_set_weights: uid list empty after filtering, aborting")
        return False

    # netuid is passed explicitly — never derive from private subtensor attributes
    log_reason = f" ({reason})" if reason else ""
    log.info(
        "maybe_set_weights: setting weights%s — uids=%s  block=%d",
        log_reason, uids, current_block,
    )

    try:
        loop = asyncio.get_running_loop()
        success, message = await loop.run_in_executor(
            None,
            lambda: subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            ),
        )
    except Exception as exc:
        log.error("maybe_set_weights: set_weights raised: %s", exc)
        return False

    if success:
        state.last_weight_block = current_block
        log.info("maybe_set_weights: success at block %d", current_block)
        return True

    # Empty message = silent rate-limit; non-empty = real error
    if not message:
        log.warning("maybe_set_weights: rate-limited (success=False, no message)")
    else:
        log.error("maybe_set_weights: failed — %s", message)

    return False
