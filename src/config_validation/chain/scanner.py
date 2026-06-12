"""Bittensor chain I/O — read commitments and resolve them to model submissions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from config_validation.models import decode_raw, parse_reveal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitRecord:
    """One model submission discovered on-chain."""

    block: int | None
    hotkey: str          # on-chain committer (the authority)
    coldkey: str         # owning coldkey, from the metagraph
    repo: str
    digest: str


def connect(network: str) -> Any:
    """Open a read-only Subtensor connection to ``network``."""
    import bittensor as bt  # imported lazily so the module loads without bittensor

    log.info("connecting to bittensor network=%s", network)
    return bt.Subtensor(network=network)


def coldkey_map(subtensor: Any, netuid: int) -> dict[str, str]:
    """Map hotkey -> coldkey from the subnet metagraph."""
    try:
        metagraph = subtensor.metagraph(netuid)
    except Exception as exc:  # noqa: BLE001 — best-effort, missing coldkeys are non-fatal
        log.warning("coldkey_map: metagraph(%d) failed: %s", netuid, exc)
        return {}
    return {str(n.hotkey): str(n.coldkey) for n in metagraph.neurons}


def commitment_blocks(subtensor: Any, netuid: int) -> dict[str, int]:
    """Map hotkey -> commit block from the Commitments.CommitmentOf storage map.

    get_all_commitments returns no block on the dict-shape SDK, so the block is
    resolved separately here. Best-effort: returns {} if the query fails.
    """
    out: dict[str, int] = {}
    try:
        for k, v in subtensor.query_map("Commitments", "CommitmentOf", [netuid]):
            hk = getattr(k, "value", k)
            val = getattr(v, "value", v)
            blk = val.get("block") if isinstance(val, dict) else None
            if blk is not None:
                out[str(hk)] = int(blk)
    except Exception as exc:  # noqa: BLE001 — block is informational, never fatal
        log.warning("commitment_blocks: CommitmentOf query failed: %s", exc)
    return out


def _iter_commitments(raw: Any) -> Iterator[tuple[str, int | None, str]]:
    """Yield (chain_hotkey, block, data_str) across both bittensor SDK shapes.

    - dict[hotkey -> data]                        (bittensor >= 10.x; no block)
    - list[(hotkey, [(block, data), ...])]        (older SDKs; newest block wins)
    """
    if isinstance(raw, dict):
        for hotkey, value in raw.items():
            yield str(hotkey), None, decode_raw(value)
        return

    for pair in raw:
        try:
            hotkey = str(pair[0])
            entries = [(int(item[0]), decode_raw(item[1])) for item in pair[1]]
            if not entries:
                continue
            block, data = max(entries, key=lambda t: t[0])
            yield hotkey, block, data
        except Exception as exc:  # noqa: BLE001 — skip undecodable rows
            log.debug("chain scan: failed to decode pair: %s", exc)


def scan_commits(subtensor: Any, netuid: int) -> list[CommitRecord]:
    """Return every well-formed v5 model submission committed on ``netuid``.

    Non-v5 and malformed payloads are skipped.
    """
    log.info("chain scan: querying commitments for netuid=%d", netuid)
    raw = subtensor.get_all_commitments(netuid=netuid)
    coldkeys = coldkey_map(subtensor, netuid)
    blocks = commitment_blocks(subtensor, netuid)

    def _coldkey(hotkey: str) -> str:
        # Metagraph only covers currently-registered hotkeys; a hotkey that committed
        # then deregistered isn't there. Its coldkey ownership still lives on-chain, so
        # fall back to the owner map for the miss.
        ck = coldkeys.get(hotkey)
        if ck:
            return ck
        try:
            return str(subtensor.get_hotkey_owner(hotkey) or "")
        except Exception as exc:  # noqa: BLE001 — coldkey is informational, never fatal
            log.debug("coldkey lookup failed for %s: %s", hotkey, exc)
            return ""

    records: list[CommitRecord] = []
    n_total = n_skipped = 0
    for chain_hotkey, block, data in _iter_commitments(raw):
        n_total += 1
        if block is None:
            block = blocks.get(chain_hotkey)
        if not data.startswith("v5|"):
            n_skipped += 1
            continue
        try:
            reveal = parse_reveal(data)
        except ValueError as exc:
            log.debug("chain scan: skip (parse error) hotkey=%s: %s", chain_hotkey, exc)
            n_skipped += 1
            continue

        records.append(
            CommitRecord(
                block=block,
                hotkey=chain_hotkey,
                coldkey=_coldkey(chain_hotkey),
                repo=reveal.ref.repo,
                digest=reveal.ref.digest,
            )
        )

    log.info("chain scan: done — total=%d  models=%d  skipped=%d", n_total, len(records), n_skipped)
    return records
