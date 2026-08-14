from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from config_validation.models import decode_raw, parse_reveal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitRecord:
    block: int | None
    hotkey: str
    coldkey: str
    repo: str
    digest: str


def connect(network: str) -> Any:
    import bittensor as bt

    log.info("connecting to bittensor network=%s", network)
    return bt.Subtensor(network=network)


def coldkey_map(subtensor: Any, netuid: int) -> dict[str, str]:
    try:
        metagraph = subtensor.metagraph(netuid)
    except Exception as exc:
        log.warning("coldkey_map: metagraph(%d) failed: %s", netuid, exc)
        return {}
    return {str(n.hotkey): str(n.coldkey) for n in metagraph.neurons}


def commitment_blocks(subtensor: Any, netuid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for k, v in subtensor.query_map("Commitments", "CommitmentOf", [netuid]):
            hk = getattr(k, "value", k)
            val = getattr(v, "value", v)
            blk = val.get("block") if isinstance(val, dict) else None
            if blk is not None:
                out[str(hk)] = int(blk)
    except Exception as exc:
        log.warning("commitment_blocks: CommitmentOf query failed: %s", exc)
    return out


def _iter_commitments(raw: Any) -> Iterator[tuple[str, int | None, str]]:
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
        except Exception as exc:
            log.debug("chain scan: failed to decode pair: %s", exc)


def scan_commits(subtensor: Any, netuid: int) -> list[CommitRecord]:
    log.info("chain scan: querying commitments for netuid=%d", netuid)
    raw = subtensor.get_all_commitments(netuid=netuid)
    coldkeys = coldkey_map(subtensor, netuid)
    blocks = commitment_blocks(subtensor, netuid)

    def _coldkey(hotkey: str) -> str:
        ck = coldkeys.get(hotkey)
        if ck:
            return ck
        try:
            return str(subtensor.get_hotkey_owner(hotkey) or "")
        except Exception as exc:
            log.debug("coldkey lookup failed for %s: %s", hotkey, exc)
            return ""

    records: list[CommitRecord] = []
    n_total = n_skipped = 0
    for chain_hotkey, block, data in _iter_commitments(raw):
        n_total += 1
        if block is None:
            block = blocks.get(chain_hotkey)
        if not data.startswith("v7|"):
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
