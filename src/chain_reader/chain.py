"""Bittensor chain reading — discover v5 model commits as Commit records.

Self-contained (no external albedo imports). A v5 commitment is a pipe-delimited
reveal string: ``v5|<repo>|<sha256:digest>``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator

from loguru import logger as log

_BLOCK_HASH_CACHE: dict[int, str] = {}


@dataclass(frozen=True)
class Commit:
    netuid: int
    block_number: int
    block_hash: str | None
    extrinsic_hash: str | None
    uid: int | None
    hotkey: str
    commit_payload: dict[str, Any]
    model_uri: str
    payload_hash: str


def connect(network: str) -> Any:
    import bittensor as bt

    log.info("connecting to bittensor network={}", network)
    return bt.Subtensor(network=network)


def _decode_raw(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if raw.startswith("0x"):
            try:
                return bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
            except ValueError:
                return raw
        return raw
    return str(raw)


def _iter_commitments(raw: Any) -> Iterator[tuple[str, int | None, str]]:
    """Yield (hotkey, block, data) across both bittensor SDK shapes."""
    if isinstance(raw, dict):
        for hotkey, value in raw.items():
            yield str(hotkey), None, _decode_raw(value)
        return
    for pair in raw:
        try:
            hotkey = str(pair[0])
            entries = [(int(item[0]), _decode_raw(item[1])) for item in pair[1]]
            if entries:
                block, data = max(entries, key=lambda t: t[0])
                yield hotkey, block, data
        except Exception as exc:  # noqa: BLE001
            log.debug("failed to decode commitment pair: {}", exc)


def _commitment_blocks(subtensor: Any, netuid: int) -> dict[str, int]:
    """hotkey -> commit block, from Commitments.CommitmentOf (get_all_commitments omits it)."""
    out: dict[str, int] = {}
    try:
        for k, v in subtensor.query_map("Commitments", "CommitmentOf", [netuid]):
            hk = getattr(k, "value", k)
            val = getattr(v, "value", v)
            blk = val.get("block") if isinstance(val, dict) else None
            if blk is not None:
                out[str(hk)] = int(blk)
    except Exception as exc:  # noqa: BLE001
        log.warning("CommitmentOf query failed: {}", exc)
    return out


def _uid_map(subtensor: Any, netuid: int) -> dict[str, int]:
    try:
        meta = subtensor.metagraph(netuid)
        return {str(n.hotkey): int(n.uid) for n in meta.neurons}
    except Exception as exc:  # noqa: BLE001
        log.warning("metagraph({}) failed: {}", netuid, exc)
        return {}


def _block_hash(subtensor: Any, block: int) -> str | None:
    if block in _BLOCK_HASH_CACHE:
        return _BLOCK_HASH_CACHE[block]
    try:
        bh = str(subtensor.get_block_hash(block))
        _BLOCK_HASH_CACHE[block] = bh
        return bh
    except Exception as exc:  # noqa: BLE001
        log.debug("get_block_hash({}) failed: {}", block, exc)
        return None


def _parse_v5(data: str, chain_hotkey: str) -> dict[str, Any] | None:
    """Parse a v5 reveal into a payload dict, or None if not a well-formed v5 reveal."""
    if not data.startswith("v5|"):
        return None
    parts = data.split("|")
    if len(parts) != 3:
        return None
    _, repo, digest = parts
    if "/" not in repo or not digest.startswith("sha256:"):
        return None
    return {
        "version": "v5",
        "repo": repo,
        "digest": digest,
        "author_hotkey": chain_hotkey,
        "spoofed": False,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def scan_commitments(subtensor: Any, netuid: int) -> list[Commit]:
    """Read all current v5 commitments on ``netuid`` and return Commit records."""
    raw = subtensor.get_all_commitments(netuid=netuid)
    blocks = _commitment_blocks(subtensor, netuid)
    uids = _uid_map(subtensor, netuid)

    commits: list[Commit] = []
    n_total = n_skipped = 0
    for hotkey, block, data in _iter_commitments(raw):
        n_total += 1
        payload = _parse_v5(data, hotkey)
        if payload is None:
            n_skipped += 1
            continue
        if block is None:
            block = blocks.get(hotkey)
        if block is None:
            log.warning("no commit block for hotkey={}; skipping", hotkey)
            n_skipped += 1
            continue
        uid = uids.get(hotkey)
        if uid is None:
            log.warning("no uid for hotkey={}; skipping", hotkey)
            n_skipped += 1
            continue
        commits.append(Commit(
            netuid=netuid,
            block_number=block,
            block_hash=_block_hash(subtensor, block),
            extrinsic_hash=None,
            uid=uid,
            hotkey=hotkey,
            commit_payload=payload,
            model_uri=f"{payload['repo']}@{payload['digest']}",
            payload_hash=_payload_hash(payload),
        ))

    log.info("scan: total={} v5_commits={} skipped={}", n_total, len(commits), n_skipped)
    return commits
