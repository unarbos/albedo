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


def _decode_commitment_pair(pair: tuple[Any, Any]) -> tuple[str, list[tuple[int, str]]]:
    """Return (hotkey_ss58, [(block, payload), ...]) for one RevealedCommitments row.

    Depending on the substrate client path, the payload may arrive as either a
    hex-serialized SCALE byte string (``0x...``) or raw commitment bytes wrapped in a
    Python str via latin-1. We normalize both shapes to bytes, strip the SCALE
    compact-length prefix, and decode the rest as UTF-8.
    """
    key, data = pair
    if not isinstance(key, str):
        raise ValueError(f"unexpected commitment key type {type(key).__name__}")
    out: list[tuple[int, str]] = []
    for entry in data:
        text, block = entry
        if not isinstance(text, str):
            raise ValueError(f"unexpected commitment payload type {type(text).__name__}")
        if text.startswith(("0x", "0X")):
            raw = bytes.fromhex(text[2:])
        else:
            raw = text.encode("latin-1")
        if not raw:
            raise ValueError("empty commitment payload")
        mode = raw[0] & 0b11
        offset = 1 if mode == 0 else 2 if mode == 1 else 4
        out.append((int(block), raw[offset:].decode("utf-8", errors="ignore")))
    return key, out


def _iter_revealed(subtensor: Any, netuid: int) -> Iterator[tuple[str, int, str]]:
    """Yield (hotkey, block, payload) from Commitments.RevealedCommitments.

    TimelockEncrypted commit-reveal entries are not present here until they are revealed,
    so they are skipped for free — we never attempt to decode an encrypted blob.
    """
    qm = subtensor.query_map(module="Commitments", name="RevealedCommitments", params=[netuid])
    for k, v in qm:
        hotkey = str(getattr(k, "value", k))
        data = getattr(v, "value", v)
        try:
            _, entries = _decode_commitment_pair((hotkey, data))
        except Exception as exc:  # noqa: BLE001
            log.debug("failed to decode revealed commitment for {}: {}", hotkey, exc)
            continue
        for block, payload in entries:
            yield hotkey, block, payload


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


def scan_commitments(subtensor: Any, netuid: int, start_block: int = 0) -> list[Commit]:
    """Read all revealed commitments on ``netuid`` and return v5 Commit records.

    Commits before ``start_block`` are skipped — they are not eval candidates (the chain_guard
    ledger covers them instead).
    """
    uids = _uid_map(subtensor, netuid)

    commits: list[Commit] = []
    n_total = n_skipped = 0
    for hotkey, block, data in _iter_revealed(subtensor, netuid):
        n_total += 1
        if block < start_block:
            n_skipped += 1
            continue
        payload = _parse_v5(data, hotkey)
        if payload is None:
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
