from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

from loguru import logger as log
from scalecodec.utils.ss58 import ss58_encode

_BLOCK_HASH_CACHE: dict[int, str] = {}
_PIN_RE = re.compile(r"^(sha256:[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{64})$")


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


def _iter_revealed(
    subtensor: Any, netuid: int, block: int | None = None
) -> Iterator[tuple[str, int, str]]:
    qm = subtensor.query_map(
        module="Commitments", name="RevealedCommitments", params=[netuid], block=block
    )
    for k, v in qm:
        hotkey = str(getattr(k, "value", k))
        data = getattr(v, "value", v)
        try:
            _, entries = _decode_commitment_pair((hotkey, data))
        except Exception as exc:
            log.debug("failed to decode revealed commitment for {}: {}", hotkey, exc)
            continue
        for block, payload in entries:
            yield hotkey, block, payload


def _uid_map(subtensor: Any, netuid: int) -> dict[str, int]:
    try:
        meta = subtensor.metagraph(netuid)
        return {str(n.hotkey): int(n.uid) for n in meta.neurons}
    except Exception as exc:
        log.warning("metagraph({}) failed: {}", netuid, exc)
        return {}


def metagraph_snapshot(subtensor: Any, netuid: int, block: int) -> list[tuple[int, str, int]]:
    meta = subtensor.metagraph(netuid, block=block)
    reg_blocks: dict[int, int] = {}
    qm = subtensor.query_map(
        module="SubtensorModule", name="BlockAtRegistration", params=[netuid], block=block
    )
    for k, v in qm:
        reg_blocks[int(getattr(k, "value", k))] = int(getattr(v, "value", v))
    return [(int(n.uid), str(n.hotkey), reg_blocks.get(int(n.uid), 0)) for n in meta.neurons]


_ZERO_ACCOUNT = ss58_encode(b"\x00" * 32, ss58_format=42)


def hotkey_owner(subtensor: Any, hotkey: str, block: int | None = None) -> str | None:
    owner = subtensor.query_subtensor("Owner", params=[hotkey], block=block)
    value = getattr(owner, "value", owner)
    if value is None:
        return None
    value = str(value)
    return None if value == _ZERO_ACCOUNT else value


def confirm_swaps(subtensor: Any, swaps: list[Any], block: int) -> list[Any]:
    confirmed = []
    for swap in swaps:
        owner = hotkey_owner(subtensor, swap.old_hotkey, block)
        if owner is None:
            confirmed.append(swap)
        else:
            log.warning(
                "[chain-guard] swap candidate REJECTED: uid {} old hotkey {} is still owned "
                "by coldkey {} — registration churn, not swap_hotkey; skipping ledger",
                swap.uid,
                swap.old_hotkey,
                owner,
            )
    return confirmed


def _block_hash(subtensor: Any, block: int) -> str | None:
    if block in _BLOCK_HASH_CACHE:
        return _BLOCK_HASH_CACHE[block]
    try:
        bh = str(subtensor.get_block_hash(block))
        _BLOCK_HASH_CACHE[block] = bh
        return bh
    except Exception as exc:
        log.debug("get_block_hash({}) failed: {}", block, exc)
        return None


def _parse_v7(data: str, chain_hotkey: str) -> dict[str, Any] | None:
    if not data.startswith("v7|"):
        return None
    parts = data.split("|")
    if len(parts) != 3:
        return None
    _, repo, digest = parts
    if "/" not in repo or not _PIN_RE.match(digest):
        return None
    return {
        "version": "v7",
        "repo": repo,
        "digest": digest,
        "author_hotkey": chain_hotkey,
        "spoofed": False,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


_warned_no_uid: set[str] = set()


def scan_commitments(
    subtensor: Any,
    netuid: int,
    start_block: int = 0,
    uids: dict[str, int] | None = None,
    at_block: int | None = None,
) -> list[Commit]:
    if uids is None:
        uids = _uid_map(subtensor, netuid)

    commits: list[Commit] = []
    n_total = n_skipped = 0
    for hotkey, block, data in _iter_revealed(subtensor, netuid, at_block):
        n_total += 1
        if block < start_block:
            n_skipped += 1
            continue
        payload = _parse_v7(data, hotkey)
        if payload is None:
            n_skipped += 1
            continue
        uid = uids.get(hotkey)
        if uid is None:
            if hotkey not in _warned_no_uid:
                _warned_no_uid.add(hotkey)
                log.warning("no uid for hotkey={}; skipping", hotkey)
            n_skipped += 1
            continue
        commits.append(
            Commit(
                netuid=netuid,
                block_number=block,
                block_hash=_block_hash(subtensor, block),
                extrinsic_hash=None,
                uid=uid,
                hotkey=hotkey,
                commit_payload=payload,
                model_uri=f"{payload['repo']}@{payload['digest']}",
                payload_hash=_payload_hash(payload),
            )
        )

    log.info("scan: total={} v7_commits={} skipped={}", n_total, len(commits), n_skipped)
    return commits
