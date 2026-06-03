"""albedo.validator.chain — Bittensor chain I/O for reveal commitments."""
from __future__ import annotations

import logging
from typing import Any

from albedo.models import parse_reveal_v4

log = logging.getLogger(__name__)


def _decode_commitment_pair(pair: Any) -> tuple[str, list[tuple[int, str]]]:
    """Decode a raw commitment pair into (hotkey_ss58, [(block, data), ...]).

    Normalises bytes/hex-string variants across bittensor SDK versions.
    """
    hotkey: str = str(pair[0])
    entries: list[tuple[int, str]] = []

    for item in pair[1]:
        block_num: int = int(item[0])
        raw = item[1]

        if isinstance(raw, (bytes, bytearray)):
            try:
                data = raw.decode("utf-8", errors="replace")
            except Exception:
                data = raw.hex()
        elif isinstance(raw, str):
            if raw.startswith("0x"):  # hex string from newer SDK versions
                try:
                    data = bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
                except Exception:
                    data = raw
            else:
                data = raw
        else:
            data = str(raw)

        entries.append((block_num, data))

    return hotkey, entries


def scan_reveals(
    subtensor: Any,
    netuid: int,
    completed_repos: set[str],
    seen: set[str],
    *,
    king_hotkeys: set[str] = frozenset(),
) -> list[dict]:
    """Query on-chain commitments and return new v4 entries.

    Drops non-v4 formats, non-sha256 digests, already-seen hotkeys, already-completed
    repos, and the current king/king-chain hotkeys (king_hotkeys). Spoofed reveals
    (payload hotkey != chain hotkey) are appended to rejected_out.
    """
    results: list[dict] = []
    n_total = n_seen = n_king = n_completed = n_non_v4 = n_invalid = 0

    log.info("chain scan: querying on-chain commitments for netuid=%d", netuid)

    try:
        raw_commitments = subtensor.get_all_commitments(netuid=netuid)
    except Exception as exc:
        log.warning("chain scan: get_all_commitments failed: %s", exc)
        return results

    for pair in raw_commitments:
        try:
            chain_hotkey, block_entries = _decode_commitment_pair(pair)
        except Exception as exc:
            log.debug("chain scan: decode failed for pair: %s", exc)
            n_invalid += 1
            continue

        n_total += 1

        if chain_hotkey in seen:
            log.debug("chain scan: skip (already seen) hotkey=%s", chain_hotkey)
            n_seen += 1
            continue

        if chain_hotkey in king_hotkeys:
            log.debug("chain scan: skip (king/chain) hotkey=%s", chain_hotkey)
            n_king += 1
            continue

        if not block_entries:
            n_invalid += 1
            continue

        # Use the most-recent entry for this hotkey
        block_entries_sorted = sorted(block_entries, key=lambda t: t[0], reverse=True)
        reveal_block, data = block_entries_sorted[0]

        if not data.startswith("v4|"):
            log.debug("chain scan: skip (non-v4) hotkey=%s", chain_hotkey)
            n_non_v4 += 1
            continue

        try:
            ref = parse_reveal_v4(data)
        except ValueError as exc:
            log.debug("chain scan: skip (parse error) hotkey=%s: %s", chain_hotkey, exc)
            n_invalid += 1
            continue

        if ref.repo in completed_repos:
            log.debug("chain scan: skip (already evaluated) repo=%s hotkey=%s", ref.repo, chain_hotkey)
            n_completed += 1
            continue

        entry: dict = {
            "hotkey":       chain_hotkey,
            "block":        reveal_block,
            "model_repo":   ref.repo,
            "model_digest": ref.digest,
        }
        results.append(entry)
        log.info(
            "chain scan: NEW COMMIT — hotkey=%s  repo=%s  block=%d",
            chain_hotkey, ref.repo, reveal_block,
        )

    log.info(
        "chain scan: done — total=%d  new=%d  seen=%d  king=%d  completed=%d  non_v4=%d  invalid=%d",
        n_total, len(results), n_seen, n_king, n_completed, n_non_v4, n_invalid,
    )
    return results
