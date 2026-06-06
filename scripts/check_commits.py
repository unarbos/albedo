"""Scan the chain for v4 challenger reveals and print them to the terminal.

A reveal is an on-chain commitment of the form ``v4|<repo>|<digest>[|<hotkey>]``.
This is a read-only inspector — it does not touch validator state, the king,
or the dashboard; it just shows every v4 commit currently on chain.

    cd /path/to/albedo
    source .venv/bin/activate
    python scripts/check_commits.py
    python scripts/check_commits.py --hotkey 5HQbmW...jpPd   # only that hotkey

Env (same names the validator uses):
    ALBEDO_NETUID    (default 97)
    ALBEDO_NETWORK   (default finney)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from albedo.models import ModelRef  # noqa: E402

NETUID = int(os.environ.get("ALBEDO_NETUID", "97"))
NETWORK = os.environ.get("ALBEDO_NETWORK", "finney")


# Inlined from albedo.validator.chain (importing that package drags in boto3 et al.).
def _decode_raw(raw) -> str:
    """Normalise a raw commitment value to a UTF-8 string."""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if raw.startswith("0x"):
            try:
                return bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
            except Exception:
                return raw
        return raw
    return str(raw)


def _iter_commitments(raw):
    """Yield (chain_hotkey, reveal_block, data_str) for every commitment.

    Handles both bittensor SDK shapes: dict[hotkey → data] and
    list of (hotkey, [(block, data), ...]) pairs.
    """
    if isinstance(raw, dict):
        for hotkey, value in raw.items():
            yield str(hotkey), None, _decode_raw(value)
        return
    for pair in raw:
        try:
            hotkey = str(pair[0])
            entries = [(int(item[0]), _decode_raw(item[1])) for item in pair[1]]
            if not entries:
                continue
            block, data = sorted(entries, key=lambda t: t[0], reverse=True)[0]
            yield hotkey, block, data
        except Exception:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan chain for v4 reveals.")
    parser.add_argument("--hotkey", help="only show the commit for this ss58 hotkey")
    args = parser.parse_args()

    import bittensor as bt

    print(f"connecting to bittensor network={NETWORK} ...", flush=True)
    subtensor = bt.Subtensor(network=NETWORK)
    block = getattr(subtensor, "block", "?")
    print(f"connected — block={block}  netuid={NETUID}\n", flush=True)

    try:
        raw = subtensor.get_all_commitments(netuid=NETUID)
    except Exception as exc:
        print(f"ERROR: get_all_commitments failed: {exc}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, str, str]] = []
    n_total = n_non_v4 = n_invalid = n_spoofed = 0

    for chain_hotkey, reveal_block, data in _iter_commitments(raw):
        if args.hotkey and chain_hotkey != args.hotkey:
            continue
        n_total += 1
        if not data.startswith("v4|"):
            n_non_v4 += 1
            continue

        parts = data.split("|")
        status = "ok"
        try:
            if len(parts) == 4:
                _, repo, digest, author = parts
                ref = ModelRef(repo=repo, digest=digest)
                if author != chain_hotkey:
                    status = "SPOOFED"
                    n_spoofed += 1
            elif len(parts) == 3:
                _, repo, digest = parts
                ref = ModelRef(repo=repo, digest=digest)
            else:
                raise ValueError(f"unexpected part count {len(parts)}")
        except ValueError as exc:
            n_invalid += 1
            print(f"  invalid v4 from {chain_hotkey}: {exc}", file=sys.stderr)
            continue

        blk = str(reveal_block) if reveal_block is not None else "?"
        rows.append((chain_hotkey, ref.repo, ref.digest, f"{blk} {status}".strip()))

    if not rows:
        where = f"for hotkey {args.hotkey}" if args.hotkey else "on chain"
        print(f"No v4 commits found {where}.")
    else:
        rows.sort(key=lambda r: r[1])  # by repo
        hk_w = max(len("HOTKEY"), max(len(r[0]) for r in rows))
        repo_w = max(len("REPO"), max(len(r[1]) for r in rows))
        print(f"{'HOTKEY':<{hk_w}}  {'REPO':<{repo_w}}  DIGEST / block")
        print(f"{'-' * hk_w}  {'-' * repo_w}  {'-' * 20}")
        for hk, repo, digest, blk in rows:
            print(f"{hk:<{hk_w}}  {repo:<{repo_w}}  {digest}  [{blk}]")

    print(
        f"\nscanned {n_total} commitments — "
        f"v4={len(rows)}  non_v4={n_non_v4}  spoofed={n_spoofed}  invalid={n_invalid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
