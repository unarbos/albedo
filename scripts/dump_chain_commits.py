#!/usr/bin/env python3
"""Dump Bittensor subnet commitments to a JSON file.

This is a read-only helper for inspecting on-chain commits. It preserves every
commitment returned by bittensor, decodes bytes/hex payloads, and parses Albedo
v4 reveal strings when possible:

  v4|<repo>|<sha256:digest>|<hotkey>
  v4|<repo>|<sha256:digest>

Examples:
  python3 scripts/dump_chain_commits.py --netuid 97 --out commits.json
  python3 scripts/dump_chain_commits.py --network test --netuid 97 --out /tmp/commits.json
  python3 scripts/dump_chain_commits.py --only-v4 --pretty
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _decode_raw(raw: Any) -> str:
    """Normalize a raw commitment value to a readable string."""
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


def _raw_to_jsonable(raw: Any) -> Any:
    """Return a JSON-safe representation of the raw commitment payload."""
    if isinstance(raw, bytes):
        return {"type": "bytes", "hex": raw.hex()}
    if isinstance(raw, bytearray):
        return {"type": "bytearray", "hex": bytes(raw).hex()}
    try:
        json.dumps(raw)
        return raw
    except TypeError:
        return repr(raw)


def _iter_commitments(raw: Any):
    """Yield dict records from bittensor get_all_commitments output.

    Supports the two shapes seen across bittensor SDK versions:
      - dict[hotkey -> data]
      - list[(hotkey, [(block, data), ...])]
    """
    if isinstance(raw, dict):
        for hotkey, value in raw.items():
            yield {
                "hotkey": str(hotkey),
                "block": None,
                "raw": _raw_to_jsonable(value),
                "data": _decode_raw(value),
            }
        return

    for pair in raw:
        try:
            hotkey = str(pair[0])
            entries = [(int(item[0]), item[1]) for item in pair[1]]
        except Exception as exc:
            yield {
                "hotkey": None,
                "block": None,
                "raw": _raw_to_jsonable(pair),
                "data": "",
                "decode_error": str(exc),
            }
            continue

        for block, value in sorted(entries, key=lambda t: t[0], reverse=True):
            yield {
                "hotkey": hotkey,
                "block": block,
                "raw": _raw_to_jsonable(value),
                "data": _decode_raw(value),
            }


def _parse_v4(data: str, chain_hotkey: str | None) -> dict[str, Any]:
    """Parse an Albedo v4 reveal payload, returning metadata and errors."""
    out: dict[str, Any] = {
        "is_v4": data.startswith("v4|"),
        "repo": None,
        "digest": None,
        "author_hotkey": None,
        "spoofed": False,
        "parse_error": None,
    }
    if not out["is_v4"]:
        return out

    parts = data.split("|")
    try:
        if len(parts) == 4:
            _, repo, digest, author_hotkey = parts
        elif len(parts) == 3:
            _, repo, digest = parts
            author_hotkey = chain_hotkey
        else:
            raise ValueError(f"unexpected v4 part count: {len(parts)}")

        if not repo or "/" not in repo:
            raise ValueError(f"invalid repo: {repo!r}")
        if not digest.startswith("sha256:"):
            raise ValueError(f"invalid digest: {digest!r}")

        out.update({
            "repo": repo,
            "digest": digest,
            "author_hotkey": author_hotkey,
            "spoofed": bool(chain_hotkey and author_hotkey and author_hotkey != chain_hotkey),
        })
    except Exception as exc:
        out["parse_error"] = str(exc)

    return out


def _fetch_commitments(*, network: str, netuid: int) -> tuple[Any, int | str | None]:
    try:
        import bittensor as bt  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError("bittensor is not installed in this environment") from exc

    subtensor = bt.Subtensor(network=network)
    raw = subtensor.get_all_commitments(netuid=netuid)
    return raw, getattr(subtensor, "block", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump subnet commitments to JSON")
    parser.add_argument("--network", default="finney", help="Bittensor network, default: finney")
    parser.add_argument("--netuid", type=int, default=97, help="Subnet netuid, default: 97")
    parser.add_argument("--out", default="chain_commits.json", help="Output JSON file")
    parser.add_argument("--only-v4", action="store_true", help="Write only v4 reveal commits")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    raw, block = _fetch_commitments(network=args.network, netuid=args.netuid)

    commits = []
    for record in _iter_commitments(raw):
        parsed = _parse_v4(record.get("data", ""), record.get("hotkey"))
        if args.only_v4 and not parsed["is_v4"]:
            continue
        commits.append({**record, "v4": parsed})

    payload = {
        "network": args.network,
        "netuid": args.netuid,
        "block": block,
        "fetched_at": time.time(),
        "count": len(commits),
        "commits": commits,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    out_path.write_text(json.dumps(payload, indent=indent, sort_keys=args.pretty) + "\n")
    print(f"wrote {len(commits)} commits to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
