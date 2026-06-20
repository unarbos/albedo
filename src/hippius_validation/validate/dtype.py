"""Require 16-bit model weights — every safetensors tensor must be F16 or BF16.

Rejects quantized checkpoints (8-bit / 4-bit: I8, U8, …) and full-precision F32/F64.
Reads only each shard's header, so it never loads tensor data.
"""
from __future__ import annotations

import json
from pathlib import Path

ALLOWED_DTYPES = frozenset({"F16", "BF16"})


def _shard_dtypes(path: Path) -> set[str]:
    """Distinct tensor dtypes declared in a safetensors file's header."""
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    return {info["dtype"] for k, info in header.items() if k != "__metadata__"}


def check_dtypes(shard_dtypes: dict[str, set[str]]) -> tuple[bool, str]:
    """Validate already-extracted per-shard dtypes. Lets the pre-download preflight
    reuse the same rule against headers it read remotely (no local files)."""
    for name in sorted(shard_dtypes):
        bad = sorted(shard_dtypes[name] - ALLOWED_DTYPES)
        if bad:
            return False, (f"model weights must be 16-bit (F16/BF16); shard {name} "
                           f"has dtype(s): {bad}")
    return True, ""


def check(model_dir: str) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok."""
    shard_dtypes: dict[str, set[str]] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        try:
            shard_dtypes[shard.name] = _shard_dtypes(shard)
        except Exception as exc:  # noqa: BLE001 — unreadable shard is the miner's fault
            return False, f"could not read safetensors header of {shard.name}: {exc}"
    return check_dtypes(shard_dtypes)
