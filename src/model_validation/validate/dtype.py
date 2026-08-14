from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

ALLOWED_DTYPES = frozenset({"F16", "BF16"})


def _shard_dtypes(path: Path) -> set[str]:
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    return {info["dtype"] for k, info in header.items() if k != "__metadata__"}


def check_dtypes(shard_dtypes: dict[str, set[str]]) -> tuple[bool, str]:
    for name in sorted(shard_dtypes):
        bad = sorted(shard_dtypes[name] - ALLOWED_DTYPES)
        if bad:
            return False, (
                f"model weights must be 16-bit (F16/BF16); shard {name} has dtype(s): {bad}"
            )
    return True, ""


def check(model_dir: str) -> tuple[bool, str]:
    shard_dtypes: dict[str, set[str]] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        try:
            shard_dtypes[shard.name] = _shard_dtypes(shard)
        except Exception as exc:
            logger.warning(
                f"[hippius-val] could not read safetensors header of {shard.name}: {exc}"
            )
            return False, f"could not read safetensors header of {shard.name}: {exc}"
    return check_dtypes(shard_dtypes)
