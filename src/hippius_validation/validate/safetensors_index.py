"""Validate model.safetensors.index.json against the safetensors actually present.

The index's ``weight_map`` is what ``transformers`` loads, so it is the authority for what
the model actually uses. A repo is the miner's fault if it ships more shards than the model
loads, references shards it doesn't ship, or a shard carries tensors the index never maps
(dead weights) — none of which a single layer of the model ever touches.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from loguru import logger

INDEX_NAME = "model.safetensors.index.json"
_SHARD_GLOB = "model-*-of-*.safetensors"


def _shard_tensor_keys(path: Path) -> set[str]:
    """Tensor keys declared in a safetensors file's header (all dtypes, header only)."""
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    return {k for k in header if k != "__metadata__"}


def check(model_dir: str, files: list[str]) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok."""
    mdir = Path(model_dir)
    actual = {p.name for p in mdir.glob("*.safetensors")}
    index_path = mdir / INDEX_NAME

    if not index_path.exists():
        # A single monolithic checkpoint needs no index; anything sharded does.
        if actual == {"model.safetensors"}:
            return True, ""
        return False, f"sharded checkpoint missing {INDEX_NAME}"

    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("empty or non-object weight_map")
    except Exception as exc:  # noqa: BLE001 — the index is the miner's artifact
        logger.warning(f"[hippius-val] malformed {INDEX_NAME}: {exc}")
        return False, f"malformed {INDEX_NAME}: {exc}"

    referenced = set(weight_map.values())

    extra = sorted(actual - referenced)
    if extra:
        return False, (f"repo contains {len(extra)} safetensors not used by the model "
                       f"(not referenced in {INDEX_NAME}): {extra[:10]}")

    missing = sorted(referenced - actual)
    if missing:
        return False, f"{INDEX_NAME} references missing shard(s): {missing[:10]}"

    # Actual usage by layers: the tensors on disk must match what the index maps per shard.
    for shard in sorted(referenced):
        index_keys = {k for k, v in weight_map.items() if v == shard}
        try:
            header_keys = _shard_tensor_keys(mdir / shard)
        except Exception as exc:  # noqa: BLE001 — unreadable shard is the miner's fault
            logger.warning(f"[hippius-val] could not read safetensors header of {shard}: {exc}")
            return False, f"could not read safetensors header of {shard}: {exc}"

        dead = sorted(header_keys - index_keys)
        if dead:
            return False, f"shard {shard} contains tensors not referenced by the index: {dead[:10]}"

        absent = sorted(index_keys - header_keys)
        if absent:
            return False, f"{INDEX_NAME} maps tensors not present in shard {shard}: {absent[:10]}"

    return True, ""
