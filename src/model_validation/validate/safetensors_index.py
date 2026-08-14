from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

INDEX_NAME = "model.safetensors.index.json"


def _shard_tensor_keys(path: Path) -> set[str]:
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    return {k for k in header if k != "__metadata__"}


def check(model_dir: str, files: list[str]) -> tuple[bool, str]:
    mdir = Path(model_dir)
    actual = {p.name for p in mdir.glob("*.safetensors")}
    index_path = mdir / INDEX_NAME

    if not index_path.exists():
        if actual == {"model.safetensors"}:
            return True, ""
        return False, f"sharded checkpoint missing {INDEX_NAME}"

    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("empty or non-object weight_map")
    except Exception as exc:
        logger.warning(f"[hippius-val] malformed {INDEX_NAME}: {exc}")
        return False, f"malformed {INDEX_NAME}: {exc}"

    extra_keys = sorted(set(index) - {"metadata", "weight_map"})
    if extra_keys:
        return False, (
            f"{INDEX_NAME} has disallowed top-level keys {extra_keys}; "
            "only 'metadata' and 'weight_map' are allowed"
        )

    referenced = set(weight_map.values())

    extra = sorted(actual - referenced)
    if extra:
        return False, (
            f"repo contains {len(extra)} safetensors not used by the model "
            f"(not referenced in {INDEX_NAME}): {extra[:10]}"
        )

    missing = sorted(referenced - actual)
    if missing:
        return False, f"{INDEX_NAME} references missing shard(s): {missing[:10]}"

    for shard in sorted(referenced):
        index_keys = {k for k, v in weight_map.items() if v == shard}
        try:
            header_keys = _shard_tensor_keys(mdir / shard)
        except Exception as exc:
            logger.warning(f"[hippius-val] could not read safetensors header of {shard}: {exc}")
            return False, f"could not read safetensors header of {shard}: {exc}"

        dead = sorted(header_keys - index_keys)
        if dead:
            return False, f"shard {shard} contains tensors not referenced by the index: {dead[:10]}"

        absent = sorted(index_keys - header_keys)
        if absent:
            return False, f"{INDEX_NAME} maps tensors not present in shard {shard}: {absent[:10]}"

    return True, ""
