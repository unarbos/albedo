from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class SweZeroSampleId:
    shard_name: str
    row_idx: int
    turn_idx: int

    def as_string(self) -> str:
        return f"{self.shard_name}:{self.row_idx}:{self.turn_idx}"


def swe_zero_manifest_sample_ids(
    manifest: dict[str, Any],
    *,
    block_hash: str,
    sample_count: int = 128,
    max_turns_per_sample: int = 10,
) -> list[str]:
    """Sample SWE-ZERO trajectory coordinates from a pinned manifest.

    Implements `swe-zero-manifest-sample-v1` from Systemdesign.md. The
    manifest is expected to have the shape:
    {"shards": [{"name": "data/train-....parquet", "rows": N}], "total_rows": N}

    The backend only needs deterministic coordinate IDs. The remote eval host
    reads the parquet rows and can skip non-assistant turns while preserving the
    same `(shard, row, turn)` coordinate contract.
    """

    if not block_hash:
        raise ValueError("block_hash is required for eval dataset sampling")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if max_turns_per_sample <= 0:
        raise ValueError("max_turns_per_sample must be positive")

    shards = _normalized_shards(manifest)
    if not shards or sample_count == 0:
        return []

    rows: list[tuple[int, str, int]] = []
    for shard_idx, shard in enumerate(shards):
        for row_idx in range(shard["rows"]):
            rows.append((shard_idx, shard["name"], row_idx))

    rng = random.Random(str(block_hash))
    row_order = list(range(len(rows)))
    rng.shuffle(row_order)

    selected: list[str] = []
    seen: set[tuple[int, int, int]] = set()
    while len(selected) < sample_count:
        made_progress = False
        for row_position in row_order:
            shard_idx, shard_name, row_idx = rows[row_position]
            for turn_idx in range(max_turns_per_sample):
                key = (shard_idx, row_idx, turn_idx)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(SweZeroSampleId(shard_name, row_idx, turn_idx).as_string())
                made_progress = True
                if len(selected) >= sample_count:
                    break
            if len(selected) >= sample_count:
                break
        if not made_progress:
            break

    return selected


def _normalized_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("manifest.shards must be a list")

    normalized = []
    total_rows = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("manifest shard entries must be objects")
        name = shard.get("name")
        rows = shard.get("rows")
        if not isinstance(name, str) or not name.startswith("data/train-") or not name.endswith(".parquet"):
            raise ValueError("manifest shards must be data/train-*.parquet files")
        if not isinstance(rows, int) or rows < 0:
            raise ValueError("manifest shard rows must be non-negative integers")
        normalized.append({"name": name, "rows": rows})
        total_rows += rows

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != total_rows:
        raise ValueError("manifest total_rows does not match shard rows")

    return normalized
