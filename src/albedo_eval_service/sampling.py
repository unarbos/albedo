from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

_SHARD_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)?data/train-[A-Za-z0-9_.-]*\.parquet$")


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
    {"shards": [{"name" or "path": "data/train-....parquet", "rows": N}], "total_rows": N}

    The backend only needs deterministic coordinate IDs. The remote eval host
    reads the parquet rows and can skip non-assistant turns while preserving the
    same `(shard, row, turn)` coordinate contract.
    """

    _validate_sampling_args(block_hash, sample_count, max_turns_per_sample)
    shards = _normalized_shards(manifest)
    rng = random.Random(str(block_hash))
    return _select_from_shards(
        shards, rng=rng, sample_count=sample_count, max_turns_per_sample=max_turns_per_sample
    )


def multi_source_manifest_sample_ids(
    manifest: dict[str, Any],
    *,
    block_hash: str,
    sample_count: int = 128,
    max_turns_per_sample: int = 10,
) -> list[str]:
    if "sources" not in manifest:
        raise ValueError(
            "dataset manifest must define a 'sources' array; single-source manifests are "
            "not supported (the eval requires the combined multi-dataset manifest)"
        )

    _validate_sampling_args(block_hash, sample_count, max_turns_per_sample)
    sources = _normalized_sources(manifest)
    if not sources or sample_count == 0:
        return []

    allocations = _allocate_by_weight(sources, sample_count)

    selected: list[str] = []
    for source in sources:
        rng = random.Random(str(block_hash))
        selected.extend(
            _select_from_shards(
                source["shards"],
                rng=rng,
                sample_count=allocations[source["name"]],
                max_turns_per_sample=max_turns_per_sample,
            )
        )

    if len(selected) < sample_count:
        already = set(selected)
        for source in sources:
            if len(selected) >= sample_count:
                break
            rng = random.Random(str(block_hash))
            for sid in _select_from_shards(
                source["shards"],
                rng=rng,
                sample_count=sample_count,
                max_turns_per_sample=max_turns_per_sample,
            ):
                if sid in already:
                    continue
                selected.append(sid)
                already.add(sid)
                if len(selected) >= sample_count:
                    break

    return selected


def _validate_sampling_args(block_hash: str, sample_count: int, max_turns_per_sample: int) -> None:
    if not block_hash:
        raise ValueError("block_hash is required for eval dataset sampling")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if max_turns_per_sample <= 0:
        raise ValueError("max_turns_per_sample must be positive")


def _select_from_shards(
    shards: list[dict[str, Any]],
    *,
    rng: random.Random,
    sample_count: int,
    max_turns_per_sample: int,
) -> list[str]:
    if not shards or sample_count <= 0:
        return []

    rows: list[tuple[int, str, int]] = []
    for shard_idx, shard in enumerate(shards):
        for row_idx in range(shard["rows"]):
            rows.append((shard_idx, shard["name"], row_idx))

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


def _allocate_by_weight(sources: list[dict[str, Any]], sample_count: int) -> dict[str, int]:
    """Apportion ``sample_count`` across sources by weight (largest remainder)."""
    total_weight = sum(source["weight"] for source in sources)
    if total_weight <= 0:
        raise ValueError("manifest source weights must sum to a positive value")

    exact = [(source["name"], sample_count * source["weight"] / total_weight) for source in sources]
    allocations = {name: math.floor(value) for name, value in exact}
    remainder = sample_count - sum(allocations.values())
    ranked = sorted(exact, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for name, _ in ranked[:remainder]:
        allocations[name] += 1
    return allocations


def _normalized_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest.sources must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    grand_total = 0
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("manifest source entries must be objects")
        name = source.get("name")
        weight = source.get("weight")
        if not isinstance(name, str) or not name:
            raise ValueError("manifest source name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"duplicate manifest source name: {name}")
        seen_names.add(name)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError("manifest source weight must be a non-negative number")
        shards = _normalized_shards(source)
        source_total = sum(shard["rows"] for shard in shards)
        declared_source_total = source.get("total_rows")
        if declared_source_total is not None and declared_source_total != source_total:
            raise ValueError(f"manifest source {name} total_rows does not match shard rows")
        normalized.append({"name": name, "weight": float(weight), "shards": shards})
        grand_total += source_total

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != grand_total:
        raise ValueError("manifest total_rows does not match source rows")

    normalized.sort(key=lambda source: source["name"])
    return normalized


def _normalized_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("manifest.shards must be a list")

    normalized = []
    total_rows = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("manifest shard entries must be objects")
        name = shard.get("name") or shard.get("path")
        rows = shard.get("rows")
        if not isinstance(name, str) or not _SHARD_RE.match(name):
            raise ValueError("manifest shards must be (<source>/)data/train-*.parquet files")
        if not isinstance(rows, int) or rows < 0:
            raise ValueError("manifest shard rows must be non-negative integers")
        normalized.append({"name": name, "rows": rows})
        total_rows += rows

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != total_rows:
        raise ValueError("manifest total_rows does not match shard rows")

    return normalized
