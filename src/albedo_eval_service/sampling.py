from __future__ import annotations

import math
import random
import re
from typing import Any

_SHARD_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)?data/train-[A-Za-z0-9_.-]*\.parquet$")

# Prefix buckets: (Y = prefix length in user/assistant turns, count = #samples at that depth).
# Y is odd so the prefix ends on a user turn and the model generates the next assistant turn.
# Sum of counts is the sample budget (128). Assigned deepest-first with per-source 70/30.
BUCKETS: list[tuple[int, int]] = [
    (3, 8), (5, 10), (7, 10), (9, 14), (11, 14),
    (13, 14), (15, 14), (17, 14), (19, 15), (21, 15),
]


def multi_source_manifest_sample_ids(
    manifest: dict[str, Any],
    *,
    block_hash: str,
    sample_count: int = 128,
    max_turns_per_sample: int = 10,  # unused with bucket sampling; kept for call-site compatibility
) -> list[str]:
    """Deterministic bucketed sampling: unique instance_ids across sources (70/30 by weight), one
    random rollout each, prefix length from BUCKETS (deepest-first, feasibility asst >= (Y+1)//2).
    Returns ``shard:row:turn`` with ``turn = (Y-1)//2``. Requires the enriched manifest (rows_meta)."""
    if "sources" not in manifest:
        raise ValueError(
            "dataset manifest must define a 'sources' array; single-source manifests are not "
            "supported (the eval requires the combined multi-dataset manifest)"
        )
    if not block_hash:
        raise ValueError("block_hash is required for eval dataset sampling")

    bucket_total = sum(count for _, count in BUCKETS)
    if sample_count != bucket_total:
        raise ValueError(
            f"sample_count ({sample_count}) must equal sum(BUCKETS) ({bucket_total})"
        )

    sources = _normalized_sources(manifest)
    if not sources:
        return []

    rng = random.Random(str(block_hash))
    # Per source: one rollout per unique instance, sorted by depth asc so equal-depth ties keep the
    # rng-shuffled order (deterministic per block_hash, least-slack-first when we pick).
    available: dict[str, list[tuple[int, str, int]]] = {
        source["name"]: _instance_pool(source, rng) for source in sources
    }

    selected: list[str] = []
    # deepest buckets first: they are the most feasibility-constrained.
    for bucket_index in sorted(range(len(BUCKETS)), key=lambda i: BUCKETS[i][0], reverse=True):
        prefix_len, count = BUCKETS[bucket_index]
        turn_idx = (prefix_len - 1) // 2
        need_asst = (prefix_len + 1) // 2
        for name, want in _allocate_by_weight(sources, count).items():
            pool = available[name]
            feasible_from = next((i for i, item in enumerate(pool) if item[0] >= need_asst), len(pool))
            take = pool[feasible_from : feasible_from + want]
            if len(take) < want:
                raise ValueError(
                    f"infeasible bucket: prefix={prefix_len} source={name} needs {want}, "
                    f"only {len(take)} instances with >= {need_asst} assistant turns remain"
                )
            for _asst, shard_name, row_idx in take:
                selected.append(f"{shard_name}:{row_idx}:{turn_idx}")
            del pool[feasible_from : feasible_from + want]

    return selected


def _instance_pool(source: dict[str, Any], rng: random.Random) -> list[tuple[int, str, int]]:
    """One random rollout per unique instance_id (handles any per-instance count: 1, 5, 100…) ->
    (assistant_turns, shard_name, row_idx), rng-shuffled then depth-sorted (stable)."""
    by_instance: dict[str, list[tuple[int, str, int]]] = {}
    for shard in source["shards"]:
        shard_name = shard["name"]
        for row_idx, meta in enumerate(shard["rows_meta"]):
            by_instance.setdefault(meta["iid"], []).append((int(meta["asst"]), shard_name, row_idx))

    pool: list[tuple[int, str, int]] = []
    for instance_id in sorted(by_instance):  # sorted for determinism before rng draws
        # rng.choice is uniform over however many rollouts this instance has (handles 1..N).
        pool.append(rng.choice(by_instance[instance_id]))

    rng.shuffle(pool)
    pool.sort(key=lambda item: item[0])  # stable: within-depth order stays rng-shuffled
    return pool


def _allocate_by_weight(sources: list[dict[str, Any]], count: int) -> dict[str, int]:
    """Apportion ``count`` across sources by weight (largest remainder)."""
    total_weight = sum(source["weight"] for source in sources)
    if total_weight <= 0:
        raise ValueError("manifest source weights must sum to a positive value")
    exact = [(source["name"], count * source["weight"] / total_weight) for source in sources]
    allocations = {name: math.floor(value) for name, value in exact}
    remainder = count - sum(allocations.values())
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
        rows_meta = shard.get("rows_meta")
        if not isinstance(rows_meta, list) or len(rows_meta) != rows:
            raise ValueError(
                f"manifest shard {name} must carry rows_meta (one {{iid, asst}} per row); "
                "rebuild the manifest with scripts/build_manifest.py"
            )
        normalized.append({"name": name, "rows": rows, "rows_meta": rows_meta})
        total_rows += rows

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != total_rows:
        raise ValueError("manifest total_rows does not match shard rows")

    return normalized
