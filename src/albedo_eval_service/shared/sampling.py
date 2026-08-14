from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any

_SHARD_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)?data/train-[A-Za-z0-9_.-]*\.parquet$")

SAMPLING_ALGO = "multi-source-manifest-sample-v1"

STEP_TRIM: list[tuple[str, int]] = [("pre_edit", 45), ("at_edit", 35), ("cold", 20)]
FAMILY_MIX: list[tuple[str, int]] = [("pr", 50), ("lm", 15), ("combine", 10), ("mechanical", 25)]
REPO_CAP = 2
MAX_PREFIX_CHARS = 54_000
BENCHMARK_LANGUAGE = "python"
NON_BENCHMARK_LANGUAGE_FRACTION = 0.30


def multi_source_manifest_sample_ids(
    manifest: dict[str, Any],
    *,
    block_hash: str,
    sample_count: int = 64,
) -> list[str]:
    if "sources" not in manifest:
        raise ValueError(
            "dataset manifest must define a 'sources' array; single-source manifests are not "
            "supported (the eval requires the combined multi-dataset manifest)"
        )
    if not block_hash:
        raise ValueError("block_hash is required for eval dataset sampling")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    sources = _normalized_sources(manifest)
    if not sources:
        return []

    rng = random.Random(str(block_hash))
    pool = _instance_pool(sources, rng)

    selected: list[str] = []
    repo_counts: dict[str, int] = defaultdict(int)
    other_language_budget = round(sample_count * NON_BENCHMARK_LANGUAGE_FRACTION)
    for phase, wants in _family_grid(sample_count).items():
        for family, want in wants.items():
            taken = 0
            for entry in pool:
                if taken == want:
                    break
                if entry["used"] or entry["family"] != family:
                    continue
                if repo_counts[entry["repo"]] >= REPO_CAP:
                    continue
                if _prefix_chars(phase, entry) > MAX_PREFIX_CHARS:
                    continue
                other_language = entry["language"] != BENCHMARK_LANGUAGE
                if other_language and other_language_budget <= 0:
                    continue
                turn_idx = _turn_idx(phase, entry, rng)
                if entry["asst"] <= turn_idx:
                    continue
                entry["used"] = True
                repo_counts[entry["repo"]] += 1
                other_language_budget -= other_language
                taken += 1
                selected.append(f"{entry['shard']}:{entry['row']}:{turn_idx}")
            if taken < want:
                raise ValueError(
                    f"infeasible stratum: phase={phase} family={family} needs {want}, "
                    f"only {taken} feasible instances remain "
                    f"(REPO_CAP={REPO_CAP}, MAX_PREFIX_CHARS={MAX_PREFIX_CHARS})"
                )

    return selected


def _prefix_chars(phase: str, entry: dict[str, Any]) -> int:
    if phase == "cold":
        return 0
    return int(entry.get("chars_at" if phase == "at_edit" else "chars_pre") or 0)


def _turn_idx(phase: str, entry: dict[str, Any], rng: random.Random) -> int:
    if phase == "cold":
        return rng.choice((1, 2))
    first_edit = entry["first_edit"]
    if first_edit <= 0:
        return 1
    return max(1, first_edit - 2) if phase == "pre_edit" else first_edit


def _apportion(spec: list[tuple[str, int]], count: int) -> dict[str, int]:
    total = sum(share for _, share in spec)
    if total <= 0:
        raise ValueError("apportion shares must sum to a positive value")
    exact = [(name, count * share / total) for name, share in spec]
    out = {name: math.floor(value) for name, value in exact}
    remainder = count - sum(out.values())
    order = sorted(enumerate(exact), key=lambda item: (-(item[1][1] % 1), item[0]))
    for _, (name, _) in order[:remainder]:
        out[name] += 1
    return out


def _family_grid(count: int) -> dict[str, dict[str, int]]:
    allocated = {name: 0 for name, _ in FAMILY_MIX}
    grid: dict[str, dict[str, int]] = {}
    cumulative = 0
    for phase, phase_want in _apportion(STEP_TRIM, count).items():
        cumulative += phase_want
        target = _apportion(FAMILY_MIX, cumulative)
        row = {name: max(0, target[name] - allocated[name]) for name, _ in FAMILY_MIX}
        grid[phase] = row
        for name, value in row.items():
            allocated[name] += value
    return grid


def _instance_pool(sources: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        for shard in source["shards"]:
            for row_idx, meta in enumerate(shard["rows_meta"]):
                if meta.get("blocked"):
                    continue
                by_instance.setdefault(str(meta["iid"]), []).append(
                    {
                        "asst": int(meta["asst"]),
                        "first_edit": int(meta.get("first_edit") or 0),
                        "family": str(meta.get("family") or "mechanical"),
                        "repo": str(meta.get("repo") or str(meta["iid"]).split(".")[0]),
                        "language": str(meta.get("language") or BENCHMARK_LANGUAGE),
                        "chars_at": int(meta.get("chars_at") or 0),
                        "chars_pre": int(meta.get("chars_pre") or 0),
                        "shard": shard["name"],
                        "row": row_idx,
                        "used": False,
                    }
                )

    pool = [rng.choice(by_instance[iid]) for iid in sorted(by_instance)]
    rng.shuffle(pool)
    return pool


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
        if not isinstance(name, str) or not name:
            raise ValueError("manifest source name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"duplicate manifest source name: {name}")
        seen_names.add(name)
        shards = _normalized_shards(source)
        source_total = sum(shard["rows"] for shard in shards)
        declared_source_total = source.get("total_rows")
        if declared_source_total is not None and declared_source_total != source_total:
            raise ValueError(f"manifest source {name} total_rows does not match shard rows")
        normalized.append({"name": name, "shards": shards})
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
