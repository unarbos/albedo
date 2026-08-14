#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from albedo_eval_service.shared.sampling import (
    BENCHMARK_LANGUAGE,
    FAMILY_MIX,
    MAX_PREFIX_CHARS,
    NON_BENCHMARK_LANGUAGE_FRACTION,
    REPO_CAP,
    STEP_TRIM,
)


def _source_stats(source: dict) -> tuple[dict, set[str]]:
    by_instance: dict[str, dict] = {}
    blocked_rows = 0
    for shard in source.get("shards", []):
        for meta in shard.get("rows_meta", []):
            if meta.get("blocked"):
                blocked_rows += 1
                continue
            entry = by_instance.setdefault(
                str(meta.get("iid") or ""),
                {
                    "family": str(meta.get("family") or "mechanical"),
                    "language": str(meta.get("language") or "unknown"),
                    "has_edit": False,
                },
            )
            entry["has_edit"] = entry["has_edit"] or int(meta.get("first_edit") or 0) > 0
    stats = {
        "instances": len(by_instance),
        "instances_with_edit": sum(1 for e in by_instance.values() if e["has_edit"]),
        "blocked_rows": blocked_rows,
        "families": dict(Counter(e["family"] for e in by_instance.values())),
        "languages": dict(Counter(e["language"] for e in by_instance.values())),
    }
    return stats, set(by_instance)


def build_meta_dict(manifest: dict) -> dict:
    pool: set[str] = set()
    sources = []
    for source in manifest.get("sources", []):
        stats, instances = _source_stats(source)
        pool |= instances
        sources.append(
            {
                **{k: v for k, v in source.items() if k != "shards"},
                "shards": [
                    {k: v for k, v in shard.items() if k != "rows_meta"}
                    for shard in source.get("shards", [])
                ],
                "stats": stats,
            }
        )
    return {
        **{k: v for k, v in manifest.items() if k != "sources"},
        "sources": sources,
        "unique_instances": len(pool),
        "sampling": {
            "phases": [[name, share] for name, share in STEP_TRIM],
            "families": [[name, share] for name, share in FAMILY_MIX],
            "repo_cap": REPO_CAP,
            "max_prefix_chars": MAX_PREFIX_CHARS,
            "benchmark_language": BENCHMARK_LANGUAGE,
            "non_benchmark_language_fraction": NON_BENCHMARK_LANGUAGE_FRACTION,
        },
    }


def write_meta(manifest: dict, out_path: Path) -> Path:
    payload = json.dumps(build_meta_dict(manifest), sort_keys=True, indent=2).encode("utf-8")
    out_path.write_bytes(payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive manifest.meta.json from a full manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the full manifest.json.")
    parser.add_argument(
        "--out", default=None, help="Output path (default: <manifest dir>/manifest.meta.json)."
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("rb") as f:
        manifest = json.load(f)
    out_path = write_meta(
        manifest, Path(args.out) if args.out else manifest_path.with_name("manifest.meta.json")
    )
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
