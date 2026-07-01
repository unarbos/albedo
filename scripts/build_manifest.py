#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from albedo_eval_service.sampling import _SHARD_RE  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_datasets import SOURCES  # noqa: E402

DEFAULT_VERSION = "swe-zero+mini-coder-v1"


def _parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        weights[name.strip()] = float(value)
    if not weights:
        raise SystemExit("--weights must be like 'swe-zero=0.7,mini-coder=0.3'")
    return weights


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_source(name: str, weight: float, root: Path, *, max_workers: int = 8) -> dict:
    if name not in SOURCES:
        raise SystemExit(f"{name}: unknown source (not in prepare_datasets.SOURCES)")
    repo = SOURCES[name]["repo"]
    shard_glob = SOURCES[name]["shard_glob"]
    data_dir = root / name / "data"
    name_pattern = shard_glob.rsplit("/", 1)[-1]
    files = sorted(data_dir.glob(name_pattern), key=lambda p: p.name)
    if not files:
        raise SystemExit(f"{name}: no parquet shards under {data_dir} (run prepare_datasets.py first)")

    def _shard(path: Path) -> dict:
        shard_path = f"{name}/data/{path.name}"
        if not _SHARD_RE.match(shard_path):
            raise SystemExit(f"{name}: shard name {shard_path!r} is not a valid (<source>/)data/train-*.parquet")
        rows = pq.ParquetFile(path).metadata.num_rows
        return {"path": shard_path, "rows": rows, "sha256": _sha256(path)}

    # Hashing reads every shard's bytes; overlap that I/O across shards.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shards = list(pool.map(_shard, files))  # map preserves input (sorted) order
    total_rows = sum(s["rows"] for s in shards)

    return {
        "name": name,
        "repo": repo,
        "shard_glob": shard_glob,
        "weight": weight,
        "shards": shards,
        "total_rows": total_rows,
    }


def build_manifest_dict(
    root: Path, weights: dict[str, float], *, version: str = DEFAULT_VERSION, max_workers: int = 8
) -> dict:
    sources = [_build_source(name, weight, root, max_workers=max_workers) for name, weight in weights.items()]
    sources.sort(key=lambda s: s["name"])
    return {
        "version": version,
        "sources": sources,
        "total_rows": sum(s["total_rows"] for s in sources),
    }


def write_manifest(
    root: Path,
    weights: dict[str, float],
    *,
    out_path: Path | None = None,
    version: str = DEFAULT_VERSION,
    max_workers: int = 8,
) -> tuple[Path, dict, str]:
    """Build the combined manifest, write it locally, and return (path, manifest, sha256)."""
    out_path = Path(out_path) if out_path else Path(root) / "manifest.json"
    manifest = build_manifest_dict(Path(root), weights, version=version, max_workers=max_workers)
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    out_path.write_bytes(payload)
    return out_path, manifest, hashlib.sha256(payload).hexdigest()


def print_manifest_summary(out_path: Path, manifest: dict, digest: str) -> None:
    sources = manifest["sources"]
    print(f"wrote {out_path} ({manifest['total_rows']} rows across {len(sources)} sources)")
    for source in sources:
        print(f"  {source['name']}: weight={source['weight']} rows={source['total_rows']} shards={len(source['shards'])}")
    print()
    print(f"sha256: {digest}")
    print()
    print("Update these to pin the new manifest:")
    print(f"  ALBEDO_EVAL_DATASET_MANIFEST_HASH={digest}")
    print(f"  SANITY_DISPATCH_DATASET_MANIFEST_HASH={digest}")
    print("  ALBEDO_EVAL_SAMPLING_ALGO=swe-zero-multi-source-sample-v1")
    print("  src/albedo_eval_service/config.py  -> dataset_manifest_hash default")
    print("  src/sanity_service/settings.py     -> dataset_manifest_hash default")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined eval dataset manifest.")
    parser.add_argument("--dataset-root", required=True, help="Root dir holding <source>/data/*.parquet.")
    parser.add_argument("--weights", required=True, help="e.g. 'swe-zero=0.7,mini-coder=0.3'")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Manifest version label.")
    parser.add_argument("--out", default=None, help="Output path (default: <root>/manifest.json).")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel shard-hashing workers (default: 8).")
    args = parser.parse_args()

    out_path, manifest, digest = write_manifest(
        Path(args.dataset_root),
        _parse_weights(args.weights),
        out_path=Path(args.out) if args.out else None,
        version=args.version,
        max_workers=args.max_workers,
    )
    print_manifest_summary(out_path, manifest, digest)


if __name__ == "__main__":
    main()
