#!/usr/bin/env python3
"""Prefetch the full SWE-ZERO parquet corpus for the eval server.

Downloads every shard matching `chain.toml [dataset].shard_glob` from
`[dataset].repo` into a local directory (default `/var/albedo/dataset/`),
builds `manifest.json`, and prints the manifest sha256 to paste into
`chain.toml [dataset].manifest_sha256`.

Usage:
    source .venv/bin/activate
    python scripts/prefetch_dataset.py [--out /var/albedo/dataset]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chain_config
from huggingface_hub import snapshot_download

import trajectory_sampler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prefetch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/var/albedo/dataset",
                        help="Destination directory")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only rebuild manifest from shards already on disk")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = chain_config.DATASET_SHARD_GLOB

    if not args.skip_download:
        log.info(
            "downloading %s :: %s to %s",
            chain_config.DATASET_REPO,
            pattern,
            out_dir,
        )
        snapshot_download(
            repo_id=chain_config.DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[pattern],
            local_dir=str(out_dir),
        )
        log.info("download complete")

    manifest_path = trajectory_sampler.build_manifest(out_dir, shard_glob=pattern)
    catalog = trajectory_sampler.load_catalog(out_dir)

    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    sha = h.hexdigest()

    print()
    print("=" * 72)
    print(f" dataset_dir:  {out_dir}")
    print(f" shard_glob:   {pattern}")
    print(f" shards:       {len(catalog.shards)}")
    print(f" total_rows:   {catalog.total_rows}")
    print(f" manifest:     {manifest_path}")
    print(f" manifest_sha256: {sha}")
    print("=" * 72)
    print()
    print("Paste into chain.toml:")
    print()
    print("  [dataset]")
    print(f"  manifest_sha256 = \"{sha}\"")
    print()
    print("Set on the eval server:")
    print(f"  export ALBEDO_DATASET_DIR={out_dir}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
