#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("prepare_datasets")


SOURCES: dict[str, dict[str, str]] = {
    "swe-zero": {"repo": "AlienKevin/SWE-ZERO-12M-trajectories", "shard_glob": "data/train-*.parquet"},
    "mini-coder": {"repo": "ricdomolm/mini-coder-trajs-400k", "shard_glob": "data/train-*.parquet"},
}


def _enable_fast_transfer() -> None:

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _expected_parquet_shards(repo_id: str, shard_glob: str) -> set[str]:
    """Repo-relative paths of every shard matching shard_glob (one metadata call, no
    file content) so we can fetch only what is missing."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return {f for f in files if fnmatch.fnmatch(f, shard_glob)}


def _local_parquet_shards(dest: Path, shard_glob: str) -> set[str]:
    """Repo-relative paths of shards already present under dest/ (matched by shard_glob)."""
    subdir, _, name_pat = shard_glob.rpartition("/")
    data_dir = dest / subdir if subdir else dest
    if not data_dir.is_dir():
        return set()
    prefix = f"{subdir}/" if subdir else ""
    return {f"{prefix}{p.name}" for p in data_dir.glob(name_pat)}


def download_source(name: str, repo_id: str, shard_glob: str, root: Path, *, force: bool, max_workers: int) -> Path:
    from huggingface_hub import hf_hub_download

    dest = root / name

    expected = _expected_parquet_shards(repo_id, shard_glob)
    if not expected:
        raise RuntimeError(f"{name}: no shards in repo {repo_id} matching {shard_glob!r}")
    present = _local_parquet_shards(dest, shard_glob)
    to_fetch = sorted(expected) if force else sorted(expected - present)

    if not to_fetch:
        log.info("%s: complete (%d/%d shards present) — skipping", name, len(present), len(expected))
        return dest
    log.info(
        "%s: %d/%d present, downloading %d missing -> %s (%d parallel workers)",
        name, len(present), len(expected), len(to_fetch), dest, max_workers,
    )

    def _one(rel: str) -> None:
        # Resumes partial files and skips already-complete ones; writes to dest/<rel>.
        # HF_TOKEN (if set) is picked up from the env automatically.
        hf_hub_download(
            repo_id, rel, repo_type="dataset", local_dir=str(dest),
            force_download=force, token=os.environ.get("HF_TOKEN"),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, rel) for rel in to_fetch]
        for fut in as_completed(futures):
            fut.result()  # re-raise the first download failure
            done += 1
            if done % 50 == 0 or done == len(to_fetch):
                log.info("%s: %d/%d downloaded", name, done, len(to_fetch))

    still_missing = expected - _local_parquet_shards(dest, shard_glob)
    if still_missing:
        raise RuntimeError(
            f"{name}: {len(still_missing)} shard(s) still missing after download, "
            f"e.g. {sorted(still_missing)[:3]}"
        )
    log.info("%s: done (%d shards)", name, len(expected))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download eval datasets from HuggingFace and build the combined manifest locally."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dir to download <source>/data/*.parquet into and write manifest.json (local only).",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help=f"Comma-separated source names to fetch (default: all of {','.join(SOURCES)}).",
    )
    parser.add_argument(
        "--weights",
        default="swe-zero=0.7,mini-coder=0.3",
        help="Per-source manifest weights (default: swe-zero=0.7,mini-coder=0.3).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if shards already exist.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent files downloaded in parallel (default: 16). Raise for many small shards.",
    )
    parser.add_argument("--skip-manifest", action="store_true", help="Only download; do not build manifest.json.")
    parser.add_argument("--out", default=None, help="Manifest output path (default: <dataset-root>/manifest.json).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _enable_fast_transfer()

    root = Path(args.dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; known: {list(SOURCES)}")

    for name in names:
        meta = SOURCES[name]
        download_source(
            name, meta["repo"], meta["shard_glob"], root, force=args.force, max_workers=args.max_workers
        )

    if args.skip_manifest:
        log.info("skipping manifest build (--skip-manifest)")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_manifest import _parse_weights, print_manifest_summary, write_manifest

    all_weights = _parse_weights(args.weights)
    missing = [n for n in names if n not in all_weights]
    if missing:
        raise SystemExit(f"no --weights entry for downloaded source(s): {missing}")
    weights = {n: all_weights[n] for n in names}

    out_path, manifest, digest = write_manifest(
        root, weights, out_path=Path(args.out) if args.out else None, max_workers=args.max_workers
    )
    print_manifest_summary(out_path, manifest, digest)


if __name__ == "__main__":
    main()
