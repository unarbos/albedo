#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
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


def _upload_manifest_to_hippius(manifest_path: Path, key: str) -> str:
    """Upload manifest.json to Hippius S3 (public-read) and return the s3:// URI.

    Uploads the exact on-disk bytes so the object's sha256 equals the pinned manifest hash.
    Reuses the validators' ALBEDO_S3_* Hippius credentials (auto-loaded from albedo/.env).
    """
    import boto3
    from botocore.config import Config

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from model_validation import config as hv

    if not (hv.S3_BUCKET and hv.S3_ACCESS_KEY and hv.S3_SECRET_KEY):
        raise SystemExit(
            "--upload needs Hippius S3 credentials: set ALBEDO_S3_BUCKET, ALBEDO_S3_ACCESS_KEY "
            "and ALBEDO_S3_SECRET_KEY (in albedo/.env)."
        )

    body = manifest_path.read_bytes()
    client = boto3.client(
        "s3",
        endpoint_url=hv.S3_ENDPOINT,
        aws_access_key_id=hv.S3_ACCESS_KEY,
        aws_secret_access_key=hv.S3_SECRET_KEY,
        region_name="decentralized",
        config=Config(connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}),
    )
    client.put_object(
        Bucket=hv.S3_BUCKET, Key=key, Body=body,
        ContentType="application/json", ACL="public-read",
    )
    log.info("manifest sha256: %s", hashlib.sha256(body).hexdigest())
    return f"s3://{hv.S3_BUCKET}/{key}"


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
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload only manifest.json to Hippius S3 (ALBEDO_S3_* creds); does not upload the datasets.",
    )
    parser.add_argument(
        "--upload-key", default="datasets/manifest.json",
        help="Destination key in the Hippius bucket for --upload (default: datasets/manifest.json).",
    )
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

    out_path = Path(args.out) if args.out else root / "manifest.json"

    if args.skip_manifest:
        log.info("skipping manifest build (--skip-manifest)")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_manifest import _parse_weights, print_manifest_summary, write_manifest

        all_weights = _parse_weights(args.weights)
        missing = [n for n in names if n not in all_weights]
        if missing:
            raise SystemExit(f"no --weights entry for downloaded source(s): {missing}")
        weights = {n: all_weights[n] for n in names}

        out_path, manifest, digest = write_manifest(
            root, weights, out_path=out_path, max_workers=args.max_workers
        )
        print_manifest_summary(out_path, manifest, digest)

    if args.upload:
        if not out_path.exists():
            raise SystemExit(f"--upload: no manifest at {out_path} (build one first, or drop --skip-manifest).")
        log.info("uploaded manifest -> %s", _upload_manifest_to_hippius(out_path, args.upload_key))


if __name__ == "__main__":
    main()
