#!/usr/bin/env python3
"""
seed_genesis.py — Upload a model to Hippius as the Albedo genesis king.

Two sources supported:
  --hf-model   Pull from HuggingFace, then upload to Hippius
  --local-dir  Upload a locally fine-tuned checkpoint directly

Usage — base model (competition genesis):
  python scripts/seed_genesis.py --hf-model Qwen/Qwen3-4B --repo unarbos/albedo-qwen3-4b-genesis

Usage — custom fine-tune:
  python scripts/seed_genesis.py --local-dir checkpoints/v1/final --repo unarbos/albedo-qwen3-4b-genesis

Environment:
  HIPPIUS_HUB_TOKEN — your Hippius API token (required)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from albedo.config import EXTRA_LOCK_KEYS
from albedo.models import upload_model_folder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed_genesis")

REQUIRED_FILES = ["config.json", "tokenizer_config.json"]


def validate_local_dir(path: str) -> list[str]:
    warnings = []
    root = Path(path)
    for f in REQUIRED_FILES:
        if not (root / f).exists():
            warnings.append(f"missing {f}")
    st_files = list(root.glob("*.safetensors"))
    if not st_files:
        warnings.append("no .safetensors weight files found")
    else:
        log.info("found %d safetensors shard(s)", len(st_files))
    cfg_path = root / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        log.info("architecture: %s", cfg.get("architectures"))
        log.info("vocab_size:   %s", cfg.get("vocab_size"))
        for key in EXTRA_LOCK_KEYS:
            val = cfg.get(key)
            if val is not None:
                log.info("%s: %s  (will be locked for challengers)", key, val)
    return warnings


def download_from_hf(hf_model: str, workdir: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed: pip install huggingface_hub")
        sys.exit(1)
    log.info("downloading %s from HuggingFace → %s", hf_model, workdir)
    local_dir = snapshot_download(
        repo_id=hf_model, local_dir=workdir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                        "special_tokens*", "*.model", "*.txt"],
        max_workers=8,
    )
    log.info("downloaded to %s", local_dir)
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload genesis king to Hippius")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-model",  metavar="HF_REPO")
    source.add_argument("--local-dir", metavar="DIR")
    parser.add_argument("--repo",    required=True)
    parser.add_argument("--workdir", default="/tmp/albedo/genesis")
    parser.add_argument("--token",   default=None)
    args = parser.parse_args()

    if args.token:
        os.environ["HIPPIUS_HUB_TOKEN"] = args.token
    has_token    = bool(os.environ.get("HIPPIUS_HUB_TOKEN"))
    has_userpass = bool(os.environ.get("HIPPIUS_HUB_USERNAME") and os.environ.get("HIPPIUS_HUB_PASSWORD"))
    if not has_token and not has_userpass:
        sys.exit("Set HIPPIUS_HUB_TOKEN or HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD (or pass --token)")

    if args.local_dir:
        local_dir = args.local_dir
        if not Path(local_dir).exists():
            sys.exit(f"Local directory not found: {local_dir}")
        warnings = validate_local_dir(local_dir)
        if warnings:
            for w in warnings:
                log.warning("⚠  %s", w)
            if any("missing" in w or "no .safetensors" in w for w in warnings):
                sys.exit("Cannot upload — critical files missing")
    else:
        local_dir = download_from_hf(args.hf_model, args.workdir)
        validate_local_dir(local_dir)

    log.info("uploading to %s …", args.repo)
    ref = upload_model_folder(local_dir, repo=args.repo)
    digest = ref.digest if hasattr(ref, "digest") else str(ref)
    repo   = ref.repo   if hasattr(ref, "repo")   else args.repo
    tokenizer_repo = args.hf_model if args.hf_model else repo

    print(f"\n{'='*72}")
    print(f" repo:    {repo}")
    print(f" digest:  {digest}")
    print(f"{'='*72}")
    print(f"""
Paste into chain.toml:

  [chain]
  seed_repo = "{repo}"

  [seed]
  tokenizer_repo = "{tokenizer_repo}"
  seed_digest = "{digest}"

Then:
  1. python scripts/reset_state.py
  2. ssh templar 'git pull && pm2 restart albedo-validator'
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
