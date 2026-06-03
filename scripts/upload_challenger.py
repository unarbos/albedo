#!/usr/bin/env python3
"""
upload_challenger.py — Upload a fine-tuned checkpoint to Hippius and print the reveal string.

Usage:
  cd /path/to/albedo-refactor
  python scripts/upload_challenger.py \
    --model  checkpoints/v1/final \
    --repo   youruser/albedo-qwen3-4b-v1 \
    --hotkey 5GcD3Pk5XscAESmb...

Environment:
  HIPPIUS_HUB_TOKEN — your Hippius API token (required)
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from albedo.config import ALL_LOCK_KEYS, REPO_PATTERN
from albedo.models import ModelRef, build_reveal_v4, upload_model_folder


def check_arch_compat(model_dir: str, dashboard_url: str) -> None:
    """Print locked config values so the user can verify against the current king."""
    chal_cfg_path = Path(model_dir) / "config.json"
    if not chal_cfg_path.exists():
        print("⚠  No config.json found — skipping arch check")
        return

    with open(chal_cfg_path) as f:
        chal_cfg = json.load(f)

    print(f"Checking arch compatibility …")
    for key in ALL_LOCK_KEYS:
        val = chal_cfg.get(key)
        if val is not None:
            print(f"  {key}: {val}")
    print("  (verify these match the current king's config.json before submitting)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload challenger to Hippius and print reveal")
    ap.add_argument("--model",   required=True, help="Local checkpoint directory")
    ap.add_argument("--repo",    required=True, help="Hippius repo, e.g. youruser/albedo-qwen3-4b-v1")
    ap.add_argument("--hotkey",  required=True, help="Your Bittensor SS58 hotkey address")
    ap.add_argument("--token",   default=None,  help="Hippius token (or set HIPPIUS_HUB_TOKEN)")
    ap.add_argument("--dashboard",
                    default="https://us-east-1.hippius.com/albedo/dashboard.json")
    ap.add_argument("--skip-check", action="store_true",
                    help="Skip arch compatibility check")
    args = ap.parse_args()

    # Validate repo name against subnet pattern
    if not re.fullmatch(REPO_PATTERN, args.repo):
        print(f"✗  Repo name '{args.repo}' does not match required pattern: {REPO_PATTERN}")
        print(f"   Example valid name: youruser/albedo-qwen3-4b-v1")
        return 1

    if args.token:
        os.environ["HIPPIUS_HUB_TOKEN"] = args.token
    if not os.environ.get("HIPPIUS_HUB_TOKEN"):
        print("✗  Set HIPPIUS_HUB_TOKEN or pass --token")
        return 1

    if not args.skip_check:
        check_arch_compat(args.model, args.dashboard)

    print(f"\nUploading {args.model} → {args.repo} …")
    print("(This may take several minutes depending on model size)")

    ref = upload_model_folder(args.model, repo=args.repo)
    digest = ref.digest if hasattr(ref, "digest") else str(ref)

    print(f"\n✓  Upload complete")
    print(f"   Repo:   {args.repo}")
    print(f"   Digest: {digest}")

    reveal = build_reveal_v4(args.repo, digest)

    print(f"\n{'═'*64}")
    print("REVEAL STRING — submit this on-chain:")
    print(f"\n  {reveal}\n")
    print(f"{'═'*64}")
    print("""
Next: submit on-chain
  python3 -c "
import bittensor as bt
wallet    = bt.Wallet(name='default', hotkey='default')
subtensor = bt.Subtensor(network='finney')
result    = subtensor.commit(wallet, 97, '{reveal}')
print('Committed:', result)
  "
""".replace("{reveal}", reveal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
