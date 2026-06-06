#!/usr/bin/env python3
"""miner.py — Build a challenger and submit it on-chain.

Pipeline: discover king → download weights → train_or_perturb() → upload → reveal.
The gaussian-noise stub will not beat a trained king; swap train_or_perturb() with
real SFT/RL training. See scripts/train_sft.py for a full example.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import bittensor as bt
import httpx
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from albedo.config import REPO_PATTERN, SEED_DIGEST, SEED_REPO
from albedo.models import (
    ModelRef, build_reveal_v4, config_lock_violation, materialize_model,
    upload_model_folder,
)

log = logging.getLogger("miner")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

DASHBOARD_URL = os.environ.get("ALBEDO_DASHBOARD_URL",
                               "https://us-east-1.hippius.com/albedo/dashboard.json")
NETUID      = int(os.environ.get("ALBEDO_NETUID", "0"))
NETWORK     = os.environ.get("ALBEDO_NETWORK", "finney")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "albedo")
_REPO_RE    = re.compile(REPO_PATTERN)


def _load_cfg(d: str) -> dict | None:
    p = Path(d) / "config.json"
    return json.load(open(p)) if p.exists() else None


def validate_local_config(king_dir: str, chal_dir: str) -> str | None:
    """Return rejection reason or None — mirrors server-side checks to catch mismatches before upload."""
    king = _load_cfg(king_dir)
    chal = _load_cfg(chal_dir)

    if chal is None:
        return "challenger config.json missing"

    if king:
        reason = config_lock_violation(king, chal)
        if reason:
            return reason

    if "auto_map" in chal:
        return "auto_map in config.json (not allowed)"
    if not list(Path(chal_dir).glob("*.safetensors")):
        return "no .safetensors files in challenger"
    if list(Path(chal_dir).rglob("*.py")):
        return "challenger ships .py files (not allowed)"

    return None


def train_or_perturb(king_dir: str, chal_dir: str, noise: float) -> None:
    """Copy king weights and add gaussian noise — smoke-test stub, not competitive.

    Replace with: python scripts/train_sft.py --base Qwen/Qwen3-4B --data data/traces.jsonl
    """
    if Path(chal_dir).exists():
        shutil.rmtree(chal_dir)
    shutil.copytree(king_dir, chal_dir)

    for st in sorted(Path(chal_dir).glob("*.safetensors")):
        log.info("perturbing %s", st.name)
        sd = load_file(str(st))
        save_file(
            {k: (t.float() + torch.randn_like(t.float()) * noise).to(t.dtype)
             if t.dtype in (torch.bfloat16, torch.float16, torch.float32) else t
             for k, t in sd.items()},
            str(st),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Albedo miner")
    ap.add_argument("--hotkey", default="h0")
    ap.add_argument("--noise",  type=float, default=0.001)
    ap.add_argument("--suffix", default=None, help="Repo suffix (default: hotkey name)")
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    if NETUID == 0:
        log.error("set ALBEDO_NETUID before mining")
        return 1

    suffix   = args.suffix or args.hotkey
    namespace = os.environ.get("ALBEDO_CHALLENGER_NAMESPACE", "miner")
    repo      = f"{namespace}/albedo-qwen3-4b-{suffix}"

    if not _REPO_RE.fullmatch(repo):
        log.error("repo %r does not match pattern %s", repo, REPO_PATTERN)
        return 1

    log.info("miner | hotkey=%s repo=%s", args.hotkey, repo)

    wallet    = bt.Wallet(name=WALLET_NAME, hotkey=args.hotkey)
    subtensor = bt.Subtensor(network=NETWORK)
    my_hotkey = wallet.hotkey.ss58_address

    king_repo, king_digest = SEED_REPO, SEED_DIGEST
    try:
        d           = httpx.get(DASHBOARD_URL, timeout=15).raise_for_status().json()
        king_repo   = d["king"]["model_repo"]
        king_digest = d["king"].get("king_digest") or d["king"].get("model_digest")
        log.info("king: %s@%s", king_repo, (king_digest or "")[:19])
    except Exception:
        log.warning("dashboard unreachable — using seed king")

    if not king_digest:
        log.error("no king digest available")
        return 1

    king_dir = "/tmp/albedo/miner/king"
    chal_dir = f"/tmp/albedo/miner/challenger-{suffix}"
    if Path(king_dir).exists():
        shutil.rmtree(king_dir)

    log.info("downloading king …")
    materialize_model(ModelRef(king_repo, king_digest), local_dir=king_dir, max_workers=16)

    train_or_perturb(king_dir, chal_dir, args.noise)

    err = validate_local_config(king_dir, chal_dir)
    if err:
        log.error("validation failed: %s", err)
        return 1

    log.info("uploading to %s …", repo)
    ref = upload_model_folder(chal_dir, repo=repo, revision=suffix)
    log.info("uploaded: %s", ref.immutable_ref)

    payload = build_reveal_v4(ref)
    log.info("submitting reveal …")
    result = subtensor.set_commitment(
        wallet=wallet, netuid=NETUID, data=payload
    )
    if result.success:
        log.info("reveal committed — validator picks up within ~30 s")
    else:
        log.error("commit failed: %s", result.message)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
