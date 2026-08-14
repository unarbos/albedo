#!/usr/bin/env python3
"""Verify how close a candidate checkpoint's weights are to a reference (e.g. the
king), using the REAL validator dedup mechanism from config_validation.fingerprint.

This is a verification tool, not a workaround: it computes the exact same
per-tensor "layer_norms_v2_with_samples" fingerprint and cosine-based
similarity() the live validator uses (src/config_validation/fingerprint/compute.py),
so you can see the actual number a submission would get *before* submitting,
and — more usefully — see WHICH tensor groups are still untouched, so you know
whether your training gave genuinely broad enough weight coverage.

There is no flag here to change what threshold a real validator applies: that
value lives on validator infrastructure (chain.toml / ALBEDO_SIM_THRESHOLD env
var), not in your submission, and this tool does not and cannot affect it.
--threshold below is purely a local display label for the report.

Usage
-----
    python scripts/local_eval/fingerprint_check.py \\
        --model-a /path/to/your/candidate/checkpoint \\
        --model-b /path/to/king/checkpoint

Each --model-* must be a directory containing *.safetensors shard(s) (single-file
or model-NNNNN-of-NNNNN.safetensors sharded layout, same as chain.toml expects).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from albedo_config.chain_spec import SIM_THRESHOLD  # noqa: E402
from config_validation.fingerprint import compute_fingerprint, similarity  # noqa: E402

# Best-effort grouping of a safetensors key into a human-readable bucket, so a
# report can point at e.g. "layer 12 expert 37" instead of a wall of raw tensor
# names. Falls back to the raw key (minus trailing numeric ids) if nothing matches.
_LAYER_RE = re.compile(r"layers?\.(\d+)")
_LAYER_STRIP_RE = re.compile(r"layers?\.\d+\.")
_EXPERT_RE = re.compile(r"experts?\.(\d+)")


def _group_for(key: str) -> str:
    layer = _LAYER_RE.search(key)
    expert = _EXPERT_RE.search(key)
    if layer and expert:
        return f"layer_{layer.group(1)}/expert_{expert.group(1)}"
    if layer:
        return f"layer_{layer.group(1)}/{_LAYER_STRIP_RE.sub('', key)}"
    return key


def per_tensor_unchanged(fp_a: dict, fp_b: dict) -> list[tuple[str, bool]]:
    """Same unchanged-detection loop as config_validation.fingerprint.compute.similarity,
    but returns the per-tensor verdict instead of just the aggregate fraction."""
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        raise ValueError(
            "layer_keys differ between the two models — they aren't the same "
            "architecture/tensor layout, so a tensor-level comparison isn't meaningful "
            "(the real validator's similarity() would just return 0.0 here)."
        )
    keys = fp_a["layer_keys"]
    sa, sb = fp_a.get("tensor_samples"), fp_b.get("tensor_samples")
    out = []
    for key, a, b in zip(keys, sa, sb):
        cos = _cosine(a, b)
        unchanged = (not any(a) and not any(b)) or cos >= (1.0 - 1e-6)
        out.append((key, unchanged))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


def group_breakdown(verdicts: list[tuple[str, bool]]) -> list[dict[str, Any]]:
    groups: dict[str, list[bool]] = {}
    for key, unchanged in verdicts:
        groups.setdefault(_group_for(key), []).append(unchanged)
    rows = [
        {
            "group": g,
            "n_tensors": len(v),
            "n_unchanged": sum(v),
            "fully_unchanged": all(v),
            "unchanged_fraction": round(sum(v) / len(v), 3),
        }
        for g, v in groups.items()
    ]
    rows.sort(key=lambda r: (-r["unchanged_fraction"], r["group"]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", required=True, help="directory with candidate checkpoint's *.safetensors")
    p.add_argument("--model-b", required=True, help="directory with reference (e.g. king) checkpoint's *.safetensors")
    p.add_argument("--threshold", type=float, default=SIM_THRESHOLD, help="display-only label; the real validator's threshold is set on validator infra, not here")
    p.add_argument("--out", default="", help="optional path to write the full JSON report")
    p.add_argument("--top", type=int, default=25, help="how many fully/most-unchanged groups to print")
    args = p.parse_args()

    print(f"computing fingerprint for {args.model_a} ...")
    fp_a = compute_fingerprint(args.model_a)
    print(f"computing fingerprint for {args.model_b} ...")
    fp_b = compute_fingerprint(args.model_b)

    sim = similarity(fp_a, fp_b)
    is_dup = sim >= args.threshold
    print("\n=== FINGERPRINT SIMILARITY (real config_validation.fingerprint.similarity) ===")
    print(f"similarity        = {sim:.6f}")
    print(f"threshold (local) = {args.threshold}  (informational only — the real validator applies its own)")
    print(f"would be flagged as near-duplicate: {is_dup}")

    verdicts = per_tensor_unchanged(fp_a, fp_b)
    n_unchanged = sum(1 for _, u in verdicts if u)
    print(f"\ntensors unchanged: {n_unchanged}/{len(verdicts)} ({n_unchanged/len(verdicts):.1%})")

    rows = group_breakdown(verdicts)
    fully_unchanged_groups = [r for r in rows if r["fully_unchanged"]]
    print(f"\ngroups fully untouched by training: {len(fully_unchanged_groups)}/{len(rows)}")
    print(f"\n=== TOP {args.top} MOST-UNCHANGED GROUPS (these are what's dragging similarity up) ===")
    for row in rows[: args.top]:
        print(
            f"  {row['group']:40s} unchanged={row['n_unchanged']:4d}/{row['n_tensors']:<4d} "
            f"({row['unchanged_fraction']:.0%})"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "similarity": sim,
                    "threshold_local_display": args.threshold,
                    "would_be_flagged": is_dup,
                    "n_tensors": len(verdicts),
                    "n_unchanged": n_unchanged,
                    "groups": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
