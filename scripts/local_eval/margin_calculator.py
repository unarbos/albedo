#!/usr/bin/env python3
"""Project how much aggregate duel margin fixing specific weak buckets could buy you,
BEFORE spending training compute on them.

Grounded in the real scoring code, not docs/SCORING.md's "weighted yes-rate" section:
`judge_yes_rate` -> `response_score` -> `aggregate_scores` (src/albedo_eval_service/judge_core.py)
is a plain, unweighted mean of 0/1 answers (mean-of-means across judges, then across samples).
There is no requires-label weight (action/read/neutral) and no size-floor multiplier anywhere in
that call chain in this checkout. The only thing that gives a bucket leverage over the aggregate
score is how many individual answered questions belong to it — i.e. the "n" field that
aggregate_category_breakdown() (scripts/local_eval/duel.py) already reports.

    share(bucket)          = n(bucket) / sum(n(all buckets))
    projected_gain(bucket) = share(bucket) * (target_rate(bucket) - current_rate(bucket))
    projected_total_margin = sum(projected_gain(bucket) for bucket in buckets_you_plan_to_fix)

Input: the JSON written by
    python scripts/local_eval/run_duel.py --mode weakness-report --out report.json
(shape: {"n_samples": int, "by_category": [{"key", "n", "previous_king"}, ...], "by_category_and_phase": [...]})

Usage
-----
Rank buckets and see the projected margin if every bucket below --default-target were pushed up to it:

    python scripts/local_eval/margin_calculator.py --report local_weakness_report.json

Only consider a specific set of buckets you actually plan to train on, each with its own target:

    python scripts/local_eval/margin_calculator.py --report local_weakness_report.json \\
        --bucket "workflow/action=0.92" --bucket "grounding/action=0.9"

Use the phase-split breakdown instead of the plain category breakdown:

    python scripts/local_eval/margin_calculator.py --report local_weakness_report.json --phase
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from albedo_eval_service.judge_core import CHALLENGER_WIN_MARGIN  # noqa: E402


def load_rows(report_path: str, *, phase: bool) -> list[dict[str, Any]]:
    with open(report_path) as f:
        report = json.load(f)
    rows = report["by_category_and_phase"] if phase else report["by_category"]
    if not rows:
        raise ValueError(
            f"no rows in {'by_category_and_phase' if phase else 'by_category'} — "
            f"{'the report was generated without --phase support' if phase else 'the report file looks empty'}"
        )
    return rows


def current_rate(row: dict[str, Any]) -> float | None:
    # weakness-report rows are single-sided ("previous_king"); duel/noise-floor rows may carry
    # "previous_king" + "challenger" — prefer the king side since that's the pre-training baseline.
    for side in ("previous_king", "challenger"):
        if row.get(side) is not None:
            return row[side]
    return None


def parse_bucket_targets(specs: list[str]) -> dict[str, float]:
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--bucket must be 'key=target_rate', got {spec!r}")
        key, target = spec.rsplit("=", 1)
        out[key.strip()] = float(target)
    return out


def project(
    rows: list[dict[str, Any]],
    *,
    bucket_targets: dict[str, float],
    default_target: float,
    only_selected: bool,
) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    if total_n == 0:
        raise ValueError("all buckets have n=0 — nothing to project from")

    per_bucket = []
    for row in rows:
        key = row["key"]
        rate = current_rate(row)
        if rate is None:
            continue
        share = row["n"] / total_n
        if key in bucket_targets:
            target = bucket_targets[key]
            selected = True
        elif not only_selected:
            target = default_target
            selected = True
        else:
            target = rate
            selected = False
        gain = share * max(0.0, target - rate) if selected else 0.0
        per_bucket.append(
            {
                "key": key,
                "n": row["n"],
                "share_of_vote_mass": round(share, 4),
                "current_rate": rate,
                "target_rate": round(target, 4) if selected else None,
                "selected": selected,
                "projected_gain": round(gain, 5),
            }
        )

    per_bucket.sort(key=lambda r: -r["projected_gain"])
    total_gain = round(sum(r["projected_gain"] for r in per_bucket), 5)
    return {
        "total_n": total_n,
        "projected_total_margin": total_gain,
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "clears_margin": total_gain >= CHALLENGER_WIN_MARGIN,
        "buckets": per_bucket,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", required=True, help="weakness-report JSON from run_duel.py --mode weakness-report --out ...")
    p.add_argument("--phase", action="store_true", help="use by_category_and_phase rows instead of by_category")
    p.add_argument(
        "--bucket",
        action="append",
        default=[],
        metavar="KEY=TARGET",
        help="e.g. 'workflow/action=0.92' — set a specific realistic target for a bucket you plan to train. Repeatable.",
    )
    p.add_argument(
        "--default-target",
        type=float,
        default=0.9,
        help="target rate applied to every bucket NOT given an explicit --bucket override (ignored if --only-selected)",
    )
    p.add_argument(
        "--only-selected",
        action="store_true",
        help="only project gains for buckets explicitly named via --bucket, ignoring all others (use this once you've decided your training scope)",
    )
    p.add_argument("--top", type=int, default=20, help="how many buckets to print")
    p.add_argument("--out", default="", help="optional path to write the full JSON projection")
    args = p.parse_args()

    rows = load_rows(args.report, phase=args.phase)
    bucket_targets = parse_bucket_targets(args.bucket)
    result = project(
        rows,
        bucket_targets=bucket_targets,
        default_target=args.default_target,
        only_selected=args.only_selected,
    )

    print(f"total answered questions across all buckets: {result['total_n']}")
    print(f"required win margin: {result['required_win_margin']}")
    print(f"\n=== TOP {args.top} BUCKETS BY PROJECTED GAIN ===")
    for row in result["buckets"][: args.top]:
        tag = "" if row["selected"] else "  (not targeted)"
        target_str = f"-> {row['target_rate']}" if row["target_rate"] is not None else ""
        print(
            f"  {row['key']:34s} n={row['n']:4d} share={row['share_of_vote_mass']:.1%} "
            f"rate={row['current_rate']:.3f} {target_str:12s} gain=+{row['projected_gain']:.4f}{tag}"
        )

    print(f"\nprojected total margin: {result['projected_total_margin']:+.4f}")
    print(f"clears {CHALLENGER_WIN_MARGIN} requirement: {result['clears_margin']}")
    print(
        "\nNOTE: this assumes zero regression in buckets you don't touch, and that every fixed "
        "bucket generalizes to its full target rate across the real duel's 100 fresh samples — not "
        "just your curated training pairs. Treat this as a planning ceiling, then confirm for real "
        "with run_duel.py --mode duel once a checkpoint exists, and size your safety buffer against "
        "--mode noise-floor's mean_abs_delta."
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote full projection to {args.out}")


if __name__ == "__main__":
    main()
