#!/usr/bin/env python3
"""Single pre-submission gate: fingerprint + sanity + noise-floor + duel, combined into
one PASS/FAIL decision, before you ever spend an on-chain commit and a hotkey on it.

Why this exists
----------------
Once a submission clears the live pipeline's file-allowlist/genesis-hash/dedup checks
and reaches PRE_EVAL_PASSED / EVAL_QUEUED, that hotkey is permanently recorded as
"validated" (src/model_validation/db.py's _VALIDATED_OR_BEYOND / hotkey_validated) —
there is no reset, no cooldown, no retry-with-the-same-hotkey. Any later duel loss
(COMPLETE_LOSS) is just as terminal as any earlier one. And since the live "win both
evaluations" rule (src/albedo_eval_service/control/repository.py::mark_eval_succeeded)
replays the SAME 100 samples a second time (same dataset_sample_seed = your submission's
block_hash) rather than drawing a fresh set, the only way pass 2 can go differently from
pass 1 is judge/generation noise — so a margin that only just clears CHALLENGER_WIN_MARGIN
is a real coin flip on permanently burning your hotkey for nothing.

This script runs everything you can check for free/cheaply *before* that point:

  1. fingerprint    — candidate vs king weight-tensor similarity, using the exact
                       validator mechanism (config_validation.fingerprint). Must clear
                       a LOCAL safety threshold below the real 0.95 dedup gate.
  2. noise-floor     — king vs itself, on the SAME sample set used for the duel below,
                       to measure how much score movement is pure judge/generation noise
                       on your exact task set (not a generic number).
  3. duel            — your checkpoint vs the king, on that same sample set. Also
                       collects sanity flags (repetition/vocab/encoding/collapse) on
                       every generated candidate turn as a free byproduct.
  4. combined gate   — PASS only if: fingerprint clears its safety threshold, no
                       hotkey-risk sanity flags (esp. "low vocab", which is the exact
                       substring model_validation.db.hotkey_sanity_block_reason matches
                       on), and the duel margin clears
                       CHALLENGER_WIN_MARGIN + safety_multiplier * noise mean_abs_delta.

Usage
-----
    python scripts/local_eval/preflight_gate.py \\
        --candidate-checkpoint-dir /path/to/your/checkpoint \\
        --king-checkpoint-dir /path/to/king/checkpoint \\
        --candidate-base-url http://localhost:8000/v1 --candidate-model my-checkpoint \\
        --n-samples 30

Fingerprint check is skipped (with a loud warning, not a silent pass) if you don't pass
both checkpoint dirs — everything else still runs.

Only --check-setup runs with no network/disk-heavy calls at all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duel import CHALLENGER_WIN_MARGIN, JUDGE_MODELS, load_settings  # noqa: E402
from fingerprint_check import group_breakdown, per_tensor_unchanged  # noqa: E402
from run_duel import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_SCORING_CACHE,
    run_duel,
    run_noise_floor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from albedo_config.chain_spec import SIM_THRESHOLD  # noqa: E402
from config_validation.fingerprint import compute_fingerprint, similarity  # noqa: E402


def _ns(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def check_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    if not args.candidate_checkpoint_dir or not args.king_checkpoint_dir:
        return {
            "check": "fingerprint",
            "passed": None,
            "skipped": True,
            "reason": (
                "--candidate-checkpoint-dir and --king-checkpoint-dir not both given — "
                "dedup similarity was NOT verified. This is not a pass; submitting without "
                "checking this risks a hard TERMINAL_INVALID dedup rejection that ALSO "
                "permanently locks your hotkey (model_validation.db._VALIDATED_OR_BEYOND "
                "includes PRE_EVAL_PASSED, not just eval win/loss)."
            ),
        }
    print(f"computing fingerprint for candidate: {args.candidate_checkpoint_dir} ...")
    fp_a = compute_fingerprint(args.candidate_checkpoint_dir)
    print(f"computing fingerprint for king: {args.king_checkpoint_dir} ...")
    fp_b = compute_fingerprint(args.king_checkpoint_dir)
    sim = similarity(fp_a, fp_b)
    passed = sim < args.fingerprint_safety_threshold
    result: dict[str, Any] = {
        "check": "fingerprint",
        "passed": passed,
        "skipped": False,
        "similarity": round(sim, 6),
        "real_dedup_threshold": SIM_THRESHOLD,
        "local_safety_threshold": args.fingerprint_safety_threshold,
    }
    if not passed:
        try:
            verdicts = per_tensor_unchanged(fp_a, fp_b)
            rows = group_breakdown(verdicts)
            result["top_unchanged_groups"] = rows[:10]
        except Exception as exc:  # pragma: no cover - best-effort diagnostics only
            result["diagnostics_error"] = str(exc)
    return result


def check_sanity(duel_result: dict[str, Any]) -> dict[str, Any]:
    sanity = duel_result["sanity"]
    passed = (
        not sanity["hotkey_risk_flags"]
        and sanity["collapsed_check"]["passed"]
        and sanity["uniform_length_check"]["passed"]
    )
    return {
        "check": "sanity",
        "passed": passed,
        "n_flagged_turns": sanity["n_flagged_turns"],
        "hotkey_risk_flags": sanity["hotkey_risk_flags"],
        "collapsed_check": sanity["collapsed_check"],
        "uniform_length_check": sanity["uniform_length_check"],
    }


def check_margin(
    duel_result: dict[str, Any], noise_result: dict[str, Any], safety_multiplier: float
) -> dict[str, Any]:
    margin = duel_result["margin"]
    mean_abs_delta = noise_result["summary"]["mean_abs_delta"] or 0.0
    max_abs_delta = noise_result["summary"]["max_abs_delta"] or 0.0
    required_margin = round(CHALLENGER_WIN_MARGIN + safety_multiplier * mean_abs_delta, 4)
    passed = (
        bool(duel_result["summary"].get("challenger_won"))
        and margin is not None
        and margin >= required_margin
    )
    return {
        "check": "margin",
        "passed": passed,
        "margin": margin,
        "challenger_won": duel_result["summary"].get("challenger_won"),
        "challenger_win_margin": CHALLENGER_WIN_MARGIN,
        "noise_mean_abs_delta": mean_abs_delta,
        "noise_max_abs_delta": max_abs_delta,
        "safety_multiplier": safety_multiplier,
        "required_margin": required_margin,
        "note": (
            "required_margin = CHALLENGER_WIN_MARGIN + safety_multiplier * noise mean_abs_delta. "
            "This is a planning heuristic, not a guarantee — the live 'win both evaluations' "
            "replay is a second independent noisy roll on the SAME samples, so also sanity-check "
            "margin against noise_max_abs_delta (a worse-case single-run swing) before trusting a "
            "margin that's only just above required_margin."
        ),
    }


async def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    print("=== STEP 1/3: fingerprint check ===")
    fp_check = check_fingerprint(args)
    print(json.dumps({k: v for k, v in fp_check.items() if k != "top_unchanged_groups"}, indent=2))

    shared = dict(
        samples=args.samples,
        scoring_cache=args.scoring_cache,
        sample_ids=args.sample_ids,
        n_samples=args.n_samples,
        seed=args.seed,
        judge_models=args.judge_models,
        env=args.env,
        eval_run_id=args.eval_run_id,
    )

    print("\n=== STEP 2/3: noise floor (king vs itself, same sample set as the duel below) ===")
    noise_result = await run_noise_floor(
        _ns(**shared, candidate_base_url="", candidate_model="", candidate_api_key="", out=None)
    )

    print("\n=== STEP 3/3: duel (your checkpoint vs king) ===")
    duel_result = await run_duel(
        _ns(
            **shared,
            candidate_base_url=args.candidate_base_url,
            candidate_model=args.candidate_model,
            candidate_api_key=args.candidate_api_key,
            max_tokens=args.max_tokens,
            show_categories=False,
            min_margin=None,  # gate applied below, combined with fingerprint+sanity
            out=None,
        )
    )

    sanity_check = check_sanity(duel_result)
    margin_check = check_margin(duel_result, noise_result, args.safety_multiplier)

    checks = [fp_check, sanity_check, margin_check]
    # A skipped fingerprint check is a loud warning, not a pass — it counts as failing the
    # combined gate unless the caller explicitly acknowledged the risk with --allow-unverified-fingerprint.
    fp_ok = fp_check["passed"] is True or (
        fp_check.get("skipped") and args.allow_unverified_fingerprint
    )
    gate_passed = fp_ok and sanity_check["passed"] and margin_check["passed"]

    report = {
        "gate_passed": gate_passed,
        "checks": checks,
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "judge_models": list(JUDGE_MODELS),
    }

    print("\n" + "=" * 72)
    print("SAFE TO SUBMIT" if gate_passed else "NOT SAFE TO SUBMIT")
    print("=" * 72)
    for c in checks:
        status = "SKIPPED" if c.get("skipped") else ("PASS" if c["passed"] else "FAIL")
        print(f"  [{status}] {c['check']}")
    if not gate_passed:
        print(
            "\nDo not commit this checkpoint on-chain yet — a failed/skipped check here means "
            "you have a real chance of burning your hotkey (permanently, per "
            "model_validation.db._VALIDATED_OR_BEYOND) for no gain."
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote full report to {args.out}")

    return report


def check_setup(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    print(f"repo root:                    {REPO_ROOT}")
    print(f"env file:                     {args.env or REPO_ROOT / '.env'}")
    print(f"openrouter key set:           {bool(settings.openrouter_api_key)}")
    print(f"judge models:                 {list(JUDGE_MODELS)}")
    print(f"required win margin:          {CHALLENGER_WIN_MARGIN}")
    print(f"real dedup threshold:         {SIM_THRESHOLD}")
    print(f"local fingerprint safety thr: {args.fingerprint_safety_threshold}")
    print(f"safety multiplier on noise:   {args.safety_multiplier}")
    samples_path = Path(args.samples)
    print(f"samples file:                 {samples_path} (exists={samples_path.exists()})")
    print(
        f"candidate checkpoint dir:     {args.candidate_checkpoint_dir or '(not given — fingerprint check will be skipped)'}"
    )
    print(
        f"king checkpoint dir:          {args.king_checkpoint_dir or '(not given — fingerprint check will be skipped)'}"
    )
    if not settings.openrouter_api_key:
        print("\nWARNING: ALBEDO_JUDGE_OPENROUTER_API_KEY is not set — noise-floor/duel calls will fail.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    p.add_argument("--scoring-cache", default=str(DEFAULT_SCORING_CACHE))
    p.add_argument("--sample-ids", default="", help="comma-separated sample_id filter (else random --n-samples)")
    p.add_argument("--n-samples", type=int, default=20, help="shared sample count for BOTH the noise-floor and duel measurements, so they're directly comparable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidate-base-url", default="http://localhost:8000/v1")
    p.add_argument("--candidate-model", default="candidate")
    p.add_argument("--candidate-api-key", default="")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--judge-models", default="")
    p.add_argument("--env", default="")
    p.add_argument("--eval-run-id", default="local-preflight-gate")
    p.add_argument("--candidate-checkpoint-dir", default="", help="dir with candidate's *.safetensors, for the fingerprint check")
    p.add_argument("--king-checkpoint-dir", default="", help="dir with king's *.safetensors, for the fingerprint check")
    p.add_argument("--fingerprint-safety-threshold", type=float, default=0.90, help="local safety threshold BELOW the real dedup gate (0.95) — fail the gate if similarity is above this, to keep a buffer")
    p.add_argument("--allow-unverified-fingerprint", action="store_true", help="let the combined gate pass even if the fingerprint check was skipped (no checkpoint dirs given). Off by default — an unverified dedup risk should block submission, not silently pass")
    p.add_argument("--safety-multiplier", type=float, default=2.0, help="required_margin = CHALLENGER_WIN_MARGIN + this * noise-floor mean_abs_delta")
    p.add_argument("--out", default="preflight_report.json")
    p.add_argument("--check-setup", action="store_true")
    args = p.parse_args()
    args.env = args.env or None
    args.scoring_cache = args.scoring_cache or None
    args.candidate_checkpoint_dir = args.candidate_checkpoint_dir or None
    args.king_checkpoint_dir = args.king_checkpoint_dir or None
    args.out = args.out or None
    return args


def main() -> None:
    args = parse_args()
    if args.check_setup:
        check_setup(args)
        return
    report = asyncio.run(run_gate(args))
    sys.exit(0 if report["gate_passed"] else 1)


if __name__ == "__main__":
    main()
