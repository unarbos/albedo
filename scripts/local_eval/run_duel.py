#!/usr/bin/env python3
"""Locally replicate the Albedo judge duel pipeline, in three modes.

Setup
-----
1. For --mode duel, serve your checkpoint behind an OpenAI-compatible
   /chat/completions endpoint (e.g. `vllm serve <checkpoint> --port 8000`).
   The other two modes don't need a candidate endpoint at all.
2. Make sure the repo root .env has ALBEDO_JUDGE_OPENROUTER_API_KEY set (and
   optionally ALBEDO_JUDGE_ENGY_* if you want reference/simulation traffic
   routed through Engy). These are the same models the live validator uses:
   the evaluator/reference/judge calls hit real GLM-5.2 etc. and cost real money.

Modes
-----
--mode duel (default): generate a trajectory from YOUR checkpoint and score it
    against the stored king trajectory for the same samples.

        python scripts/local_eval/run_duel.py --mode duel \\
            --candidate-base-url http://localhost:8000/v1 \\
            --candidate-model my-checkpoint --n-samples 10

--mode noise-floor: score the king's stored trajectory against *itself*
    (identical text on both sides, no candidate needed). Any nonzero score gap
    is pure judge/measurement noise (provider rotation, inter-judge-model
    disagreement, parse retries) rather than a real quality difference — use
    this to calibrate how much of a real duel's margin you should trust before
    concluding your checkpoint actually beat the king.

        python scripts/local_eval/run_duel.py --mode noise-floor --n-samples 20

--mode weakness-report: score the king ALONE (no challenger, cheaper) across
    many samples and rank checklist categories x requires-label x phase by
    yes-rate, weakest first. This is "step 3: find the king's weaknesses" from
    the training-strategy loop, done with real judge calls instead of guessing.

        python scripts/local_eval/run_duel.py --mode weakness-report --n-samples 30

Reusing cached questions (--scoring-cache, on by default against
_king_analysis/artifact_SCORING_RESULTS_scoring-results.jsonl) skips the
SOTA-reference + evaluator calls entirely for samples that were already
scored before, which is the expensive part of the pipeline. Samples not
found in the cache fall back to generating a fresh checklist.

Only --check-setup runs with no network calls at all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duel import (  # noqa: E402
    JUDGE_MODELS,
    aggregate_category_breakdown,
    aggregate_scores,
    build_question_service,
    category_breakdown,
    generate_candidate_turns,
    load_settings,
    prefix_and_turn_count,
    prepare_questions,
    sample_phase,
    score_challenger_vs_king,
    score_single_side,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.remote.generation import format_scored_trajectory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "_king_analysis" / "artifact_GENERATED_SAMPLES_generated-samples.jsonl"
DEFAULT_SCORING_CACHE = REPO_ROOT / "_king_analysis" / "artifact_SCORING_RESULTS_scoring-results.jsonl"
DEFAULT_OUT = {
    "duel": "local_duel_results.jsonl",
    "noise-floor": "local_noise_floor_results.jsonl",
    "weakness-report": "local_weakness_report.json",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_records = load_jsonl(Path(args.samples))
    if args.sample_ids:
        wanted = set(args.sample_ids.split(","))
        return [r for r in all_records if r["sample_id"] in wanted]
    if args.n_samples:
        rng = random.Random(args.seed)
        return rng.sample(all_records, min(args.n_samples, len(all_records)))
    return all_records


def load_cache(scoring_cache: str | None) -> dict[str, dict[str, Any]]:
    cached_by_id: dict[str, dict[str, Any]] = {}
    if scoring_cache and Path(scoring_cache).exists():
        for rec in load_jsonl(Path(scoring_cache)):
            cached_by_id[rec["sample_id"]] = rec
    return cached_by_id


def judge_models_for(args: argparse.Namespace) -> list[str]:
    return (
        [m.strip() for m in args.judge_models.split(",") if m.strip()]
        if args.judge_models
        else list(JUDGE_MODELS)
    )


async def resolve_questions(
    *,
    question_service,
    cached_by_id: dict[str, dict[str, Any]],
    sample_id: str,
    prompt: str,
    prefix_turns: list[dict[str, Any]],
    n_turns: int,
) -> tuple[list[dict[str, Any]], bool | None, bool]:
    """Returns (questions, reference_made_edit, was_cached)."""
    cached = cached_by_id.get(sample_id)
    if cached and cached.get("questions"):
        questions = cached["questions"]
        reference_made_edit = (cached.get("question_source") or {}).get("reference_made_edit")
        if (cached.get("question_source") or {}).get("pruned_out") is not None:
            reference_made_edit = None
        return questions, reference_made_edit, True
    questions, source = await prepare_questions(
        question_service,
        sample_id=sample_id,
        prompt=prompt,
        prefix_turns=prefix_turns,
        n_turns=n_turns,
    )
    reference_made_edit = source.get("reference_made_edit")
    if source.get("pruned_out") is not None:
        reference_made_edit = None
    return questions, reference_made_edit, False


def print_category_rows(rows: list[dict[str, Any]], *, limit: int | None = None) -> None:
    for row in rows[:limit] if limit else rows:
        sides = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("key", "n"))
        n = f"n={row['n']:4d} " if "n" in row else ""
        print(f"  {row['key']:34s} {n}{sides}")


def print_single_sample_categories(record: dict[str, Any]) -> None:
    rows = [{"key": key, **sides} for key, sides in sorted(category_breakdown(record).items())]
    print_category_rows(rows)


async def run_duel(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    all_records = select_samples(args)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_f = open(args.out, "w") if args.out else None
    results: list[dict[str, Any]] = []
    phase_by_sample: dict[str, str] = {}
    try:
        for i, record in enumerate(all_records, start=1):
            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
            phase_by_sample[sample_id] = phase
            print(f"[{i}/{len(all_records)}] {sample_id} phase={phase} turns_budget={n_turns}")

            print("  generating candidate trajectory...")
            challenger_turns = await generate_candidate_turns(
                simulator=simulator,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                candidate_base_url=args.candidate_base_url,
                candidate_model=args.candidate_model,
                candidate_api_key=args.candidate_api_key,
                n_turns=n_turns,
                max_tokens=args.max_tokens,
                eval_run_id=args.eval_run_id,
            )
            challenger_output = format_scored_trajectory(challenger_turns)

            questions, reference_made_edit, was_cached = await resolve_questions(
                question_service=question_service,
                cached_by_id=cached_by_id,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                n_turns=n_turns,
            )
            print("  reusing cached checklist" if was_cached else "  generated fresh checklist")

            print(f"  scoring with judges: {judge_models}")
            result = await score_challenger_vs_king(
                client_llm=client,
                settings=settings,
                sample_id=sample_id,
                questions=questions,
                king_output=record["previous_king_output"],
                challenger_output=challenger_output,
                judge_models=judge_models,
                reference_made_edit=reference_made_edit,
            )
            margin = (
                round(result["challenger_score"] - result["king_score"], 4)
                if result["scored"]
                else None
            )
            print(
                f"  king={result['king_score']} challenger={result['challenger_score']} "
                f"margin={margin} won={result['challenger_won']}"
            )
            if args.show_categories and result["scored"]:
                print_single_sample_categories(result)

            results.append(result)
            if out_f:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()
        await client.aclose()
        if repo_context is not None:
            await repo_context.aclose()

    summary = aggregate_scores(results, min_valid_fraction=settings.min_valid_fraction)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    breakdown = aggregate_category_breakdown(results, phase_by_sample=phase_by_sample)
    print(f"\n=== CATEGORY BREAKDOWN (weakest first, n_samples={breakdown['n_samples']}) ===")
    print_category_rows(breakdown["by_category"])


async def run_noise_floor(args: argparse.Namespace) -> None:
    """King vs itself: isolates pure judge-scoring noise from the real win-margin signal."""
    settings = load_settings(args.env)
    all_records = select_samples(args)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_f = open(args.out, "w") if args.out else None
    results: list[dict[str, Any]] = []
    phase_by_sample: dict[str, str] = {}
    try:
        for i, record in enumerate(all_records, start=1):
            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
            phase_by_sample[sample_id] = phase
            print(f"[{i}/{len(all_records)}] {sample_id} phase={phase}")

            questions, reference_made_edit, was_cached = await resolve_questions(
                question_service=question_service,
                cached_by_id=cached_by_id,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                n_turns=n_turns,
            )
            print("  reusing cached checklist" if was_cached else "  generated fresh checklist")

            king_output = record["previous_king_output"]
            result = await score_challenger_vs_king(
                client_llm=client,
                settings=settings,
                sample_id=sample_id,
                questions=questions,
                king_output=king_output,
                challenger_output=king_output,  # identical text on both sides
                judge_models=judge_models,
                reference_made_edit=reference_made_edit,
            )
            delta = (
                round(result["challenger_score"] - result["king_score"], 4)
                if result["scored"]
                else None
            )
            print(
                f"  king={result['king_score']} self_rescored={result['challenger_score']} "
                f"delta={delta} would_spuriously_win={bool(result.get('challenger_won'))}"
            )
            results.append(result)
            if out_f:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()
        await client.aclose()
        if repo_context is not None:
            await repo_context.aclose()

    scored = [r for r in results if r["scored"]]
    deltas = [r["challenger_score"] - r["king_score"] for r in scored]
    spurious_wins = sum(1 for r in scored if r.get("challenger_won"))
    print("\n=== NOISE FLOOR SUMMARY ===")
    print(
        json.dumps(
            {
                "n_samples": len(results),
                "n_scored": len(scored),
                "mean_abs_delta": round(mean(abs(d) for d in deltas), 4) if deltas else None,
                "max_abs_delta": round(max((abs(d) for d in deltas), default=0.0), 4)
                if deltas
                else None,
                "spurious_win_rate": round(spurious_wins / len(scored), 4) if scored else None,
                "required_win_margin": 0.03,
                "interpretation": (
                    "A real duel margin should be judged relative to mean_abs_delta above: "
                    "if your challenger's margin over the king is not comfortably larger than "
                    "this noise floor, don't trust a single duel run as proof of improvement."
                ),
            },
            indent=2,
        )
    )
    breakdown = aggregate_category_breakdown(results, phase_by_sample=phase_by_sample)
    print(f"\n=== CATEGORY NOISE (most volatile first, n_samples={breakdown['n_samples']}) ===")
    print_category_rows(breakdown["by_category"], limit=20)


async def run_weakness_report(args: argparse.Namespace) -> None:
    """Score the king alone across many samples; rank checklist categories weakest-first."""
    settings = load_settings(args.env)
    all_records = select_samples(args)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    results: list[dict[str, Any]] = []
    phase_by_sample: dict[str, str] = {}
    try:
        for i, record in enumerate(all_records, start=1):
            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
            phase_by_sample[sample_id] = phase
            print(f"[{i}/{len(all_records)}] {sample_id} phase={phase}")

            questions, reference_made_edit, was_cached = await resolve_questions(
                question_service=question_service,
                cached_by_id=cached_by_id,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                n_turns=n_turns,
            )
            print("  reusing cached checklist" if was_cached else "  generated fresh checklist")

            result = await score_single_side(
                client_llm=client,
                settings=settings,
                sample_id=sample_id,
                questions=questions,
                output_text=record["previous_king_output"],
                side="previous_king",
                judge_models=judge_models,
                reference_made_edit=reference_made_edit,
            )
            print(f"  king_score={result['score']}")
            results.append(result)
    finally:
        await client.aclose()
        if repo_context is not None:
            await repo_context.aclose()

    breakdown = aggregate_category_breakdown(results, phase_by_sample=phase_by_sample)
    print(f"\n=== KING WEAKNESS REPORT (weakest categories first, n_samples={breakdown['n_samples']}) ===")
    print_category_rows(breakdown["by_category"])
    if breakdown["by_category_and_phase"]:
        print("\n=== BY CATEGORY x PHASE (weakest first) ===")
        print_category_rows(breakdown["by_category_and_phase"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(breakdown, f, indent=2)
        print(f"\nWrote weakness report to {args.out}")


def check_setup(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    print(f"repo root:          {REPO_ROOT}")
    print(f"mode:               {args.mode}")
    print(f"env file:           {args.env or REPO_ROOT / '.env'}")
    print(f"openrouter key set: {bool(settings.openrouter_api_key)}")
    print(f"engy key set:       {bool(settings.engy_api_key)}")
    print(f"engy base url:      {settings.engy_base_url}")
    print(f"evaluator model:    {settings.evaluator_model}")
    print(f"sota models:        {settings.sota_models}")
    print(f"simulation model:   {settings.simulation_model}")
    print(f"judge models:       {list(JUDGE_MODELS)}")
    samples_path = Path(args.samples)
    print(f"samples file:       {samples_path} (exists={samples_path.exists()})")
    cache_path = Path(args.scoring_cache) if args.scoring_cache else None
    print(f"scoring cache:      {cache_path} (exists={cache_path.exists() if cache_path else False})")
    if not settings.openrouter_api_key:
        print("\nWARNING: ALBEDO_JUDGE_OPENROUTER_API_KEY is not set — scoring calls will fail.")
    print("\nImports OK, setup looks structurally sound." if settings.openrouter_api_key else "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["duel", "noise-floor", "weakness-report"], default="duel")
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="generated-samples.jsonl to draw tasks from")
    p.add_argument("--scoring-cache", default=str(DEFAULT_SCORING_CACHE), help="scoring-results.jsonl to reuse checklists from (skips SOTA-reference/evaluator calls); pass '' to disable")
    p.add_argument("--sample-ids", default="", help="comma-separated sample_id filter")
    p.add_argument("--n-samples", type=int, default=5, help="random sample count if --sample-ids not given")
    p.add_argument("--seed", type=int, default=0, help="seed for random sample selection")
    p.add_argument("--candidate-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint serving your checkpoint (--mode duel only)")
    p.add_argument("--candidate-model", default="candidate", help="model name as registered on --candidate-base-url (--mode duel only)")
    p.add_argument("--candidate-api-key", default="", help="API key for the candidate endpoint, if any")
    p.add_argument("--judge-models", default="", help="comma-separated override of judge models (default: JUDGE_MODELS from judge_core.py)")
    p.add_argument("--max-tokens", type=int, default=4096, help="max tokens per candidate turn (--mode duel only)")
    p.add_argument("--env", default="", help="path to .env file (default: repo root .env)")
    p.add_argument("--out", default=None, help="output path (jsonl for duel/noise-floor, json for weakness-report); default depends on --mode; pass '' to disable")
    p.add_argument("--eval-run-id", default="local-eval-run", help="tag used for Engy per-run error budget tracking")
    p.add_argument("--show-categories", action="store_true", help="print per-category/requires yes-rate breakdown per sample (--mode duel only)")
    p.add_argument("--check-setup", action="store_true", help="print config + import sanity check and exit, no network calls")
    args = p.parse_args()
    args.env = args.env or None
    args.scoring_cache = args.scoring_cache or None
    if args.out is None:
        args.out = DEFAULT_OUT[args.mode]
    args.out = args.out or None
    return args


def main() -> None:
    args = parse_args()
    if args.check_setup:
        check_setup(args)
        return
    runner = {
        "duel": run_duel,
        "noise-floor": run_noise_floor,
        "weakness-report": run_weakness_report,
    }[args.mode]
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
