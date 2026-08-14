#!/usr/bin/env python3
"""Locally replicate the Albedo judge duel pipeline, in four modes.

Setup
-----
1. For --mode duel / regen-noise-floor, serve your checkpoint behind an
   OpenAI-compatible /chat/completions endpoint (e.g. `vllm serve <checkpoint>
   --port 8000`). The other two modes don't need a candidate endpoint at all.
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
    is pure judge-scoring noise (provider rotation, inter-judge-model
    disagreement, parse retries) rather than a real quality difference. This
    is a LOWER BOUND on real duel-to-duel noise: it doesn't capture generation
    non-determinism at all, since the input text is identical on both sides
    by construction. See --mode regen-noise-floor for that.

        python scripts/local_eval/run_duel.py --mode noise-floor --n-samples 20

--mode regen-noise-floor: generate TWO independent trajectories from the SAME
    served model on the SAME sample (same prefix, same cached checklist), then
    score run A vs run B. Unlike --mode noise-floor, the input text is NOT
    forced identical — this is meant to surface exactly the noise sources
    noise-floor can't see: your model's own decode non-determinism across two
    separate inference passes, and the environment-observation simulator (an
    LLM call itself) returning a different observation and cascading a
    divergent trajectory from that turn onward. This mirrors what actually
    happens between pass 1 and pass 2 of the live "win both evaluations"
    mechanic far more closely than noise-floor does, since a real confirmation
    duel regenerates both sides from scratch rather than rescoring stored text
    (src/albedo_eval_service/remote/worker.py's king_generator/
    challenger_generator both run every eval_run). Point --candidate-base-url
    at whatever checkpoint you want to measure (your own candidate, or the
    king if you can serve it) — it's the same flag --mode duel uses.

        python scripts/local_eval/run_duel.py --mode regen-noise-floor \\
            --candidate-base-url http://localhost:8000/v1 \\
            --candidate-model my-checkpoint --n-samples 15

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
    CHALLENGER_WIN_MARGIN,
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
from sanity_service.checks import check_collapsed, check_one, check_uniform_length  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "_king_analysis" / "artifact_GENERATED_SAMPLES_generated-samples.jsonl"
DEFAULT_SCORING_CACHE = REPO_ROOT / "_king_analysis" / "artifact_SCORING_RESULTS_scoring-results.jsonl"
DEFAULT_OUT = {
    "duel": "local_duel_results.jsonl",
    "noise-floor": "local_noise_floor_results.jsonl",
    "regen-noise-floor": "local_regen_noise_floor_results.jsonl",
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


def sanity_scan_trajectory(sample_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the same free, local heuristics generate_candidate_turns already gates on
    (repetition/vocab/encoding/empty/truncated/unclosed-think), but over every assistant
    turn, and report *what* failed instead of just silently truncating the trajectory.

    check_vocabulary's "low vocabulary diversity" reason is the one that overlaps with
    model_validation.db.hotkey_sanity_block_reason's ILIKE '%low vocab%' — a real hit on
    that specific check is flagged hotkey_risk=True since it (unlike the others here)
    permanently locks the submitting hotkey out of ever submitting again, live.
    """
    flags = []
    for turn in turns:
        if turn.get("role") != "assistant" or not turn.get("score_target"):
            continue
        gate = check_one(turn.get("content", ""))
        if not gate.passed:
            flags.append(
                {
                    "sample_id": sample_id,
                    "reason": gate.reason,
                    "hotkey_risk": "low vocab" in gate.reason.lower(),
                }
            )
    return flags


def sanity_summary(
    flags: list[dict[str, Any]], final_turn_texts: list[str]
) -> dict[str, Any]:
    collapsed = check_collapsed(final_turn_texts)
    uniform = check_uniform_length(final_turn_texts)
    return {
        "n_flagged_turns": len(flags),
        "hotkey_risk_flags": [f for f in flags if f["hotkey_risk"]],
        "flags": flags,
        "collapsed_check": {"passed": collapsed.passed, "reason": collapsed.reason},
        "uniform_length_check": {"passed": uniform.passed, "reason": uniform.reason},
    }


async def run_duel(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(args.env)
    all_records = select_samples(args)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_f = open(args.out, "w") if args.out else None
    results: list[dict[str, Any]] = []
    phase_by_sample: dict[str, str] = {}
    sanity_flags: list[dict[str, Any]] = []
    final_turn_texts: list[str] = []
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
            new_flags = sanity_scan_trajectory(sample_id, challenger_turns)
            sanity_flags.extend(new_flags)
            assistant_texts = [
                t["content"]
                for t in challenger_turns
                if t.get("role") == "assistant" and t.get("score_target")
            ]
            if assistant_texts:
                final_turn_texts.append(assistant_texts[-1])
            if new_flags:
                for f in new_flags:
                    tag = " [HOTKEY-RISK]" if f["hotkey_risk"] else ""
                    print(f"  SANITY FLAG{tag}: {f['reason']}")

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

    sanity = sanity_summary(sanity_flags, final_turn_texts)
    if sanity["n_flagged_turns"] or not sanity["collapsed_check"]["passed"] or not sanity["uniform_length_check"]["passed"]:
        print(f"\n=== SANITY FLAGS ({sanity['n_flagged_turns']} flagged turns) ===")
        for f in sanity["hotkey_risk_flags"]:
            print(f"  [HOTKEY-RISK] {f['sample_id']}: {f['reason']}")
        if not sanity["collapsed_check"]["passed"]:
            print(f"  {sanity['collapsed_check']['reason']}")
        if not sanity["uniform_length_check"]["passed"]:
            print(f"  {sanity['uniform_length_check']['reason']}")

    margin = (
        (summary["score_challenger"] - summary["score_king"])
        if summary.get("score_challenger") is not None and summary.get("score_king") is not None
        else None
    )
    if args.min_margin is not None:
        gate_passed = bool(summary.get("challenger_won")) and margin is not None and margin >= args.min_margin
        print(
            f"\n=== GATE: {'PASS' if gate_passed else 'FAIL'} "
            f"(margin={margin}, required>={args.min_margin}) ==="
        )
        if not gate_passed:
            sys.exit(1)

    return {"summary": summary, "breakdown": breakdown, "sanity": sanity, "margin": margin}


async def run_noise_floor(args: argparse.Namespace) -> dict[str, Any]:
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
    summary = {
        "n_samples": len(results),
        "n_scored": len(scored),
        "mean_abs_delta": round(mean(abs(d) for d in deltas), 4) if deltas else None,
        "max_abs_delta": round(max((abs(d) for d in deltas), default=0.0), 4)
        if deltas
        else None,
        "spurious_win_rate": round(spurious_wins / len(scored), 4) if scored else None,
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "interpretation": (
            "A real duel margin should be judged relative to mean_abs_delta above: "
            "if your challenger's margin over the king is not comfortably larger than "
            "this noise floor, don't trust a single duel run as proof of improvement. "
            "Remember the live 'win both evaluations' requirement replays the SAME "
            "samples a second time, so this noise floor is exactly the risk of losing "
            "that second roll after already winning the first."
        ),
    }
    print("\n=== NOISE FLOOR SUMMARY ===")
    print(json.dumps(summary, indent=2))
    breakdown = aggregate_category_breakdown(results, phase_by_sample=phase_by_sample)
    print(f"\n=== CATEGORY NOISE (most volatile first, n_samples={breakdown['n_samples']}) ===")
    print_category_rows(breakdown["by_category"], limit=20)
    return {"summary": summary, "breakdown": breakdown}


def _turn_divergence(turns_a: list[dict[str, Any]], turns_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two independently-generated trajectories for the same sample turn-by-turn.

    Reports where they first diverged (by index into the generated, non-prefix turns),
    which is the concrete evidence for whether a score delta came from the environment
    simulator returning a different observation mid-trajectory (cascading everything
    downstream) vs. just the final answer differing.
    """
    texts_a = [t.get("content", "") for t in turns_a]
    texts_b = [t.get("content", "") for t in turns_b]
    n = min(len(texts_a), len(texts_b))
    first_divergence = next((i for i in range(n) if texts_a[i] != texts_b[i]), None)
    identical = sum(1 for i in range(n) if texts_a[i] == texts_b[i])
    return {
        "n_turns_a": len(texts_a),
        "n_turns_b": len(texts_b),
        "first_divergence_turn": first_divergence,
        "identical_turn_fraction": round(identical / n, 3) if n else None,
    }


async def run_regen_noise_floor(args: argparse.Namespace) -> dict[str, Any]:
    """Same model, same sample, generated TWICE independently: isolates end-to-end noise
    (generation non-determinism + environment-simulator non-determinism + judge-scoring
    noise) that --mode noise-floor's identical-text comparison structurally cannot see.
    """
    settings = load_settings(args.env)
    all_records = select_samples(args)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_f = open(args.out, "w") if args.out else None
    results: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    phase_by_sample: dict[str, str] = {}
    sanity_flags: list[dict[str, Any]] = []
    try:
        for i, record in enumerate(all_records, start=1):
            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
            phase_by_sample[sample_id] = phase
            print(f"[{i}/{len(all_records)}] {sample_id} phase={phase} turns_budget={n_turns}")

            print("  generating run A...")
            turns_a = await generate_candidate_turns(
                simulator=simulator,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                candidate_base_url=args.candidate_base_url,
                candidate_model=args.candidate_model,
                candidate_api_key=args.candidate_api_key,
                n_turns=n_turns,
                max_tokens=args.max_tokens,
                eval_run_id=f"{args.eval_run_id}-a",
            )
            print("  generating run B...")
            turns_b = await generate_candidate_turns(
                simulator=simulator,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                candidate_base_url=args.candidate_base_url,
                candidate_model=args.candidate_model,
                candidate_api_key=args.candidate_api_key,
                n_turns=n_turns,
                max_tokens=args.max_tokens,
                eval_run_id=f"{args.eval_run_id}-b",
            )
            output_a = format_scored_trajectory(turns_a)
            output_b = format_scored_trajectory(turns_b)
            sanity_flags.extend(sanity_scan_trajectory(sample_id, turns_a))
            sanity_flags.extend(sanity_scan_trajectory(sample_id, turns_b))

            divergence = _turn_divergence(turns_a[len(prefix_turns) :], turns_b[len(prefix_turns) :])
            divergence["sample_id"] = sample_id
            divergences.append(divergence)
            print(
                f"  first_divergence_turn={divergence['first_divergence_turn']} "
                f"identical_turn_fraction={divergence['identical_turn_fraction']}"
            )

            # Fixed checklist for both runs, same as a real confirmation duel reuses the
            # cached checklist within its question_prep_ttl_seconds window (mirrors what
            # QuestionPrepStore does live) — this isolates generation+simulation+judge
            # noise, without also mixing in checklist-regeneration noise as a confound.
            questions, reference_made_edit, was_cached = await resolve_questions(
                question_service=question_service,
                cached_by_id=cached_by_id,
                sample_id=sample_id,
                prompt=prompt,
                prefix_turns=prefix_turns,
                n_turns=n_turns,
            )
            print("  reusing cached checklist" if was_cached else "  generated fresh checklist")

            result = await score_challenger_vs_king(
                client_llm=client,
                settings=settings,
                sample_id=sample_id,
                questions=questions,
                king_output=output_a,
                challenger_output=output_b,
                judge_models=judge_models,
                reference_made_edit=reference_made_edit,
            )
            delta = (
                round(result["challenger_score"] - result["king_score"], 4)
                if result["scored"]
                else None
            )
            print(
                f"  run_A={result['king_score']} run_B={result['challenger_score']} "
                f"delta={delta} would_spuriously_win={bool(result.get('challenger_won'))}"
            )
            results.append(result)
            if out_f:
                out_f.write(json.dumps({**result, "divergence": divergence}) + "\n")
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
    fully_diverged = sum(1 for d in divergences if d["first_divergence_turn"] == 0)
    identical_fractions = [d["identical_turn_fraction"] for d in divergences if d["identical_turn_fraction"] is not None]
    summary = {
        "n_samples": len(results),
        "n_scored": len(scored),
        "mean_abs_delta": round(mean(abs(d) for d in deltas), 4) if deltas else None,
        "max_abs_delta": round(max((abs(d) for d in deltas), default=0.0), 4) if deltas else None,
        "spurious_win_rate": round(spurious_wins / len(scored), 4) if scored else None,
        "samples_fully_diverged_from_turn_0": fully_diverged,
        "mean_identical_turn_fraction": round(mean(identical_fractions), 3) if identical_fractions else None,
        "n_hotkey_risk_sanity_flags": sum(1 for f in sanity_flags if f["hotkey_risk"]),
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "interpretation": (
            "This is the fuller noise estimate --mode noise-floor structurally can't produce: "
            "same served model, same sample, generated independently twice, so mean_abs_delta "
            "here includes generation + environment-simulator non-determinism on top of judge "
            "noise. Compare against --mode noise-floor's mean_abs_delta on the same samples — "
            "the gap between the two numbers is roughly how much of your real duel-to-duel risk "
            "comes from regeneration rather than judge scoring alone. samples_fully_diverged_from_"
            "turn_0 counts samples where run A and run B differ from the very first generated "
            "turn (usually decode-sampling variance); a lower mean_identical_turn_fraction than "
            "that first-turn-only view means the environment simulator is also driving divergence "
            "further into the trajectory, not just turn 1."
        ),
    }
    print("\n=== REGEN NOISE FLOOR SUMMARY ===")
    print(json.dumps(summary, indent=2))
    breakdown = aggregate_category_breakdown(results, phase_by_sample=phase_by_sample)
    print(f"\n=== CATEGORY NOISE (most volatile first, n_samples={breakdown['n_samples']}) ===")
    print_category_rows(breakdown["by_category"], limit=20)
    return {"summary": summary, "breakdown": breakdown, "divergences": divergences}


async def run_weakness_report(args: argparse.Namespace) -> dict[str, Any]:
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
    return breakdown


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
    p.add_argument("--mode", choices=["duel", "noise-floor", "regen-noise-floor", "weakness-report"], default="duel")
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="generated-samples.jsonl to draw tasks from")
    p.add_argument("--scoring-cache", default=str(DEFAULT_SCORING_CACHE), help="scoring-results.jsonl to reuse checklists from (skips SOTA-reference/evaluator calls); pass '' to disable")
    p.add_argument("--sample-ids", default="", help="comma-separated sample_id filter")
    p.add_argument("--n-samples", type=int, default=5, help="random sample count if --sample-ids not given")
    p.add_argument("--seed", type=int, default=0, help="seed for random sample selection")
    p.add_argument("--candidate-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint serving your checkpoint (--mode duel / regen-noise-floor only)")
    p.add_argument("--candidate-model", default="candidate", help="model name as registered on --candidate-base-url (--mode duel / regen-noise-floor only)")
    p.add_argument("--candidate-api-key", default="", help="API key for the candidate endpoint, if any")
    p.add_argument("--judge-models", default="", help="comma-separated override of judge models (default: JUDGE_MODELS from judge_core.py)")
    p.add_argument("--max-tokens", type=int, default=4096, help="max tokens per candidate turn (--mode duel / regen-noise-floor only)")
    p.add_argument("--env", default="", help="path to .env file (default: repo root .env)")
    p.add_argument("--out", default=None, help="output path (jsonl for duel/noise-floor/regen-noise-floor, json for weakness-report); default depends on --mode; pass '' to disable")
    p.add_argument("--eval-run-id", default="local-eval-run", help="tag used for Engy per-run error budget tracking")
    p.add_argument("--show-categories", action="store_true", help="print per-category/requires yes-rate breakdown per sample (--mode duel only)")
    p.add_argument("--min-margin", type=float, default=None, help="(--mode duel only) exit 1 with 'GATE: FAIL' unless challenger wins with margin >= this value. Use this to enforce a safety buffer over CHALLENGER_WIN_MARGIN before ever submitting on-chain — remember the live 'win both evaluations' rule replays the same samples a second time, so a margin that only just clears 0.025 is a coin flip on losing that replay.")
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
        "regen-noise-floor": run_regen_noise_floor,
        "weakness-report": run_weakness_report,
    }[args.mode]
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
