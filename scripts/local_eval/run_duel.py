#!/usr/bin/env python3
"""Locally duel your fine-tuned checkpoint against a stored king trajectory,
using the real Albedo judge pipeline (judge_core.py / judge_api.py) in-process.

Setup
-----
1. Serve your checkpoint behind an OpenAI-compatible /chat/completions endpoint
   (e.g. `vllm serve <checkpoint> --port 8000`).
2. Make sure the repo root .env has ALBEDO_JUDGE_OPENROUTER_API_KEY set (and
   optionally ALBEDO_JUDGE_ENGY_* if you want reference/simulation traffic
   routed through Engy). These are the same models the live validator uses:
   the evaluator/reference/judge calls hit real GLM-5.2 etc. and cost real money.
3. Run:

   python scripts/local_eval/run_duel.py \\
       --candidate-base-url http://localhost:8000/v1 \\
       --candidate-model my-checkpoint \\
       --n-samples 10

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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duel import (  # noqa: E402
    JUDGE_MODELS,
    aggregate_scores,
    build_question_service,
    generate_candidate_turns,
    load_settings,
    prefix_and_turn_count,
    prepare_questions,
    sample_phase,
    score_challenger_vs_king,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.remote_generation import format_scored_trajectory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "_king_analysis" / "artifact_GENERATED_SAMPLES_generated-samples.jsonl"
DEFAULT_SCORING_CACHE = REPO_ROOT / "_king_analysis" / "artifact_SCORING_RESULTS_scoring-results.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def category_breakdown(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per-category, per-requires-label yes-rate for king vs challenger on one scored sample.

    Purely additive reporting on top of the real judge_results — does not affect scoring.
    """
    questions_by_id = {q["id"]: q for q in record.get("questions", [])}
    by_key: dict[str, dict[str, list[float]]] = {}
    for rec in record.get("judge_results", []):
        side = rec["side"]
        for qid, ans in (rec.get("answers") or {}).items():
            q = questions_by_id.get(qid)
            if q is None or ans is None:
                continue
            key = f"{q.get('category', 'other')}/{q.get('requires', 'neutral')}"
            by_key.setdefault(key, {"previous_king": [], "challenger": []})
            by_key[key][side if side in ("previous_king", "challenger") else "challenger"].append(
                float(ans)
            )
    out: dict[str, dict[str, float]] = {}
    for key, sides in by_key.items():
        out[key] = {
            side: round(sum(v) / len(v), 3) if v else None for side, v in sides.items()
        }
    return out


async def run(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    all_records = load_jsonl(Path(args.samples))
    if args.sample_ids:
        wanted = set(args.sample_ids.split(","))
        all_records = [r for r in all_records if r["sample_id"] in wanted]
    elif args.n_samples:
        rng = random.Random(args.seed)
        all_records = rng.sample(all_records, min(args.n_samples, len(all_records)))

    cached_by_id: dict[str, dict[str, Any]] = {}
    if args.scoring_cache and Path(args.scoring_cache).exists():
        for rec in load_jsonl(Path(args.scoring_cache)):
            cached_by_id[rec["sample_id"]] = rec

    judge_models = (
        [m.strip() for m in args.judge_models.split(",") if m.strip()]
        if args.judge_models
        else list(JUDGE_MODELS)
    )

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_path = Path(args.out) if args.out else None
    out_f = open(out_path, "w") if out_path else None
    results: list[dict[str, Any]] = []
    try:
        for i, record in enumerate(all_records, start=1):
            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
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

            cached = cached_by_id.get(sample_id)
            if cached and cached.get("questions"):
                print("  reusing cached checklist")
                questions = cached["questions"]
                reference_made_edit = (cached.get("question_source") or {}).get(
                    "reference_made_edit"
                )
                if (cached.get("question_source") or {}).get("pruned_out") is not None:
                    reference_made_edit = None
            else:
                print("  no cached checklist, generating fresh SOTA-anchored checklist...")
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
                for key, sides in sorted(category_breakdown(result).items()):
                    print(f"    {key:30s} king={sides.get('previous_king')} chal={sides.get('challenger')}")

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


def check_setup(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    print(f"repo root:          {REPO_ROOT}")
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
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="generated-samples.jsonl to draw tasks from")
    p.add_argument("--scoring-cache", default=str(DEFAULT_SCORING_CACHE), help="scoring-results.jsonl to reuse checklists from (skips SOTA-reference/evaluator calls); pass '' to disable")
    p.add_argument("--sample-ids", default="", help="comma-separated sample_id filter")
    p.add_argument("--n-samples", type=int, default=5, help="random sample count if --sample-ids not given")
    p.add_argument("--seed", type=int, default=0, help="seed for random sample selection")
    p.add_argument("--candidate-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint serving your checkpoint")
    p.add_argument("--candidate-model", default="candidate", help="model name as registered on --candidate-base-url")
    p.add_argument("--candidate-api-key", default="", help="API key for the candidate endpoint, if any")
    p.add_argument("--judge-models", default="", help="comma-separated override of judge models (default: JUDGE_MODELS from judge_core.py)")
    p.add_argument("--max-tokens", type=int, default=4096, help="max tokens per candidate turn")
    p.add_argument("--env", default="", help="path to .env file (default: repo root .env)")
    p.add_argument("--out", default="local_duel_results.jsonl", help="output jsonl path; pass '' to disable")
    p.add_argument("--eval-run-id", default="local-eval-run", help="tag used for Engy per-run error budget tracking")
    p.add_argument("--show-categories", action="store_true", help="print per-category/requires yes-rate breakdown per sample")
    p.add_argument("--check-setup", action="store_true", help="print config + import sanity check and exit, no network calls")
    args = p.parse_args()
    args.env = args.env or None
    args.scoring_cache = args.scoring_cache or None
    args.out = args.out or None
    return args


def main() -> None:
    args = parse_args()
    if args.check_setup:
        check_setup(args)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
