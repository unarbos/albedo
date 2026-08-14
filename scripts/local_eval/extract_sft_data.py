#!/usr/bin/env python3
"""Build a verified SFT dataset for the king's weak checklist buckets.

Pipeline, per candidate sample:
1. Resolve the sample's checklist (cached from a prior duel/weakness-report run, or generated
   fresh via the real evaluator).
2. Generate the SOTA reference trajectory for that sample (generate_reference_turns) — the same
   model the checklist is *anchored to*, so it's checklist-satisfying by construction, not by
   assumption.
3. Score the reference against its own checklist with the real judge pipeline
   (score_single_side) and read off, per targeted bucket ("category/requires"), whether the
   reference actually answers "1" on that bucket's questions for this sample.
4. Keep the sample as a training example ONLY for buckets it verifiably satisfies, tagged with
   which buckets those are, until each bucket's --per-bucket quota is filled.

Output is a JSONL of {sample_id, phase, prompt, prefix_turns, target_turns, buckets_satisfied,
reference_model} records — prefix_turns is context (mask out of the loss), target_turns is what
you train on (mask IN, matching the score_target flag exactly the validator itself scores).

This makes real, billed judge/evaluator/reference API calls (GLM-5.2 etc. via OpenRouter/Engy) —
same cost profile as run_duel.py. Use --check-setup first, and start with a small --max-samples.

Usage
-----
Auto-select the 8 weakest buckets from a weakness-report and pull up to 100 verified examples
for each:

    python scripts/local_eval/extract_sft_data.py \\
        --weakness-report local_weakness_report.json --top-n-buckets 8 --per-bucket 100 \\
        --out sft_data.jsonl

Target specific buckets you already decided on (e.g. from margin_calculator.py's output):

    python scripts/local_eval/extract_sft_data.py \\
        --bucket workflow/action=150 --bucket grounding/action=100 \\
        --out sft_data.jsonl
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
    build_question_service,
    generate_reference_turns,
    load_settings,
    prefix_and_turn_count,
    sample_phase,
    score_single_side,
)
from run_duel import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_SCORING_CACHE,
    judge_models_for,
    load_cache,
    load_jsonl,
    resolve_questions,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.remote.generation import format_scored_trajectory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_bucket_specs(specs: list[str], default_quota: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in specs:
        if "=" in spec:
            key, quota = spec.rsplit("=", 1)
            out[key.strip()] = int(quota)
        else:
            out[spec.strip()] = default_quota
    return out


def buckets_from_weakness_report(
    report_path: str, *, top_n: int, use_phase: bool, default_quota: int
) -> dict[str, int]:
    with open(report_path) as f:
        report = json.load(f)
    rows = report["by_category_and_phase"] if use_phase else report["by_category"]
    # rows are already sorted weakest-first by aggregate_category_breakdown()
    return {row["key"]: default_quota for row in rows[:top_n]}


def question_bucket_key(question: dict[str, Any]) -> str:
    return f"{question.get('category', 'other')}/{question.get('requires', 'neutral')}"


async def build_dataset(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    all_records = load_jsonl(Path(args.samples))
    if args.seed is not None:
        random.Random(args.seed).shuffle(all_records)
    cached_by_id = load_cache(args.scoring_cache)
    judge_models = judge_models_for(args)

    if args.bucket:
        quotas = parse_bucket_specs(args.bucket, args.per_bucket)
    elif args.weakness_report:
        quotas = buckets_from_weakness_report(
            args.weakness_report,
            top_n=args.top_n_buckets,
            use_phase=args.phase,
            default_quota=args.per_bucket,
        )
    else:
        raise SystemExit("pass --bucket KEY[=QUOTA] (repeatable) or --weakness-report FILE")

    remaining = dict(quotas)
    print("target buckets and quotas:")
    for key, quota in quotas.items():
        print(f"  {key:34s} quota={quota}")

    client = JudgeLLMClient(settings)
    question_service, simulator, repo_context = build_question_service(client, settings)
    out_f = open(args.out, "w")
    accepted = 0
    attempted = 0
    per_bucket_hits: dict[str, int] = {key: 0 for key in quotas}
    try:
        for record in all_records:
            if attempted >= args.max_samples:
                print(f"\nhit --max-samples={args.max_samples}, stopping")
                break
            if all(v <= 0 for v in remaining.values()):
                print("\nall bucket quotas filled, stopping")
                break

            sample_id = record["sample_id"]
            prompt = record.get("prompt", "")
            prefix_turns, n_turns = prefix_and_turn_count(record["previous_king_turns"])
            phase = sample_phase(prefix_turns)
            attempted += 1
            print(f"[{attempted}] {sample_id} phase={phase}")

            try:
                questions, reference_made_edit, was_cached = await resolve_questions(
                    question_service=question_service,
                    cached_by_id=cached_by_id,
                    sample_id=sample_id,
                    prompt=prompt,
                    prefix_turns=prefix_turns,
                    n_turns=n_turns,
                )
            except Exception as exc:
                print(f"  skip: couldn't resolve questions ({exc})")
                continue

            candidate_keys = {question_bucket_key(q) for q in questions} & {
                key for key, left in remaining.items() if left > 0
            }
            if not candidate_keys:
                print("  skip: no questions in this sample match a bucket still under quota")
                continue

            try:
                full_turns, reference_model = await generate_reference_turns(
                    client_llm=client,
                    settings=settings,
                    simulator=simulator,
                    sample_id=sample_id,
                    prompt=prompt,
                    prefix_turns=prefix_turns,
                    n_turns=n_turns,
                    eval_run_id=args.eval_run_id,
                )
            except Exception as exc:
                print(f"  skip: reference generation failed ({exc})")
                continue

            reference_output = format_scored_trajectory(full_turns)
            result = await score_single_side(
                client_llm=client,
                settings=settings,
                sample_id=sample_id,
                questions=questions,
                output_text=reference_output,
                side="reference",
                judge_models=judge_models,
                reference_made_edit=reference_made_edit,
            )
            if not result["scored"]:
                print("  skip: reference didn't score cleanly (judge parse failure)")
                continue

            # A question only counts as "verified satisfied" if EVERY judge that answered it
            # said yes — one judge dissent means this sample is not a clean positive for that
            # bucket, better to skip it than teach a borderline case as if it were a clear win.
            per_question_yes: dict[str, list[float]] = {}
            for rec in result["judge_results"]:
                for qid, ans in (rec.get("answers") or {}).items():
                    if ans is not None:
                        per_question_yes.setdefault(qid, []).append(float(ans))

            questions_by_id = {q["id"]: q for q in questions}
            satisfied_keys = set()
            for qid, votes in per_question_yes.items():
                q = questions_by_id.get(qid)
                if q is None or not votes or min(votes) < 1.0:
                    continue
                key = question_bucket_key(q)
                if key in candidate_keys:
                    satisfied_keys.add(key)

            satisfied_keys = {k for k in satisfied_keys if remaining.get(k, 0) > 0}
            if not satisfied_keys:
                print("  skip: reference didn't cleanly satisfy any still-needed bucket")
                continue

            target_turns = full_turns[len(prefix_turns):]
            out_f.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "phase": phase,
                        "prompt": prompt,
                        "prefix_turns": prefix_turns,
                        "target_turns": target_turns,
                        "buckets_satisfied": sorted(satisfied_keys),
                        "reference_model": reference_model,
                    }
                )
                + "\n"
            )
            out_f.flush()
            accepted += 1
            for key in satisfied_keys:
                remaining[key] -= 1
                per_bucket_hits[key] += 1
            print(f"  accepted -> buckets {sorted(satisfied_keys)}")
    finally:
        out_f.close()
        await client.aclose()
        if repo_context is not None:
            await repo_context.aclose()

    print(f"\n=== DONE: {accepted} examples written from {attempted} samples attempted ===")
    print("per-bucket results (hit/quota):")
    for key, quota in quotas.items():
        print(f"  {key:34s} {per_bucket_hits[key]:4d}/{quota}")
    short = [key for key, quota in quotas.items() if per_bucket_hits[key] < quota]
    if short:
        print(
            f"\n{len(short)} bucket(s) didn't reach quota from this sample pool: {short}\n"
            "Try a larger --samples pool, raise --max-samples, or accept fewer examples for "
            "these buckets — don't pad with low-quality/unverified examples to force the count."
        )


def check_setup(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    print(f"repo root:          {REPO_ROOT}")
    print(f"env file:           {args.env or REPO_ROOT / '.env'}")
    print(f"openrouter key set: {bool(settings.openrouter_api_key)}")
    print(f"sota models:        {settings.sota_models}")
    print(f"judge models:       {list(JUDGE_MODELS)}")
    samples_path = Path(args.samples)
    print(f"samples file:       {samples_path} (exists={samples_path.exists()})")
    if args.weakness_report:
        wr_path = Path(args.weakness_report)
        print(f"weakness report:    {wr_path} (exists={wr_path.exists()})")
    if not settings.openrouter_api_key:
        print("\nWARNING: ALBEDO_JUDGE_OPENROUTER_API_KEY is not set — calls will fail.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="generated-samples.jsonl to draw candidate samples from")
    p.add_argument("--scoring-cache", default=str(DEFAULT_SCORING_CACHE), help="scoring-results.jsonl to reuse checklists from; pass '' to disable")
    p.add_argument("--weakness-report", default="", help="weakness-report JSON (run_duel.py --mode weakness-report --out ...) to auto-select weak buckets from")
    p.add_argument("--top-n-buckets", type=int, default=8, help="how many weakest buckets to auto-target from --weakness-report")
    p.add_argument("--phase", action="store_true", help="use by_category_and_phase rows from --weakness-report instead of by_category")
    p.add_argument("--bucket", action="append", default=[], metavar="KEY[=QUOTA]", help="explicit bucket to target, e.g. 'workflow/action=150'. Repeatable. Overrides --weakness-report if given.")
    p.add_argument("--per-bucket", type=int, default=100, help="default quota per bucket when not overridden per-bucket")
    p.add_argument("--max-samples", type=int, default=200, help="hard cap on samples attempted, regardless of quotas (cost control)")
    p.add_argument("--judge-models", default="", help="comma-separated override of judge models used to verify the reference")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed for sample draw order")
    p.add_argument("--env", default="", help="path to .env file (default: repo root .env)")
    p.add_argument("--eval-run-id", default="local-sft-extract", help="tag used for Engy per-run error budget tracking")
    p.add_argument("--out", default="sft_data.jsonl", help="output JSONL path")
    p.add_argument("--check-setup", action="store_true", help="print config + import sanity check and exit, no network calls")
    args = p.parse_args()
    args.env = args.env or None
    args.scoring_cache = args.scoring_cache or None
    return args


def main() -> None:
    args = parse_args()
    if args.check_setup:
        check_setup(args)
        return
    asyncio.run(build_dataset(args))


if __name__ == "__main__":
    main()
