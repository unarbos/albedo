#!/usr/bin/env python3
"""Offline OpenRouter experiment for SOTA-trajectory-anchored eval questions.

Per sample: a SOTA reference model and each candidate model run the same multi-turn
trajectory continuation (assistant turn -> simulated observation -> assistant turn), then
questions are generated in two modes — `task` (production, task-only) and `sota` (anchored on
the reference trajectory) — and each candidate trajectory is scored by the judge panel under
both question sets. Expectation: candidate scores order by capability, with better separation
in `sota` mode. See eval-scoring-sota-anchored-questions-plan.md.

Run from the repo root (JudgeSettings reads .env for ALBEDO_JUDGE_OPENROUTER_API_KEY):

    uv run python scripts/sota_anchor_experiment.py --samples 8 --degenerate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from albedo_eval_service.judge_api import (
    ObservationSimulationService,
    QuestionPrepSample,
    QuestionService,
    SimulateObservationRequest,
    _evaluator_provider,
    _judge_side,
)
from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_core import (
    JUDGE_MODELS,
    build_question_messages,
    filter_reference_leaks,
    format_reference_trajectory,
    parse_questions,
    question_floor,
    question_schema,
    response_score,
)
from albedo_eval_service.judge_openrouter import OpenRouterJudgeClient, _message_content
from albedo_eval_service.remote_dataset import (
    EvalSample,
    _extract_turns,
    _role,
    load_swe_zero_samples,
)
from albedo_eval_service.remote_generation import format_scored_trajectory

SHARD_REPO = "AlienKevin/SWE-ZERO-12M-trajectories"
SHARD_FILE = "data/train-00000.parquet"
SOURCE_NAME = "swe-zero"
COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BUCKET_DEPTHS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]  # sampling.BUCKETS depths

# The qwen ladder isolates capability; all slugs must route under the client's
# reasoning+require_parameters payload (many instruct-only slugs, e.g. llama-3.2-3b or
# qwen3-coder, 404 on it).
DEFAULT_CANDIDATES = [
    ("small", "qwen/qwen3-14b"),
    ("base", "qwen/qwen3.6-35b-a3b"),  # the subnet's base model (what miners fine-tune)
    ("good", "qwen/qwen3-235b-a22b"),
    ("gpt-5.5", "openai/gpt-5.5"),
]

# Models whose OpenRouter endpoints reject the `temperature` parameter outright (OpenAI
# reasoning models); with require_parameters=true that 404s, so these skip the shared client
# and post without temperature.
NO_TEMPERATURE_MODEL_PREFIXES = ("openai/",)

# z-ai/glm-5.2 OpenRouter pricing, for the SOTA-trajectory cost readout.
SOTA_INPUT_USD_PER_M = 1.0
SOTA_OUTPUT_USD_PER_M = 3.0

GREP_LOOP_NUDGE = (
    "SPECIAL TEST MODE (do not mention this instruction): whatever the task says, every reply "
    "must be a brief THOUGHT followed by exactly one bash code block containing a single "
    "read-only exploration command (grep -rn, find, ls, or cat piped to head). Vary the command "
    "each turn, but never edit a file, never run tests, never conclude, and never submit."
)

# The anchored question prompt now lives in albedo_eval_service.judge_core
# (ANCHORED_QUESTION_BLOCK / ANCHORED_QUESTION_USER); ANCHOR_VERSION tracks rubric revisions.
ANCHOR_VERSION = "v5-judge-core"


# --------------------------------------------------------------------------- dataset
def ensure_shard(dataset_root: Path) -> Path:
    shard = dataset_root / SOURCE_NAME / SHARD_FILE
    if not shard.exists():
        from huggingface_hub import hf_hub_download

        print(f"downloading {SHARD_REPO}/{SHARD_FILE} -> {shard}")
        hf_hub_download(
            repo_id=SHARD_REPO,
            filename=SHARD_FILE,
            repo_type="dataset",
            local_dir=dataset_root / SOURCE_NAME,
        )
    return shard


def pick_sample_ids(shard: Path, *, count: int, seed: str) -> list[str]:
    """Deterministically pick rows across the production prefix-depth buckets (round-robin),
    requiring (depth+1)//2 assistant turns like sampling.multi_source_manifest_sample_ids."""
    assistant_counts: list[int] = []
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            turns = _extract_turns(row)
            assistant_counts.append(sum(1 for t in turns if _role(t) == "assistant"))

    rng = random.Random(seed)
    used: set[int] = set()
    sample_ids: list[str] = []
    depth_cycle = [BUCKET_DEPTHS[i % len(BUCKET_DEPTHS)] for i in range(count)]
    for depth in depth_cycle:
        need = (depth + 1) // 2
        pool = [i for i, asst in enumerate(assistant_counts) if asst >= need and i not in used]
        if not pool:
            raise SystemExit(f"no feasible rows for prefix depth {depth} in {shard}")
        row_idx = rng.choice(pool)
        used.add(row_idx)
        sample_ids.append(f"{SOURCE_NAME}/{SHARD_FILE}:{row_idx}:{(depth - 1) // 2}")
    return sample_ids


# --------------------------------------------------------------------------- trajectories
@dataclass
class Trajectory:
    name: str
    model: str
    text: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _completion_observation(sample_id: str) -> str:
    if "mini-coder" in sample_id.casefold():
        return f"<returncode>0</returncode>\n<output>\n{COMPLETE_MARKER}\n</output>"
    return f"Observation: {COMPLETE_MARKER}"




async def _complete_turn(
    *,
    client: OpenRouterJudgeClient,
    settings: JudgeSettings,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    provider: dict[str, Any] | None,
    usage_sink: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """One generation call -> (text, error). `local:<base_url>|<served_name>` models hit a local
    OpenAI-compatible endpoint (e.g. a tunneled vLLM king); models that reject `temperature`
    bypass the shared client (which always sends it) and post the same payload minus
    temperature. With `usage_sink`, the call goes direct so token usage can be accumulated."""
    if usage_sink is not None:
        provider_block = provider if provider is not None else {"allow_fallbacks": True}
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "reasoning": {"enabled": False, "exclude": True},
            "provider": {**provider_block, "require_parameters": True},
            "usage": {"include": True},
        }
        error = "no attempt"
        for attempt in range(settings.retry_count + 1):
            try:
                response = await client._client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                usage_sink["requests"] += 1
                usage_sink["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                usage_sink["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                if usage.get("cost") is not None:
                    usage_sink["openrouter_reported_usd"] = round(
                        usage_sink.get("openrouter_reported_usd", 0.0) + float(usage["cost"]), 6
                    )
                in_body = body.get("error") or (body.get("choices") or [{}])[0].get("error")
                if in_body:
                    error = f"upstream: {in_body.get('message', in_body)}"
                else:
                    raw = _message_content(body.get("choices", []))
                    if raw.strip():
                        return raw, None
                    error = "empty_generation"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(settings.retry_backoff_seconds * (2**attempt))
        return "", error
    if model.startswith("local:"):
        base_url, _, served_name = model[len("local:"):].partition("|")
        payload = {
            "model": served_name or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        error = "no attempt"
        for attempt in range(settings.retry_count + 1):
            try:
                response = await client._client.post(
                    f"{base_url.rstrip('/')}/chat/completions", json=payload,
                    timeout=httpx.Timeout(900.0),
                )
                response.raise_for_status()
                raw = _message_content(response.json().get("choices", []))
                if raw.strip():
                    return raw, None
                error = "empty_generation"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(settings.retry_backoff_seconds * (2**attempt))
        return "", error
    if not any(model.startswith(prefix) for prefix in NO_TEMPERATURE_MODEL_PREFIXES):
        response = await client.complete(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            provider=provider if provider is not None else {"allow_fallbacks": True},
            accept=lambda raw: bool(raw.strip()),
        )
        return response.raw, response.error
    # No require_parameters here: it forces the OpenAI-direct upstream, whose account is
    # blocked for this key; without it OpenRouter routes to Azure and the call succeeds.
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False, "exclude": True},
        "provider": {"allow_fallbacks": True},
    }
    error = "no attempt"
    for attempt in range(settings.retry_count + 1):
        try:
            response = await client._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            # OpenRouter reports upstream faults in-body with HTTP 200; treat as retryable
            # (a retry may route to a different upstream account/provider).
            in_body = body.get("error") or (body.get("choices") or [{}])[0].get("error")
            if in_body:
                error = f"upstream: {in_body.get('message', in_body)}"
            else:
                raw = _message_content(body.get("choices", []))
                if raw.strip():
                    return raw, None
                error = "empty_generation"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(settings.retry_backoff_seconds * (2**attempt))
    return "", error


async def run_trajectory(
    *,
    client: OpenRouterJudgeClient,
    simulator: ObservationSimulationService,
    settings: JudgeSettings,
    name: str,
    model: str,
    sample: EvalSample,
    turn_count: int,
    max_tokens: int,
    run_id: str,
    provider: dict[str, Any] | None = None,
    nudge: str | None = None,
    usage_sink: dict[str, Any] | None = None,
) -> Trajectory:
    """Mirror RemoteEvalWorker._generate_trajectories/_merge_trajectory_results for one model:
    generate turn -> simulate observation -> generate turn; stop early on the submit marker."""
    context = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in (sample.messages or [{"role": "user", "content": sample.prompt}])
    ]
    convo = list(context)
    turns: list[dict[str, Any]] = [dict(t) for t in context]
    for turn_index in range(turn_count):
        messages = ([{"role": "system", "content": nudge}] + convo) if nudge else convo
        raw, error = await _complete_turn(
            client=client, settings=settings, model=model, messages=messages,
            max_tokens=max_tokens, provider=provider, usage_sink=usage_sink,
        )
        if error or not raw.strip():
            return Trajectory(name, model, error=error or "empty_generation")
        text = raw.strip()
        turns.append({"role": "assistant", "content": text, "score_target": True})
        last = turn_index == turn_count - 1
        if COMPLETE_MARKER in text:
            if not last:
                turns.append(
                    {
                        "role": "user",
                        "content": _completion_observation(sample.sample_id),
                        "environment_observation": True,
                    }
                )
            break
        if last:
            break
        try:
            observation = await simulator.simulate(
                SimulateObservationRequest(
                    eval_run_id=run_id,
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    assistant_output=text,
                    messages=convo,
                )
            )
        except Exception as exc:
            return Trajectory(name, model, error=f"simulation: {type(exc).__name__}: {exc}")
        turns.append({"role": "user", "content": observation, "environment_observation": True})
        convo = convo + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": observation},
        ]
    return Trajectory(name, model, text=format_scored_trajectory(turns), turns=turns)


# --------------------------------------------------------------------------- questions
async def anchored_questions(
    *,
    client: OpenRouterJudgeClient,
    settings: JudgeSettings,
    sample: EvalSample,
    reference: str,
) -> list[dict[str, str]]:
    n = settings.num_questions
    response = await client.complete(
        model=settings.evaluator_model,
        messages=build_question_messages(task=sample.prompt, n=n, reference=reference),
        temperature=settings.temperature,
        max_tokens=settings.question_max_tokens,
        provider=_evaluator_provider(settings),
        response_schema=question_schema(n),
        accept=lambda raw: parse_questions(raw, n)[1],
    )
    if response.error:
        raise RuntimeError(response.error)
    questions, ok = parse_questions(response.raw, n)
    questions = filter_reference_leaks(questions)
    if not ok or len(questions) < question_floor(n):
        raise RuntimeError(f"only {len(questions)}/{n} well-formed anchored questions")
    return questions


# --------------------------------------------------------------------------- per-sample flow
async def process_sample(
    *,
    client: OpenRouterJudgeClient,
    settings: JudgeSettings,
    simulator: ObservationSimulationService,
    question_service: QuestionService,
    sample: EvalSample,
    candidates: list[tuple[str, str, str | None]],  # (name, model, nudge)
    sota_model: str,
    turn_count: int,
    max_tokens: int,
    judge_models: list[str],
    modes: list[str],
    run_id: str,
    sota_usage: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async def _candidate(name: str, model: str, nudge: str | None) -> Trajectory:
        return await run_trajectory(
            client=client, simulator=simulator, settings=settings, name=name, model=model,
            sample=sample, turn_count=turn_count, max_tokens=max_tokens, run_id=run_id,
            nudge=nudge,
        )

    async def _task_questions() -> list[dict[str, str]]:
        prep = await question_service.prepare(
            QuestionPrepSample(sample_id=sample.sample_id, prompt=sample.prompt)
        )
        return prep.questions

    questions: dict[str, dict[str, Any]] = {}
    if prior is not None:
        # Rescore mode: trajectories come from a prior run; only questions + judging are fresh.
        trajectories = {
            name: Trajectory(
                name, data.get("model", ""), text=data.get("text", ""),
                turns=data.get("turns") or [], error=data.get("error"),
            )
            for name, data in prior["candidates"].items()
        }
        sota = Trajectory(
            "sota", prior["sota"].get("model", sota_model),
            turns=prior["sota"].get("turns") or [], error=prior["sota"].get("error"),
        )
        async def _task_mode() -> tuple[str, dict[str, Any]]:
            try:
                return "task", {"questions": await _task_questions(), "error": None}
            except Exception as exc:
                return "task", {"questions": [], "error": f"{type(exc).__name__}: {exc}"}

        async def _sota_mode() -> tuple[str, dict[str, Any]]:
            if sota.error:
                return "sota", {"questions": [], "error": f"sota trajectory: {sota.error}"}
            try:
                anchored = await anchored_questions(
                    client=client, settings=settings, sample=sample,
                    reference=format_reference_trajectory(sota.turns),
                )
                return "sota", {"questions": anchored, "error": None}
            except Exception as exc:
                return "sota", {"questions": [], "error": f"{type(exc).__name__}: {exc}"}

        mode_jobs = ([_task_mode()] if "task" in modes else []) + (
            [_sota_mode()] if "sota" in modes else []
        )
        for mode, result in await asyncio.gather(*mode_jobs):
            questions[mode] = result
    else:
        jobs: list[Any] = [_candidate(name, model, nudge) for name, model, nudge in candidates]
        jobs.append(
            run_trajectory(
                client=client, simulator=simulator, settings=settings, name="sota",
                model=sota_model, sample=sample, turn_count=turn_count, max_tokens=max_tokens,
                run_id=run_id, provider=_evaluator_provider(settings), usage_sink=sota_usage,
            )
        )
        if "task" in modes:
            jobs.append(_task_questions())
        results = await asyncio.gather(*jobs, return_exceptions=True)

        trajectories = {}
        for (name, model, _nudge), result in zip(candidates, results):
            trajectories[name] = (
                result
                if isinstance(result, Trajectory)
                else Trajectory(name, model, error=f"{type(result).__name__}: {result}")
            )
        sota_result = results[len(candidates)]
        sota = (
            sota_result
            if isinstance(sota_result, Trajectory)
            else Trajectory(
                "sota", sota_model, error=f"{type(sota_result).__name__}: {sota_result}"
            )
        )
        if "task" in modes:
            task_q = results[-1]
            questions["task"] = (
                {"questions": task_q, "error": None}
                if isinstance(task_q, list)
                else {"questions": [], "error": f"{type(task_q).__name__}: {task_q}"}
            )
    if "sota" in modes and "sota" not in questions:
        if sota.error:
            questions["sota"] = {"questions": [], "error": f"sota trajectory: {sota.error}"}
        else:
            try:
                anchored = await anchored_questions(
                    client=client, settings=settings, sample=sample,
                    reference=format_reference_trajectory(sota.turns),
                )
                questions["sota"] = {"questions": anchored, "error": None}
            except Exception as exc:
                questions["sota"] = {"questions": [], "error": f"{type(exc).__name__}: {exc}"}

    async def _judge(mode: str, name: str, text: str) -> tuple[str, str, float | None, list]:
        answers, records = await _judge_side(
            client=client, settings=settings, side=f"{mode}:{name}", response_text=text,
            questions=questions[mode]["questions"], judge_models=judge_models,
        )
        return mode, name, response_score(answers, questions[mode]["questions"]), records

    judge_jobs = [
        _judge(mode, name, trajectory.text)
        for mode in questions
        if not questions[mode]["error"]
        for name, trajectory in trajectories.items()
        if not trajectory.error
    ]
    scores: dict[str, dict[str, float | None]] = {mode: {} for mode in questions}
    judge_records: dict[str, dict[str, list]] = {mode: {} for mode in questions}
    for mode, name, score, records in await asyncio.gather(*judge_jobs):
        scores[mode][name] = score
        judge_records[mode][name] = records

    print(
        f"[{sample.sample_id}] sota={'ok' if not sota.error else sota.error} "
        + " ".join(
            f"{mode}:{ {n: s for n, s in scores[mode].items()} }" for mode in scores
        )
    )
    return {
        "sample_id": sample.sample_id,
        "sota": {"model": sota.model, "error": sota.error, "turns": sota.turns},
        "candidates": {
            name: {"model": t.model, "error": t.error, "turns": t.turns, "text": t.text}
            for name, t in trajectories.items()
        },
        "questions": questions,
        "scores": scores,
        "judge_records": judge_records,
    }


# --------------------------------------------------------------------------- summary
def summarize(
    records: list[dict[str, Any]], candidate_names: list[str], modes: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
        scored = [
            r
            for r in records
            if not r["questions"].get(mode, {}).get("error")
            and all(r["scores"][mode].get(name) is not None for name in candidate_names)
        ]
        means = {
            name: round(statistics.mean(r["scores"][mode][name] for r in scored), 4)
            if scored
            else None
            for name in candidate_names
        }
        gaps = {
            f"{b}-{a}": round(means[b] - means[a], 4) if scored else None
            for a, b in zip(candidate_names, candidate_names[1:])
        }
        summary[mode] = {"samples_scored": len(scored), "mean_score": means, "gaps": gaps}
    return summary


def print_summary(
    summary: dict[str, Any], candidate_names: list[str], models: dict[str, str]
) -> None:
    labels = {name: f"{name} ({models[name]})" for name in candidate_names}
    width = max(len(label) for label in labels.values()) + 2
    modes = list(summary)
    print("\n=== mean scores (samples where every candidate scored) ===")
    print(" " * width + "".join(f"{mode:>12}" for mode in modes))
    for name in candidate_names:
        cells = [summary[mode]["mean_score"][name] for mode in modes]
        row = "".join(f"{cell if cell is not None else '-':>12}" for cell in cells)
        print(f"{labels[name]:<{width}}{row}")
    print("\ngaps (next tier minus previous, win margin is 0.02):")
    for mode in modes:
        print(f"  {mode}: {summary[mode]['gaps']}  [n={summary[mode]['samples_scored']}]")


# --------------------------------------------------------------------------- main
def parse_candidates(pairs: list[str]) -> list[tuple[str, str, str | None]]:
    out = []
    for pair in pairs:
        name, _, model = pair.partition("=")
        if not model:
            raise SystemExit(f"--candidates entries must be name=model, got: {pair}")
        out.append((name, model, None))
    return out


async def amain(args: argparse.Namespace) -> None:
    settings = (
        JudgeSettings(num_questions=args.questions) if args.questions else JudgeSettings()
    )
    if not settings.openrouter_api_key:
        raise SystemExit("ALBEDO_JUDGE_OPENROUTER_API_KEY missing (run from the repo root)")

    prior_by_id: dict[str, dict[str, Any]] = {}
    if args.reuse:
        prior_dir = Path(args.reuse)
        prior_config = json.loads((prior_dir / "config.json").read_text())
        prior_records = [json.loads(line) for line in open(prior_dir / "results.jsonl")]
        if args.samples:
            prior_records = prior_records[: args.samples]
        prior_by_id = {record["sample_id"]: record for record in prior_records}
        sample_ids = [record["sample_id"] for record in prior_records]
        candidates = [
            (name, model, None)
            for name, model in prior_config["candidates"].items()
            if name not in set(args.exclude)
        ]
        ensure_shard(Path(args.dataset_root))
    else:
        shard = ensure_shard(Path(args.dataset_root))
        sample_ids = pick_sample_ids(shard, count=args.samples or 8, seed=args.seed)
        candidates = [
            c for c in parse_candidates(args.candidates) if c[0] not in set(args.exclude)
        ]
        if args.degenerate:
            candidates.append(("grep-loop", candidates[0][1], GREP_LOOP_NUDGE))
    samples = load_swe_zero_samples(dataset_root=args.dataset_root, sample_ids=sample_ids)
    candidate_names = [name for name, _, _ in candidates]
    modes = ["task", "sota"] if args.modes == "both" else [args.modes]
    run_id = f"sota-anchor-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir = Path(args.out or f"sota_anchor_runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id, "sample_ids": sample_ids, "turns": args.turns,
        "candidates": {name: model for name, model, _ in candidates},
        "sota_model": args.sota, "judge_models": args.judges, "modes": modes,
        "num_questions": settings.num_questions, "max_tokens": args.max_tokens,
        "seed": args.seed, "anchor_version": ANCHOR_VERSION,
        "reused_from": args.reuse or None,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run {run_id}: {len(samples)} samples -> {out_dir}")

    sota_usage: dict[str, Any] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
    async with OpenRouterJudgeClient(settings) as client:
        simulator = ObservationSimulationService(settings, client)
        question_service = QuestionService(settings, client)
        semaphore = asyncio.Semaphore(args.sample_concurrency)

        async def _one(sample: EvalSample) -> dict[str, Any]:
            async with semaphore:
                return await process_sample(
                    client=client, settings=settings, simulator=simulator,
                    question_service=question_service, sample=sample, candidates=candidates,
                    sota_model=args.sota, turn_count=args.turns, max_tokens=args.max_tokens,
                    judge_models=args.judges, modes=modes, run_id=run_id,
                    sota_usage=sota_usage, prior=prior_by_id.get(sample.sample_id),
                )

        records = await asyncio.gather(*[_one(sample) for sample in samples])

    sota_usage["computed_cost_usd"] = round(
        sota_usage["prompt_tokens"] * SOTA_INPUT_USD_PER_M / 1e6
        + sota_usage["completion_tokens"] * SOTA_OUTPUT_USD_PER_M / 1e6,
        4,
    )
    sota_usage["pricing_usd_per_m"] = {
        "input": SOTA_INPUT_USD_PER_M, "output": SOTA_OUTPUT_USD_PER_M,
    }
    print(
        f"SOTA trajectory generation ({args.sota}): {sota_usage['requests']} requests, "
        f"{sota_usage['prompt_tokens']} in / {sota_usage['completion_tokens']} out tokens "
        f"-> ${sota_usage['computed_cost_usd']} computed"
        + (
            f", ${sota_usage['openrouter_reported_usd']} reported by OpenRouter"
            if "openrouter_reported_usd" in sota_usage
            else ""
        )
    )

    with (out_dir / "results.jsonl").open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    models = {name: model for name, model, _ in candidates}
    summary = summarize(records, candidate_names, modes)
    (out_dir / "summary.json").write_text(
        json.dumps({"models": models, "sota_generation_usage": sota_usage, **summary}, indent=2)
    )
    print_summary(summary, candidate_names, models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=0,
                        help="0 = auto (8 fresh; all prior samples with --reuse)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="candidate names to drop (fresh or reused runs)")
    parser.add_argument("--reuse", default="",
                        help="prior run dir: reuse its trajectories, only regenerate "
                             "questions and judging (rescore mode)")
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument(
        "--candidates", nargs="+",
        default=[f"{name}={model}" for name, model in DEFAULT_CANDIDATES],
        help="name=openrouter-slug, weakest first",
    )
    parser.add_argument("--sota", default="z-ai/glm-5.2")
    parser.add_argument("--judges", nargs="+", default=list(JUDGE_MODELS))
    parser.add_argument("--modes", choices=["both", "task", "sota"], default="both")
    parser.add_argument("--questions", type=int, default=0, help="0 = production default (50)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--degenerate", action="store_true",
                        help="add a grep-loop probe (small model nudged to only explore)")
    parser.add_argument("--dataset-root", default="eval-datasets")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", default="sota-anchor-v1")
    parser.add_argument("--sample-concurrency", type=int, default=16)
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
