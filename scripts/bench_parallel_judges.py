#!/usr/bin/env python3
"""Benchmark sequential vs bounded-parallel judge scoring in isolation.

This script does not change the production judge path. It uses the real
ChutesJudge transport and verdict parser, then compares:

  sequential: ChutesJudge.query_judges(), the current production behavior
  parallel:   one task per judge model, with per-model concurrency/spacing caps

Environment:
  CHUTES_API_KEY is required for Chutes.
  OPENROUTER_API_KEY is optional but recommended for fallback.

Examples:
  python scripts/bench_parallel_judges.py --mode both
  python scripts/bench_parallel_judges.py --mode parallel --runs 5 --task-concurrency 2
  python scripts/bench_parallel_judges.py --case-json /tmp/judge_case.json

case-json shape:
  {
    "messages": [{"role": "user", "content": "..."}],
    "king_reply": "...",
    "chal_reply": "..."
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from albedo.config import (
    JUDGE_CHUTES_GIVEUP_TASKS,
    JUDGE_MAX_CONCURRENCY_PER_MODEL,
    JUDGE_MODELS,
    JUDGE_RATE_LIMITS,
    JUDGE_TOTAL_S,
)
from albedo.duel.turn import strip_reply_injection
from albedo.judge import ChutesJudge
from albedo.judge.client import _max_tokens_for
from albedo.judge.verdict import parse_metric_verdict


DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a coding agent. Continue the trajectory by making concrete "
            "progress toward fixing the bug."
        ),
    },
    {
        "role": "user",
        "content": (
            "The unit test test_parse_timeout is failing because parse_timeout('5m') "
            "returns 5 instead of 300. Find and fix the issue."
        ),
    },
]

DEFAULT_KING_REPLY = """THOUGHT: The parser is probably dropping the unit suffix and
returning only the numeric portion. I will inspect the parser and add unit conversion.
ACTION:
```bash
rg -n "def parse_timeout|parse_timeout" .
```"""

DEFAULT_CHAL_REPLY = """THOUGHT: The failure says minutes are parsed as raw numbers,
so I need to update the unit handling and add a regression test for m -> seconds.
ACTION:
```bash
rg -n "parse_timeout|timeout" . && sed -n '1,220p' tests/test_config.py
```"""


@dataclass
class JudgeRun:
    mode: str
    run_idx: int
    elapsed_s: float
    resolved: int
    parsed: int
    per_model: dict[str, dict] = field(default_factory=dict)


class ModelLimiter:
    """Small async limiter for isolated judge experiments.

    The production config has per-model limits in chain.toml. This harness enforces
    them around each model's whole Chutes -> fallback resolution so parallel testing
    does not immediately become an accidental rate-limit test.
    """

    def __init__(self) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._last_start: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _limits(self, model: str) -> tuple[int, float]:
        cfg = JUDGE_RATE_LIMITS.get(model, {}) if isinstance(JUDGE_RATE_LIMITS, dict) else {}
        max_conc = int(cfg.get("max_concurrency", JUDGE_MAX_CONCURRENCY_PER_MODEL))
        min_interval = float(cfg.get("min_interval_s", 0.0))
        return max(1, max_conc), max(0.0, min_interval)

    async def run(self, model: str, fn: Callable[[], asyncio.Future]) -> str | None:
        max_conc, min_interval = self._limits(model)
        sem = self._sems.get(model)
        if sem is None:
            sem = self._sems[model] = asyncio.Semaphore(max_conc)

        async with sem:
            if min_interval > 0:
                async with self._lock:
                    now = time.monotonic()
                    wait = self._last_start.get(model, 0.0) + min_interval - now
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_start[model] = time.monotonic()
            return await fn()


class ParallelJudgeExperiment:
    """Bounded-parallel version of ChutesJudge.query_judges for experiments only."""

    def __init__(self, judge: ChutesJudge) -> None:
        self.judge = judge
        self.limiter = ModelLimiter()

    async def query_judges_parallel(
        self,
        models: list[str],
        messages: list[dict],
        *,
        accept: Callable[[str], bool],
    ) -> dict[str, str | None]:
        async def _one(model: str) -> tuple[str, str | None, bool]:
            async def _resolve() -> str | None:
                deadline = time.monotonic() + JUDGE_TOTAL_S
                raw: str | None = None

                if not self.judge._chutes_off:  # noqa: SLF001 - isolated experiment
                    candidate = await self.judge._chutes_stream(  # noqa: SLF001
                        model, messages, _max_tokens_for(model)
                    )
                    if candidate is not None and accept(candidate):
                        return candidate

                raw = await self.judge._openrouter(  # noqa: SLF001
                    model,
                    messages,
                    _max_tokens_for(model),
                    accept=accept,
                    deadline=deadline,
                )
                return raw

            raw = await self.limiter.run(model, _resolve)
            # Approximate the production circuit-breaker signal: a parsed Chutes
            # result is indistinguishable from fallback here, so treat any resolved
            # result while Chutes is on as a dry-task reset only at the batch level.
            return model, raw, raw is not None

        results = await asyncio.gather(*[_one(m) for m in models])
        out = {model: raw for model, raw, _ in results}

        if not self.judge._chutes_off:  # noqa: SLF001
            if any(raw is not None for raw in out.values()):
                self.judge._chutes_dry_tasks = 0  # noqa: SLF001
            else:
                self.judge._chutes_dry_tasks += 1  # noqa: SLF001
                if self.judge._chutes_dry_tasks >= JUDGE_CHUTES_GIVEUP_TASKS:  # noqa: SLF001
                    self.judge._chutes_off = True  # noqa: SLF001

        return out


def _load_case(path: str | None) -> tuple[list[dict], str, str]:
    if not path:
        return DEFAULT_MESSAGES, DEFAULT_KING_REPLY, DEFAULT_CHAL_REPLY
    data = json.loads(Path(path).read_text())
    return data["messages"], data["king_reply"], data["chal_reply"]


def _summarize_raws(mode: str, run_idx: int, elapsed_s: float, raws: dict[str, str | None]) -> JudgeRun:
    per_model: dict[str, dict] = {}
    for model, raw in raws.items():
        verdict = parse_metric_verdict(raw or "")
        per_model[model] = {
            "resolved": raw is not None,
            "parse_ok": verdict.parse_ok,
            "judge_mean": verdict.judge_mean,
            "raw_chars": len(raw or ""),
        }
    return JudgeRun(
        mode=mode,
        run_idx=run_idx,
        elapsed_s=elapsed_s,
        resolved=sum(1 for v in per_model.values() if v["resolved"]),
        parsed=sum(1 for v in per_model.values() if v["parse_ok"]),
        per_model=per_model,
    )


async def _run_once(
    *,
    mode: str,
    run_idx: int,
    judge: ChutesJudge,
    parallel: ParallelJudgeExperiment,
    judge_models: list[str],
    judge_messages: list[dict],
) -> JudgeRun:
    start = time.perf_counter()
    accept = lambda raw: parse_metric_verdict(raw).parse_ok
    if mode == "sequential":
        raws = await judge.query_judges(judge_models, judge_messages, accept=accept)
    elif mode == "parallel":
        raws = await parallel.query_judges_parallel(judge_models, judge_messages, accept=accept)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    elapsed = time.perf_counter() - start
    return _summarize_raws(mode, run_idx, elapsed, raws)


async def _run_mode(
    *,
    mode: str,
    runs: int,
    task_concurrency: int,
    judge_models: list[str],
    judge_messages: list[dict],
) -> list[JudgeRun]:
    async with ChutesJudge() as judge:
        parallel = ParallelJudgeExperiment(judge)
        sem = asyncio.Semaphore(task_concurrency)

        async def _guarded(i: int) -> JudgeRun:
            async with sem:
                return await _run_once(
                    mode=mode,
                    run_idx=i,
                    judge=judge,
                    parallel=parallel,
                    judge_models=judge_models,
                    judge_messages=judge_messages,
                )

        return await asyncio.gather(*[_guarded(i) for i in range(runs)])


def _print_runs(runs: list[JudgeRun], n_models: int, *, verbose: bool) -> None:
    for r in runs:
        print(
            f"{r.mode:10s} run={r.run_idx:02d} "
            f"time={r.elapsed_s:7.2f}s resolved={r.resolved}/{n_models} parsed={r.parsed}/{n_models}"
        )
        if verbose:
            for model, data in r.per_model.items():
                print(
                    f"  - {model}: resolved={data['resolved']} parse_ok={data['parse_ok']} "
                    f"mean={data['judge_mean']:.3f} raw_chars={data['raw_chars']}"
                )

    elapsed = [r.elapsed_s for r in runs]
    if elapsed:
        print(
            f"{runs[0].mode:10s} summary: "
            f"mean={statistics.mean(elapsed):.2f}s "
            f"median={statistics.median(elapsed):.2f}s "
            f"min={min(elapsed):.2f}s max={max(elapsed):.2f}s"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sequential", "parallel", "both"], default="both")
    parser.add_argument("--runs", type=int, default=1, help="Number of scoring tasks per mode")
    parser.add_argument(
        "--task-concurrency",
        type=int,
        default=1,
        help="Concurrent scoring tasks per mode; use >1 to simulate parallel duel turns",
    )
    parser.add_argument("--case-json", default=None, help="Optional JSON file with messages/replies")
    parser.add_argument("--judge-model", action="append", dest="judge_models", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    messages, king_reply, chal_reply = _load_case(args.case_json)
    judge_models = args.judge_models or list(JUDGE_MODELS)
    if not judge_models:
        raise SystemExit("No judge models configured; pass --judge-model MODEL")

    king_clean = strip_reply_injection(king_reply)
    chal_clean = strip_reply_injection(chal_reply)

    async with ChutesJudge() as builder:
        judge_messages = builder._build_pairwise_messages(messages, king_clean, chal_clean)  # noqa: SLF001

    modes = ["sequential", "parallel"] if args.mode == "both" else [args.mode]
    all_results: dict[str, list[JudgeRun]] = {}
    for mode in modes:
        print(f"\n=== {mode} ===")
        runs = await _run_mode(
            mode=mode,
            runs=args.runs,
            task_concurrency=max(1, args.task_concurrency),
            judge_models=judge_models,
            judge_messages=judge_messages,
        )
        all_results[mode] = runs
        _print_runs(runs, len(judge_models), verbose=args.verbose)

    if set(all_results) == {"sequential", "parallel"}:
        seq = statistics.mean(r.elapsed_s for r in all_results["sequential"])
        par = statistics.mean(r.elapsed_s for r in all_results["parallel"])
        if par > 0:
            print(f"\nSpeedup estimate: {seq / par:.2f}x faster wall-clock per scoring task")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
