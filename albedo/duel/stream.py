"""SSE generator for a complete duel evaluation (pairwise per-metric scoring)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from albedo.config import (
    DUEL_GATE_ALPHA,
    DUEL_MIN_VALID_TURN_FRAC,
    DUEL_RESAMPLES,
    DUEL_WIN_MARGIN,
    JUDGE_METRIC_KEYS,
)
from albedo.duel.sampler import Sample
from albedo.duel.turn import TurnResult, resolve_model_names, score_turn, _score_single_judge
from albedo.stats import aggregate_duel, paired_bootstrap_lcb

if TYPE_CHECKING:
    from albedo.judge import ChutesJudge
    from albedo.eval_server.sink import DatasetSink

log = logging.getLogger(__name__)


async def _rescore_failed_judges(
    result: TurnResult,
    *,
    judge:        "ChutesJudge",
    judge_models: list[str],
    hotkey:       str,
    seed:         bytes,
    sink:         "DatasetSink | None",
) -> None:
    """Re-run judge scoring for failed judges in a TurnResult; mutates in-place.

    Called after all turns complete so unscored turns get a second chance
    before the final aggregation.
    """
    failed = [e["judge_model"] for e in result.per_judge if not e["parse_ok"]]
    if not failed:
        return

    retry_raw = await asyncio.gather(
        *[
            _score_single_judge(
                judge, m, result.sample,
                result.king_reply, result.chal_reply,
                hotkey, seed, sink=sink,
            )
            for m in failed
        ],
        return_exceptions=True,
    )

    by_model = {e["judge_model"]: e for e in result.per_judge}
    for m, res in zip(failed, retry_raw):
        if isinstance(res, Exception):
            log.warning(
                "retry judge %s on turn %d still failed: %s",
                m, result.sample.global_idx, res,
            )
        else:
            by_model[m] = res

    # Preserve original judge order.
    result.per_judge = [by_model[e["judge_model"]] for e in result.per_judge]

    ok_means       = [e["judge_mean"] for e in result.per_judge if e["parse_ok"]]
    result.parse_ok       = all(e["parse_ok"] for e in result.per_judge)
    result.final_score    = sum(ok_means) / len(ok_means) if ok_means else 0.0
    result.final_score_100 = result.final_score * 100.0
    result.delta          = result.final_score - 0.5


def _sse(event: str, data: dict) -> bytes:
    """Format a server-sent event as UTF-8 bytes."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


def _collect_aggregation_inputs(
    results: list[TurnResult],
    judge_models: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Gather per-judge, per-metric score lists across scored turns.

    Returns judge_model -> {metric -> [score per turn where that judge parsed]}.
    Only judges that parsed a turn contribute that turn's metric scores.
    """
    per_judge_metric: dict[str, dict[str, list[float]]] = {
        jm: {m: [] for m in JUDGE_METRIC_KEYS} for jm in judge_models
    }
    for r in results:
        for entry in r.per_judge:
            if not entry.get("parse_ok"):
                continue
            jm = entry["judge_model"]
            bucket = per_judge_metric.setdefault(jm, {m: [] for m in JUDGE_METRIC_KEYS})
            for m, v in entry.get("metric_scores", {}).items():
                bucket.setdefault(m, []).append(v)
    return per_judge_metric


async def run_duel(
    *,
    samples:      list[Sample],
    king_client:  httpx.AsyncClient,
    chal_client:  httpx.AsyncClient,
    judge:        "ChutesJudge",
    judge_models: list[str],
    seed:         bytes,
    eval_id:      str,
    hotkey:       str,
    max_parallel: int = 8,
    sink:         "DatasetSink | None" = None,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for a full pairwise per-metric duel evaluation.

    Emits: 'start' once, 'turn' after each turn, 'verdict' at the end.
    Verdict: challenger crowned iff (challenger_score - king_score >= win_margin)
    AND bootstrap LCB on per-turn deltas > 0.
    """
    n_samples = len(samples)
    n_judges  = len(judge_models)

    yield _sse("start", {"eval_id": eval_id, "n_samples": n_samples, "n_judges": n_judges})

    # Resolve model names once — not per turn — to avoid 2N extra HTTP calls.
    king_model_name, chal_model_name = await resolve_model_names(king_client, chal_client)

    results: list[TurnResult] = []
    vllm_errors: int = 0
    king_vllm_errors: int = 0
    chal_vllm_errors: int = 0
    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_one(sample: Sample) -> TurnResult | None:
        async with semaphore:
            try:
                return await score_turn(
                    sample,
                    king_client=king_client,
                    chal_client=chal_client,
                    king_model_name=king_model_name,
                    chal_model_name=chal_model_name,
                    judge=judge,
                    judge_models=judge_models,
                    hotkey=hotkey,
                    seed=seed,
                    sink=sink,
                )
            except Exception as exc:
                log.error("score_turn failed for sample %d: %s",
                          sample.global_idx, exc, exc_info=True)
                return None

    tasks = [asyncio.create_task(_run_one(s)) for s in samples]
    for coro in asyncio.as_completed(tasks):
        result: TurnResult | None = await coro
        if result is None:
            vllm_errors += 1
            chal_vllm_errors += 1
            continue
        if result.vllm_error:
            vllm_errors += 1
            if result.vllm_error.startswith("king_"):
                king_vllm_errors += 1
            else:
                chal_vllm_errors += 1
            continue

        results.append(result)

        turn_data = {
            "eval_id":          eval_id,
            "global_idx":       result.sample.global_idx,
            "instance_id":      result.sample.instance_id,
            "turn_idx":         result.sample.turn_idx,
            "final_score":      result.final_score,
            "final_score_100":  result.final_score_100,
            "delta_avg":        result.delta,  # kept for scripts/collect_traces & inspect_dataset
            "parse_ok":         result.parse_ok,
            "vllm_error":       result.vllm_error,
            "king_vllm_errors": king_vllm_errors,
            "chal_vllm_errors": chal_vllm_errors,
            "per_judge":        result.per_judge,
            "king_usage":       result.king_usage,
            "chal_usage":       result.chal_usage,
        }
        yield _sse("turn", turn_data)

    n_done = len(results)

    # Retry any turns that weren't fully scored (judge parse failures).
    # Runs after ALL turns complete so nothing blocks the initial parallel pass.
    unscored = [r for r in results if not r.parse_ok]
    if unscored:
        log.info(
            "Retrying judge scoring for %d unscored turn(s) (of %d total)",
            len(unscored), n_done,
        )
        yield _sse("retry", {
            "eval_id":          eval_id,
            "n_retrying":       len(unscored),
            "retry_global_idx": [r.sample.global_idx for r in unscored],
        })
        await asyncio.gather(*[
            _rescore_failed_judges(
                r,
                judge=judge,
                judge_models=judge_models,
                hotkey=hotkey,
                seed=seed,
                sink=sink,
            )
            for r in unscored
        ])
        still_unscored = sum(1 for r in results if not r.parse_ok)
        log.info(
            "After retry: %d/%d turns fully scored (%d still unscored)",
            n_done - still_unscored, n_done, still_unscored,
        )

    n_valid = sum(1 for r in results if r.parse_ok)

    # Guard: if too few turns parsed, the duel is statistically unreliable.
    # This fires after the retry pass so transient judge failures don't count against us.
    if n_done > 0 and (n_valid / n_done) < DUEL_MIN_VALID_TURN_FRAC:
        log.warning(
            "duel %s: only %d/%d turns scored (< %.0f%% threshold) — rejecting",
            eval_id, n_valid, n_done, DUEL_MIN_VALID_TURN_FRAC * 100,
        )
        verdict_data = {
            "eval_id":     eval_id,
            "accepted":    False,
            "n_done":      n_done,
            "n_valid":     n_valid,
            "vllm_errors": vllm_errors,
            "error":       (
                f"min_valid_frac: only {n_valid}/{n_done} turns scored "
                f"(< {DUEL_MIN_VALID_TURN_FRAC:.0%} threshold)"
            ),
            "judge_models": judge_models,
        }
        if sink is not None:
            try:
                sink.set_scores(verdict_data)
            except Exception:
                pass
        yield _sse("verdict", verdict_data)
        return

    # Aggregate metric-first: per judge mean each metric over tasks → mean the 5 →
    # judge score; then mean across judges.
    results=[r for r in results if r.parse_ok]
    per_judge_metric = _collect_aggregation_inputs(results, judge_models)
    challenger_score, king_score, by_judge, by_metric, winner = aggregate_duel(
        per_judge_metric
    )

    # Acceptance gate 1: challenger must beat king by at least DUEL_WIN_MARGIN points.
    margin_ok = (challenger_score - king_score) >= DUEL_WIN_MARGIN

    # Acceptance gate 2 (safety): bootstrap LCB on per-turn deltas must be > 0.
    deltas: list[float] = [r.delta for r in results]
    mean_delta, lcb, se = paired_bootstrap_lcb(
        deltas,
        resamples=DUEL_RESAMPLES,
        alpha=DUEL_GATE_ALPHA,
        rng_seed=seed,
    )
    gate_lcb = lcb > 0.0

    accepted = bool(margin_ok and gate_lcb)

    verdict_data = {
        "eval_id":          eval_id,
        "accepted":         accepted,
        "n_done":           n_done,
        "n_valid":          n_valid,
        "vllm_errors":      vllm_errors,
        "challenger_score": challenger_score,
        "king_score":       king_score,
        "winner":           winner,
        "by_judge":         by_judge,
        "by_metric":        by_metric,
        "win_margin":       DUEL_WIN_MARGIN,
        "margin_ok":        margin_ok,
        "mean_delta":       mean_delta,
        "lcb":              lcb,
        "se":               se,
        "gate_alpha":       DUEL_GATE_ALPHA,
        "gate_lcb":         gate_lcb,
        "judge_models":     judge_models,
    }

    if sink is not None:
        try:
            sink.set_scores(verdict_data)
        except Exception:
            log.warning("sink.set_scores failed for eval %r", eval_id)

    yield _sse("verdict", verdict_data)
