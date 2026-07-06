"""Sanity LLM gate - per-sample injection (with re-check) + viability across the judge panel."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from albedo_eval_service.judge_core import JUDGE_MODELS
from sanity_service.judge_panel import query_panel
from sanity_service.rubric import (
    INJECTION_SYSTEM,
    VIABILITY_SYSTEM,
    build_injection_user,
    build_viability_user,
    parse_injection,
    parse_viability,
)

# Minimum resolved votes needed to decide; capped at the panel size so a single-model config works.
_MIN_RESOLVED = 2
_RAW_LOG_CHARS = 240
_ERR_LOG_CHARS = 400
# The injection re-check runs at a higher temperature so it is a genuine independent re-sample, not a
# deterministic repeat of the first probe (at temp 0.0 the re-check always confirms - even a spurious
# flag). A real injection stays flagged under variance; a borderline false positive can clear.
_RECHECK_TEMPERATURE = float(os.environ.get("SANITY_INJECTION_RECHECK_TEMPERATURE", "0.2"))


def _short(value: str | None, limit: int) -> str:
    # Truncates and flattens a string for structured log fields.
    return (value or "").replace("\n", "\\n")[:limit]


def _resolve_models(override: tuple[str, ...] | None) -> tuple[str, ...]:
    # Returns override if given; reads SANITY_JUDGE_MODELS from env next; falls back to eval defaults.
    if override:
        return override
    env = os.environ.get("SANITY_JUDGE_MODELS", "").strip()
    if env:
        return tuple(m.strip() for m in env.split(",") if m.strip())
    return JUDGE_MODELS


class LLMGate(StrEnum):
    # Outcome of the gate, surfaced in the result and the sanity_results cache.
    PASSED = "passed"
    FAILED = "failed"  # not viable (veto/consensus)
    INJECTION = "injection"  # confirmed prompt-injection (terminal miner fault)
    SKIPPED = "skipped"  # judges unavailable -> infra defer


@dataclass
class JudgeVote:
    # One judge's verdict on one probe; None flags mean the judge gave no usable answer.
    model: str
    injection: bool | None = None
    inj_evidence: str = ""
    viable: bool | None = None
    via_reason: str = ""
    error: str | None = None


@dataclass
class SampleVerdict:
    # Combined per-sample outcome after injection + viability.
    prompt_excerpt: str
    passed: bool
    reason: str
    injection: bool = False
    infra: bool = False
    rechecked: bool = False
    votes: list[JudgeVote] = field(default_factory=list)


@dataclass
class SampleInput:
    # A sampled prompt plus the challenger's generated response and its heuristic pre-filter result.
    prompt: str
    response: str
    heuristic_passed: bool = True
    heuristic_reason: str = ""


@dataclass
class GateResult:
    # Aggregate gate decision across all samples.
    passed: bool
    reason: str
    infra_fault: bool
    llm_gate: LLMGate
    decision_mode: str
    per_sample: list[SampleVerdict] = field(default_factory=list)


# ── Probes ────────────────────────────────────────────────────────────────────


async def _injection_probe(client, prompt: str, response: str, models: tuple[str, ...] = JUDGE_MODELS, temperature: float | None = None) -> tuple[bool | None, list[JudgeVote]]:
    # Returns (suspected, votes); suspected is None when too few judges resolved (treat as infra).
    raws = await query_panel(client, INJECTION_SYSTEM, build_injection_user(prompt, response), models, temperature=temperature)
    votes: list[JudgeVote] = []
    details: list[dict[str, object]] = []
    for r in raws:
        flag, evidence = (None, "") if r.error is not None else parse_injection(r.raw)
        votes.append(JudgeVote(model=r.model, injection=flag, inj_evidence=evidence, error=r.error))
        details.append(
            {
                "model": r.model,
                "resolved": flag is not None,
                "injection": flag,
                "evidence": _short(evidence, _RAW_LOG_CHARS),
                "error": _short(r.error, _ERR_LOG_CHARS),
                "raw_prefix": "" if r.error is not None else _short(r.raw, _RAW_LOG_CHARS),
            }
        )
    resolved = [v for v in votes if v.injection is not None]
    if len(resolved) < min(_MIN_RESOLVED, len(models)):
        logger.warning(
            "[sanity/gate] injection judges unavailable resolved={}/{} response_chars={} "
            "prompt_prefix={!r} details={}",
            len(resolved),
            len(votes),
            len(response or ""),
            _short(prompt, 120),
            details,
        )
        return None, votes
    return any(v.injection for v in resolved), votes


async def _viability_probe(client, prompt: str, response: str, consensus: bool, models: tuple[str, ...] = JUDGE_MODELS) -> tuple[bool | None, str, list[JudgeVote]]:
    # Returns (passed, reason, votes); passed is None when too few judges resolved (treat as infra).
    raws = await query_panel(client, VIABILITY_SYSTEM, build_viability_user(prompt, response), models)
    votes: list[JudgeVote] = []
    details: list[dict[str, object]] = []
    for r in raws:
        flag, reason = (None, "") if r.error is not None else parse_viability(r.raw)
        votes.append(JudgeVote(model=r.model, viable=flag, via_reason=reason, error=r.error))
        details.append(
            {
                "model": r.model,
                "resolved": flag is not None,
                "viable": flag,
                "reason": _short(reason, _RAW_LOG_CHARS),
                "error": _short(r.error, _ERR_LOG_CHARS),
                "raw_prefix": "" if r.error is not None else _short(r.raw, _RAW_LOG_CHARS),
            }
        )
    resolved = [v for v in votes if v.viable is not None]
    if len(resolved) < min(_MIN_RESOLVED, len(models)):
        logger.warning(
            "[sanity/gate] viability judges unavailable resolved={}/{} response_chars={} "
            "prompt_prefix={!r} details={}",
            len(resolved),
            len(votes),
            len(response or ""),
            _short(prompt, 120),
            details,
        )
        return None, "viability judges unavailable", votes
    yes = sum(1 for v in resolved if v.viable)
    # consensus = majority of resolved judges; veto (default) = unanimity, any False vetoes.
    passed = yes > len(resolved) / 2 if consensus else yes == len(resolved)
    if passed:
        return True, "", votes
    nay = next((v.via_reason for v in resolved if not v.viable), "")
    return False, f"not viable: {nay}".strip()[:200], votes


async def _judge_sample(s: SampleInput, client, consensus: bool, skip_viability: bool = False, models: tuple[str, ...] = JUDGE_MODELS,) -> SampleVerdict:
    # Runs the per-sample flow: heuristics -> injection (+re-check) -> viability.
    excerpt = (s.prompt or "")[:60]
    if not s.heuristic_passed:
        return SampleVerdict(excerpt, passed=False, reason=f"heuristic: {s.heuristic_reason}")

    suspected, votes = await _injection_probe(client, s.prompt, s.response, models)
    if suspected is None:
        return SampleVerdict(excerpt, False, "injection judges unavailable", infra=True, votes=votes)

    rechecked = False
    if suspected:
        rechecked = True
        # Re-check at a higher temperature (genuine re-sample) so a spurious first flag can clear.
        confirmed, votes = await _injection_probe(client, s.prompt, s.response, models, temperature=_RECHECK_TEMPERATURE)
        if confirmed is None:
            return SampleVerdict(
                excerpt,
                False,
                "injection judges unavailable",
                infra=True,
                rechecked=True,
                votes=votes,
            )
        if confirmed:
            evidence = next((v.inj_evidence for v in votes if v.injection), "")
            return SampleVerdict(
                excerpt,
                False,
                f"injection: {evidence}".strip()[:200],
                injection=True,
                rechecked=True,
                votes=votes,
            )
        # Re-check came back clean - the first flag was a false positive; continue to viability.

    if skip_viability:
        return SampleVerdict(excerpt, passed=True, reason="viability skipped", rechecked=rechecked)

    decided, reason, vvotes = await _viability_probe(client, s.prompt, s.response, consensus, models)
    if decided is None:
        return SampleVerdict(excerpt, False, reason, infra=True, rechecked=rechecked, votes=vvotes)
    return SampleVerdict(excerpt, passed=decided, reason=reason, rechecked=rechecked, votes=vvotes)


# ── Aggregation ─────────────────────────────────────────────────────────────────


def _aggregate(verdicts: list[SampleVerdict], mode: str) -> GateResult:
    # Combines per-sample verdicts with priority injection > infra > viability-fail > pass.
    injected = next((v for v in verdicts if v.injection), None)
    if injected is not None:
        return GateResult(
            False,
            injected.reason,
            infra_fault=False,
            llm_gate=LLMGate.INJECTION,
            decision_mode=mode,
            per_sample=verdicts,
        )
    if any(v.infra for v in verdicts):
        return GateResult(
            False,
            "judges unavailable",
            infra_fault=True,
            llm_gate=LLMGate.SKIPPED,
            decision_mode=mode,
            per_sample=verdicts,
        )
    failed = next((v for v in verdicts if not v.passed), None)
    if failed is not None:
        return GateResult(
            False,
            failed.reason,
            infra_fault=False,
            llm_gate=LLMGate.FAILED,
            decision_mode=mode,
            per_sample=verdicts,
        )
    return GateResult(
        True,
        "",
        infra_fault=False,
        llm_gate=LLMGate.PASSED,
        decision_mode=mode,
        per_sample=verdicts,
    )


async def run_gate(samples: list[SampleInput], client, *, consensus: bool = False, skip_viability: bool = False, models: tuple[str, ...] | None = None,) -> GateResult:
    # Judges every sample concurrently and returns the aggregate gate decision.
    # models=None reads SANITY_JUDGE_MODELS from env, falling back to the eval judge defaults.
    mode = "consensus" if consensus else "veto"
    resolved = _resolve_models(models)
    if not samples:
        return GateResult(False, "no samples", infra_fault=True, llm_gate=LLMGate.SKIPPED, decision_mode=mode)
    verdicts = list(await asyncio.gather(*[_judge_sample(s, client, consensus, skip_viability, resolved) for s in samples]))
    result = _aggregate(verdicts, mode)
    (logger.info if result.passed else logger.warning)(
        "[sanity/gate] passed={} gate={} mode={} reason={!r}",
        result.passed,
        result.llm_gate,
        mode,
        result.reason,
    )
    return result
