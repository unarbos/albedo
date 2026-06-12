"""Sanity LLM gate - per-sample injection (with re-check) + viability across the judge panel."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from sanity_service.judge_panel import query_panel
from sanity_service.rubric import (
    INJECTION_SYSTEM,
    VIABILITY_SYSTEM,
    build_injection_user,
    build_viability_user,
    parse_injection,
    parse_viability,
)

# Need at least this many judges with a usable verdict to decide; otherwise defer as infra.
_MIN_RESOLVED = 2


class LLMGate(StrEnum):
    # Outcome of the gate, surfaced in the result and the sanity_results cache.
    NOT_RUN   = "not_run"    # gate never reached (e.g. heuristics failed first)
    DISABLED  = "disabled"   # gate intentionally off
    PASSED    = "passed"
    FAILED    = "failed"     # not viable (veto/consensus)
    INJECTION = "injection"  # confirmed prompt-injection (terminal miner fault)
    SKIPPED   = "skipped"    # judges unavailable -> infra defer
    CACHED    = "cached"     # served from the result cache


@dataclass
class JudgeVote:
    # One judge's verdict on one probe; None flags mean the judge gave no usable answer.
    model:        str
    injection:    bool | None = None
    inj_evidence: str = ""
    viable:       bool | None = None
    via_reason:   str = ""
    error:        str | None = None


@dataclass
class SampleVerdict:
    # Combined per-sample outcome after injection + viability.
    prompt_excerpt: str
    passed:         bool
    reason:         str
    injection:      bool = False
    infra:          bool = False
    rechecked:      bool = False
    votes:          list[JudgeVote] = field(default_factory=list)


@dataclass
class SampleInput:
    # A sampled prompt plus the challenger's generated response and its heuristic pre-filter result.
    prompt:           str
    response:         str
    heuristic_passed: bool = True
    heuristic_reason: str = ""


@dataclass
class GateResult:
    # Aggregate gate decision across all samples.
    passed:        bool
    reason:        str
    infra_fault:   bool
    llm_gate:      LLMGate
    decision_mode: str
    per_sample:    list[SampleVerdict] = field(default_factory=list)


# ── Probes ────────────────────────────────────────────────────────────────────

async def _injection_probe(
    client, prompt: str, response: str
) -> tuple[bool | None, list[JudgeVote]]:
    # Returns (suspected, votes); suspected is None when too few judges resolved (treat as infra).
    raws = await query_panel(client, INJECTION_SYSTEM, build_injection_user(prompt, response))
    votes: list[JudgeVote] = []
    for r in raws:
        flag, evidence = (None, "") if r.error is not None else parse_injection(r.raw)
        votes.append(JudgeVote(model=r.model, injection=flag, inj_evidence=evidence, error=r.error))
    resolved = [v for v in votes if v.injection is not None]
    if len(resolved) < _MIN_RESOLVED:
        return None, votes
    return any(v.injection for v in resolved), votes


async def _viability_probe(
    client, prompt: str, response: str, consensus: bool
) -> tuple[bool | None, str, list[JudgeVote]]:
    # Returns (passed, reason, votes); passed is None when too few judges resolved (treat as infra).
    raws = await query_panel(client, VIABILITY_SYSTEM, build_viability_user(prompt, response))
    votes: list[JudgeVote] = []
    for r in raws:
        flag, reason = (None, "") if r.error is not None else parse_viability(r.raw)
        votes.append(JudgeVote(model=r.model, viable=flag, via_reason=reason, error=r.error))
    resolved = [v for v in votes if v.viable is not None]
    if len(resolved) < _MIN_RESOLVED:
        return None, "viability judges unavailable", votes
    yes = sum(1 for v in resolved if v.viable)
    # consensus = majority of resolved judges; veto (default) = unanimity, any False vetoes.
    passed = yes > len(resolved) / 2 if consensus else yes == len(resolved)
    if passed:
        return True, "", votes
    nay = next((v.via_reason for v in resolved if not v.viable), "")
    return False, f"not viable: {nay}".strip()[:200], votes


async def _judge_sample(s: SampleInput, client, consensus: bool) -> SampleVerdict:
    # Runs the per-sample flow: heuristics -> injection (+re-check) -> viability.
    excerpt = (s.prompt or "")[:60]
    if not s.heuristic_passed:
        return SampleVerdict(excerpt, passed=False, reason=f"heuristic: {s.heuristic_reason}")

    suspected, votes = await _injection_probe(client, s.prompt, s.response)
    if suspected is None:
        return SampleVerdict(excerpt, False, "injection judges unavailable",
                             infra=True, votes=votes)

    rechecked = False
    if suspected:
        rechecked = True
        confirmed, votes = await _injection_probe(client, s.prompt, s.response)
        if confirmed is None:
            return SampleVerdict(excerpt, False, "injection judges unavailable", infra=True,
                                 rechecked=True, votes=votes)
        if confirmed:
            evidence = next((v.inj_evidence for v in votes if v.injection), "")
            return SampleVerdict(excerpt, False, f"injection: {evidence}".strip()[:200],
                                 injection=True, rechecked=True, votes=votes)
        # Re-check came back clean - the first flag was a false positive; continue to viability.

    decided, reason, vvotes = await _viability_probe(client, s.prompt, s.response, consensus)
    if decided is None:
        return SampleVerdict(excerpt, False, reason, infra=True, rechecked=rechecked, votes=vvotes)
    return SampleVerdict(excerpt, passed=decided, reason=reason, rechecked=rechecked, votes=vvotes)


# ── Aggregation ─────────────────────────────────────────────────────────────────

def _aggregate(verdicts: list[SampleVerdict], mode: str) -> GateResult:
    # Combines per-sample verdicts with priority injection > infra > viability-fail > pass.
    injected = next((v for v in verdicts if v.injection), None)
    if injected is not None:
        return GateResult(False, injected.reason, infra_fault=False, llm_gate=LLMGate.INJECTION,
                          decision_mode=mode, per_sample=verdicts)
    if any(v.infra for v in verdicts):
        return GateResult(False, "judges unavailable", infra_fault=True, llm_gate=LLMGate.SKIPPED,
                          decision_mode=mode, per_sample=verdicts)
    failed = next((v for v in verdicts if not v.passed), None)
    if failed is not None:
        return GateResult(False, failed.reason, infra_fault=False, llm_gate=LLMGate.FAILED,
                          decision_mode=mode, per_sample=verdicts)
    return GateResult(True, "", infra_fault=False, llm_gate=LLMGate.PASSED,
                      decision_mode=mode, per_sample=verdicts)


async def run_gate(samples: list[SampleInput], client, *, consensus: bool = False) -> GateResult:
    # Judges every sample concurrently and returns the aggregate gate decision.
    mode = "consensus" if consensus else "veto"
    if not samples:
        return GateResult(False, "no samples", infra_fault=True,
                          llm_gate=LLMGate.SKIPPED, decision_mode=mode)
    verdicts = list(await asyncio.gather(*[_judge_sample(s, client, consensus) for s in samples]))
    result = _aggregate(verdicts, mode)
    (logger.info if result.passed else logger.warning)(
        "[sanity/gate] passed={} gate={} mode={} reason={!r}",
        result.passed, result.llm_gate, mode, result.reason,
    )
    return result
