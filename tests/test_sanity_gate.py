"""Tests for the sanity pre-eval gate, dispatcher decisions, and worker run store."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from albedo_eval_service.judge_openrouter import JudgeRawResponse
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_service import dispatcher as D
from sanity_service.llm_check import GateResult, LLMGate, SampleInput, run_gate
from sanity_service.rubric import parse_injection, parse_viability
from sanity_service.settings import SanitySettings


# ── rubric parsers ────────────────────────────────────────────────────────────

def test_parse_injection_and_viability():
    assert parse_injection('{"injection": false, "evidence": "none"}') == (False, "none")
    assert parse_injection('{"injection": true, "evidence": "x"}')[0] is True
    assert parse_viability('{"viable": true, "reason": "ok"}') == (True, "ok")
    assert parse_viability("not json")[0] is None


# ── gate via a fake judge client (calls 1-3 = first injection pass, 4-6 = re-check) ─────

class _FakeJudge:
    def __init__(self, inj1: bool, inj2: bool, viable: bool) -> None:
        self._inj1, self._inj2, self._viable, self._n = inj1, inj2, viable, 0

    async def complete(self, *, model, messages):
        if "security auditor" in messages[0]["content"]:
            self._n += 1
            flag = self._inj2 if self._n > 3 else self._inj1
            raw = '{"injection": true, "evidence": "x"}' if flag else '{"injection": false, "evidence": "none"}'
        else:
            raw = '{"viable": true, "reason": "ok"}' if self._viable else '{"viable": false, "reason": "bad"}'
        return JudgeRawResponse(model=model, provider="p", raw=raw)


def _samples(n: int = 3) -> list[SampleInput]:
    return [SampleInput(prompt=f"task {i}", response="def f(): return 1") for i in range(n)]


def test_gate_pass():
    gate = asyncio.run(run_gate(_samples(), _FakeJudge(False, False, True)))
    assert gate.passed and gate.llm_gate == LLMGate.PASSED


def test_gate_injection_confirmed():
    gate = asyncio.run(run_gate(_samples(1), _FakeJudge(True, True, True)))
    assert not gate.passed and gate.llm_gate == LLMGate.INJECTION


def test_gate_injection_false_positive_recovers():
    gate = asyncio.run(run_gate(_samples(1), _FakeJudge(True, False, True)))
    assert gate.passed and gate.llm_gate == LLMGate.PASSED


def test_gate_not_viable():
    gate = asyncio.run(run_gate(_samples(), _FakeJudge(False, False, False)))
    assert not gate.passed and gate.llm_gate == LLMGate.FAILED


def test_gate_heuristic_fail_skips_judges():
    samples = [SampleInput("t", "x", heuristic_passed=False, heuristic_reason="empty")]
    gate = asyncio.run(run_gate(samples, _FakeJudge(False, False, True)))
    assert not gate.passed and gate.llm_gate == LLMGate.FAILED


# ── dispatcher _complete decisions ──────────────────────────────────────────────

class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list = []

    def mark_pre_eval_passed(self, **kw):
        self.calls.append(("passed", None))

    def mark_pre_eval_failed(self, **kw):
        self.calls.append(("failed", kw["retryable"], kw["fault_class"]))


class _DummyJudge:
    async def aclose(self):
        pass


def _complete(gate: GateResult, result: dict) -> list:
    repo = _FakeRepo()
    disp = D.SanityDispatcher(settings=SanitySettings(), repository=repo)
    D.make_client = lambda: _DummyJudge()

    async def _fake_gate(samples, client, *, consensus=False):
        return gate

    D.run_gate = _fake_gate
    asyncio.run(
        disp._complete(
            submission_id=uuid4(), attempt_id=uuid4(), repo="r", digest="d",
            prompts=["p", "p", "p"], result=result,
        )
    )
    return repo.calls


_OK = {"state": "succeeded", "responses": ["a", "b", "c"], "heuristics": [{"passed": True, "reason": "ok"}] * 3}


def test_complete_passed():
    assert _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), _OK) == [("passed", None)]


def test_complete_injection_is_terminal_miner_fault():
    calls = _complete(GateResult(False, "i", False, LLMGate.INJECTION, "veto", []), _OK)
    assert calls == [("failed", False, "MINER_FAULT")]


def test_complete_infra_is_retryable():
    calls = _complete(GateResult(False, "d", True, LLMGate.SKIPPED, "veto", []), _OK)
    assert calls == [("failed", True, "INFRA_FAULT")]


def test_complete_worker_failure_is_retryable():
    fail = {"state": "failed", "fault_code": "x", "fault_message": "m", "retryable": True}
    calls = _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), fail)
    assert calls == [("failed", True, "INFRA_FAULT")]


# ── worker run store ────────────────────────────────────────────────────────────

def test_worker_store_lifecycle():
    store = SanityRunStore()
    req = SanityRunRequest(run_id="r1", model_uri="m", digest="d", prompts=["a", "b", "c"])
    run = store.start(req)
    assert store.start(req) is run  # idempotent on run_id
    assert store.mark_worker_started("r1").state == "queued"
    assert store.mark_worker_started("r1") is None  # only once
    run.succeed(responses=["x"], heuristics=[{"passed": True, "reason": "ok"}])
    assert run.as_status()["state"] == "succeeded"
    assert store.list_active() == []
