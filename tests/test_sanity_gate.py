"""Tests for the sanity pre-eval gate, dispatcher decisions, and worker run store."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from albedo_eval_service.judge_openrouter import JudgeRawResponse
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_remote.worker import _model_ref_parts
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

    async def complete(self, *, model, messages, temperature=None):
        if "security auditor" in messages[0]["content"]:
            self._n += 1
            flag = self._inj2 if self._n > 3 else self._inj1
            raw = (
                '{"injection": true, "evidence": "x"}'
                if flag
                else '{"injection": false, "evidence": "none"}'
            )
        else:
            raw = (
                '{"viable": true, "reason": "ok"}'
                if self._viable
                else '{"viable": false, "reason": "bad"}'
            )
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

    async def _fake_gate(samples, client, *, consensus=False, skip_viability=False, models=None):
        return gate

    D.run_gate = _fake_gate
    asyncio.run(
        disp._complete(
            submission_id=uuid4(),
            attempt_id=uuid4(),
            repo="r",
            digest="d",
            prompts=["p", "p", "p"],
            result=result,
        )
    )
    return repo.calls


_OK = {
    "state": "succeeded",
    "responses": ["a", "b", "c"],
    "heuristics": [{"passed": True, "reason": "ok"}] * 3,
}


def test_complete_passed():
    assert _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), _OK) == [
        ("passed", None)
    ]


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


# ── import hygiene ──────────────────────────────────────────────────────────────


def test_dispatcher_binds_canonical_repository():
    # Guards against re-introducing `from src.sanity_service.db`, which loads db.py a second
    # time under the `src.` namespace and yields a distinct PreEvalRepository class.
    import sanity_service.db as canonical

    assert D.PreEvalRepository is canonical.PreEvalRepository
    assert D.ClaimedPreEval is canonical.ClaimedPreEval


# ── worker run store ────────────────────────────────────────────────────────────


def test_model_ref_parts_accepts_chain_model_uri():
    digest = "sha256:" + "a" * 64
    assert _model_ref_parts("alice/model@" + digest, "") == ("alice/model", digest)
    assert _model_ref_parts("alice/model", digest) == ("alice/model", digest)


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


# ── sanity remote worker fixes ────────────────────────────────────────────────

from sanity_remote.config import SanityRemoteSettings
from sanity_remote.worker import VllmEngine, _strip_model_config


def test_max_model_len_default_matches_eval_context(monkeypatch):
    from albedo_eval_service.canonical_model_config import canonical_max_model_len

    monkeypatch.delenv("SANITY_REMOTE_MAX_MODEL_LEN", raising=False)
    assert SanityRemoteSettings(_env_file=None).max_model_len == canonical_max_model_len()


def test_strip_model_config_removes_forbidden_keys(tmp_path):
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": 5120,
        "auto_map": {"AutoModelForCausalLM": "modeling_qwen.Qwen3_5ForConditionalGeneration"},
        "quantization_config": {"quant_type": "gptq", "bits": 4},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    _strip_model_config(str(tmp_path))

    result = json.loads((tmp_path / "config.json").read_text())
    assert "auto_map" not in result
    assert "quantization_config" not in result
    assert result["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert result["hidden_size"] == 5120


def test_strip_model_config_noop_when_clean(tmp_path):
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "hidden_size": 5120}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    mtime_before = config_path.stat().st_mtime

    _strip_model_config(str(tmp_path))

    assert config_path.stat().st_mtime == mtime_before


def test_strip_model_config_tolerates_missing_config(tmp_path):
    _strip_model_config(str(tmp_path))  # must not raise


def test_vllm_cmd_includes_generation_config_vllm(tmp_path):
    settings = SanityRemoteSettings(vllm_port=19999)
    engine = VllmEngine(settings)

    captured: list[str] = []

    async def _run():
        with (
            patch("subprocess.Popen") as mock_popen,
            patch.object(engine, "_wait_healthy", return_value=None),
        ):
            mock_popen.return_value = MagicMock()
            await engine._start_vllm(str(tmp_path), "sha256:abc")
            captured.extend(mock_popen.call_args[0][0])

    asyncio.run(_run())
    idx = captured.index("--generation-config")
    assert captured[idx + 1] == "vllm"
