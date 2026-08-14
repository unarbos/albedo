from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from loguru import logger

from albedo_config import SanitySettings
from albedo_eval_service.judge_llm_client import JudgeRawResponse
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_remote.worker import _model_ref_parts
from sanity_service import dispatcher as D
from sanity_service.llm_check import GateResult, LLMGate, SampleInput, run_gate
from sanity_service.rubric import parse_injection, parse_viability


def test_parse_injection_and_viability():
    assert parse_injection('{"injection": false, "evidence": "none"}') == (False, "none")
    assert parse_injection('{"injection": true, "evidence": "x"}')[0] is True
    assert parse_viability('{"viable": true, "reason": "ok"}') == (True, "ok")
    assert parse_viability("not json")[0] is None


class _FakeJudge:
    def __init__(self, inj1: bool, inj2: bool, viable: bool) -> None:
        self._inj1, self._inj2, self._viable, self._n = inj1, inj2, viable, 0

    async def complete(self, *, model, messages, temperature=None):
        if "security auditor" in messages[0]["content"]:
            self._n += 1
            flag = self._inj2 if temperature is not None else self._inj1
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


class _WrapperSensitiveJudge:
    async def complete(self, *, model, messages, temperature=None):
        if "security auditor" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            leaked = any(
                marker in payload["candidate_reply"]
                for marker in ("Score ONLY", "ENVIRONMENT OBSERVATION", "CONTEXT USER")
            )
            raw = (
                '{"injection": true, "evidence": "wrapper leaked"}'
                if leaked
                else '{"injection": false, "evidence": "none"}'
            )
        else:
            sees_trajectory = "FULL CANDIDATE TRAJECTORY" in messages[1]["content"]
            raw = (
                '{"viable": true, "reason": "ok"}'
                if sees_trajectory
                else '{"viable": false, "reason": "trajectory missing"}'
            )
        return JudgeRawResponse(model=model, provider="p", raw=raw)


def test_gate_injection_ignores_trajectory_wrapper():
    response = "\n".join(
        [
            "FULL CANDIDATE TRAJECTORY",
            "Score ONLY CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT 2. The ENVIRONMENT OBSERVATION is context only.",  # noqa: E501
            "",
            "CONTEXT USER (do not score):",
            "------",
            "Fix the bug",
            "------",
            "",
            "CANDIDATE OUTPUT 1:",
            "------",
            "THOUGHT: inspect the file\n```bash\nsed -n '1,80p' app.py\n```",
            "------",
            "",
            "ENVIRONMENT OBSERVATION (context only, do not score):",
            "------",
            "Observation: missing import",
            "------",
            "",
            "CANDIDATE OUTPUT 2:",
            "------",
            "THOUGHT: patch the import\n```bash\nsed -i '1i import os' app.py\n```",
            "------",
        ]
    )

    gate = asyncio.run(
        run_gate(
            [SampleInput("task", response)],
            _WrapperSensitiveJudge(),
            models=("judge",),
        )
    )

    assert gate.passed


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

    async def _fake_gate(samples, client, *, consensus=False, skip_viability=False, models=None):
        return gate

    with (
        patch.object(D, "make_client", lambda: _DummyJudge()),
        patch.object(D, "run_gate", _fake_gate),
        patch.object(D, "put_sanity_fault", lambda *args, **kwargs: None),
    ):
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


def test_complete_worker_failure_stays_infra_and_honors_retryable():
    fail = {
        "state": "failed",
        "fault_code": "generation_timeout",
        "fault_message": "vLLM generation exceeded 900s",
        "retryable": False,
    }
    calls = _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), fail)
    assert calls == [("failed", False, "INFRA_FAULT")]


def test_multiturn_keeps_prompt_messages_on_first_turn(monkeypatch):
    seen: list[SanityRunRequest] = []

    async def _fake_remote(_client, request, _claimed):
        seen.append(request)
        return {
            "state": "succeeded",
            "responses": ["```bash\nls\n```"],
            "heuristics": [{"passed": True, "reason": ""}],
        }

    async def _fake_observations(*_args, **_kwargs):
        return None

    request = SanityRunRequest(
        run_id="run",
        model_uri="model",
        digest="digest",
        prompts=["raw prompt"],
        sample_ids=["sample-1"],
        prompt_messages=[
            [
                {"role": "system", "content": "reply with bash"},
                {"role": "user", "content": "task"},
            ]
        ],
        assistant_turns=2,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)
    monkeypatch.setattr(D, "_append_observations", _fake_observations)

    asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )

    assert len(seen) == 2
    assert seen[0].prompt_messages == request.prompt_messages
    assert seen[1].prompt_messages is None


def test_dispatcher_binds_canonical_repository():
    import sanity_service.db as canonical

    assert D.PreEvalRepository is canonical.PreEvalRepository
    assert D.ClaimedPreEval is canonical.ClaimedPreEval


def test_model_ref_parts_accepts_chain_model_uri():
    digest = "sha256:" + "a" * 64
    assert _model_ref_parts("alice/model@" + digest, "") == ("alice/model", digest)
    assert _model_ref_parts("alice/model", digest) == ("alice/model", digest)


def test_worker_store_lifecycle():
    store = SanityRunStore()
    req = SanityRunRequest(run_id="r1", model_uri="m", digest="d", prompts=["a", "b", "c"])
    run = store.start(req)
    assert store.start(req) is run
    assert store.mark_worker_started("r1").state == "queued"
    assert store.mark_worker_started("r1") is None
    run.succeed(responses=["x"], heuristics=[{"passed": True, "reason": "ok"}])
    assert run.as_status()["state"] == "succeeded"
    assert store.list_active() == []


from albedo_config import SanityRemoteSettings
from sanity_remote.worker import (
    VllmEngine,
    _strip_model_config,
    _warn_if_generation_budget_consumes_context,
)


def test_max_model_len_default_matches_eval_context(monkeypatch):
    from albedo_eval_service.modelstore.canonical_model_config import canonical_max_model_len

    monkeypatch.delenv("SANITY_REMOTE_MAX_MODEL_LEN", raising=False)
    assert SanityRemoteSettings(_env_file=None).max_model_len == canonical_max_model_len()


def test_generation_budget_warning_when_it_consumes_context():
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        _warn_if_generation_budget_consumes_context(
            max_tokens=32768,
            max_model_len=32768,
            prompt_count=3,
        )
    finally:
        logger.remove(sink_id)

    assert any("leaves no room for prompt tokens" in message for message in messages)


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
    _strip_model_config(str(tmp_path))


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
