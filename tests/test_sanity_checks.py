"""Tests for the sanity response heuristics (checks.py) and the worker's _heuristics wrapper."""

from __future__ import annotations

import asyncio

import pytest

from sanity_remote import worker as sanity_worker
from sanity_remote.models import SanityRunRequest
from sanity_remote.worker import VllmEngine, WorkerFault, _format_prompt_messages, _heuristics
from sanity_service import dispatcher as sanity_dispatcher
from sanity_service.checks import (
    check_all,
    check_code_present,
    check_collapsed,
    check_encoding,
    check_one,
    check_repetition,
    check_uniform_length,
    check_vocabulary,
)
from sanity_service.dataset import sample_prompts
from sanity_service.dispatcher import _format_scored_trajectory

# ── per-response checks ─────────────────────────────────────────────────────────


def test_check_one_passes_clean_code_answer():
    assert check_one("def add(a, b): return a + b done").passed


def test_check_one_flags_empty_then_short():
    assert check_one("").reason == "empty response"
    assert not check_one("one two").passed  # below min_tokens=5


def test_check_repetition_catches_token_loop():
    assert not check_repetition("na " * 10).passed
    assert check_repetition("the quick brown fox jumps over").passed


def test_check_encoding_catches_garbled_weights():
    assert not check_encoding("é" * 40).passed
    assert check_encoding("plain ascii sentence that is long enough").passed


def test_check_vocabulary_catches_low_variety():
    # High trigram diversity but only two unique tokens -> repetition passes, vocabulary fails.
    assert not check_vocabulary("a b a b a b a b a b").passed
    assert check_vocabulary("alpha beta gamma delta epsilon zeta eta theta").passed


# ── cross-prompt checks ─────────────────────────────────────────────────────────


def test_check_collapsed_flags_identical_responses():
    assert not check_collapsed(["same", "same", "same"]).passed
    assert check_collapsed(["one", "two", "three"]).passed
    # single response must never fail - a set of size 1 is expected for n=1
    assert check_collapsed(["any single response"]).passed


def test_check_uniform_length_flags_identical_token_counts():
    assert not check_uniform_length(["a b c", "d e f", "g h i"]).passed
    assert check_uniform_length(["a b", "c d e"]).passed
    assert check_uniform_length(["only one"]).passed  # single response cannot collapse


def test_check_code_present_requires_a_keyword_somewhere():
    assert check_code_present(["prose here", "def f(): return 1"]).passed
    assert not check_code_present(["just prose", "more prose"]).passed


def test_check_all_reports_first_failure_with_prompt_index():
    result = check_all(["def f(): return 1 ok now", "", "def g(): return 2 ok now"])
    assert not result.passed
    assert result.reason.startswith("prompt 2/3:")


# ── worker _heuristics wrapper ──────────────────────────────────────────────────


def _req() -> SanityRunRequest:
    return SanityRunRequest(run_id="r", model_uri="m", digest="d", prompts=["p"])


def test_heuristics_skip_passes_all_without_inspection():
    out = _heuristics(["", "", ""], _req(), skip=True)
    assert all(v["passed"] for v in out)
    assert out[0]["reason"] == "heuristics disabled"


def test_heuristics_set_failure_fails_every_response():
    out = _heuristics(
        ["def f(): return 1 now", "def f(): return 1 now", "def f(): return 1 now"],
        _req(),
    )
    assert not any(v["passed"] for v in out)
    assert all("collapsed" in v["reason"] for v in out)


def test_heuristics_reports_empty_before_set_collapse():
    out = _heuristics(["", "", ""], _req())
    assert not any(v["passed"] for v in out)
    assert all(v["reason"] == "empty response" for v in out)


def test_heuristics_allows_short_bash_commands():
    out = _heuristics(
        [
            "```bash\nls -la\n```",
            "```bash\ncat utils.py\n```",
            "```bash\ngrep -r \"def double\" .\n```",
        ],
        _req(),
    )

    assert all(v["passed"] for v in out), out


def test_heuristics_still_rejects_short_non_command():
    out = _heuristics(["one two"], _req())

    assert out == [{"passed": False, "reason": "too short (2 tokens, min=5)"}]


def test_fallback_prompts_carry_messages_for_worker_template():
    sample = sample_prompts(seed="seed", n=1)[0]

    assert sample.messages
    assert sample.messages[0]["role"] == "system"
    assert "fenced bash command" in sample.messages[0]["content"]
    assert sample.messages[1] == {"role": "user", "content": sample.prompt}
    assert sample.sample_id == "sanity-fallback:0"


def test_sanity_trajectory_formatter_scores_candidate_turns_only():
    text = _format_scored_trajectory(
        [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "```bash\nls\n```", "score_target": True},
            {
                "role": "user",
                "content": "Observation: src/app.py",
                "environment_observation": True,
            },
            {"role": "assistant", "content": "```bash\nsed -n '1,80p' src/app.py\n```", "score_target": True},
        ]
    )

    assert "Score ONLY CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT 2" in text
    assert "ENVIRONMENT OBSERVATION (context only, do not score)" in text


def test_pre_eval_worker_formats_messages_with_non_thinking_template(monkeypatch):
    captured = {}

    def _fake_format_messages(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "non-thinking prompt"

    monkeypatch.setattr(sanity_worker, "format_messages", _fake_format_messages)

    prompts = _format_prompt_messages("/models/qwen", [[{"role": "user", "content": "Fix it."}]])

    assert prompts == ["non-thinking prompt"]
    assert captured["messages"] == [{"role": "user", "content": "Fix it."}]
    assert captured["kwargs"] == {
        "tokenizer_path": "/models/qwen",
        "enable_thinking": False,
    }


def test_trajectory_followup_prompt_uses_non_thinking_template(monkeypatch):
    captured = {}

    class _Client:
        async def aclose(self):
            captured["closed"] = True

    async def _fake_simulate_observation(**_kwargs):
        return "Observation: src/app.py"

    def _fake_format_messages(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "next-turn prompt"

    monkeypatch.setattr(sanity_dispatcher, "get_judge_settings", lambda: object())
    monkeypatch.setattr(sanity_dispatcher, "make_client", lambda _settings: _Client())
    monkeypatch.setattr(sanity_dispatcher, "_simulate_observation", _fake_simulate_observation)
    monkeypatch.setattr(sanity_dispatcher, "format_messages", _fake_format_messages)

    state = sanity_dispatcher._TrajectoryState(
        sample_id="sanity-fallback:0",
        prompt="initial prompt",
        messages=[{"role": "user", "content": "Fix it."}],
        turns=[{"role": "assistant", "content": "```bash\nls\n```", "score_target": True}],
    )

    asyncio.run(sanity_dispatcher._append_observations([state], "run", 1))

    assert state.prompt == "next-turn prompt"
    assert captured["closed"] is True
    assert captured["messages"][-1] == {"role": "user", "content": "Observation: src/app.py"}
    assert captured["kwargs"]["enable_thinking"] is False


def test_trajectory_no_command_gets_observation_without_simulator(monkeypatch):
    def _fail_make_client(*_args, **_kwargs):
        raise AssertionError("simulator should not run without a bash command")

    monkeypatch.setattr(sanity_dispatcher, "make_client", _fail_make_client)

    state = sanity_dispatcher._TrajectoryState(
        sample_id="sanity-fallback:0",
        prompt="initial prompt",
        messages=[{"role": "user", "content": "Fix it."}],
        turns=[{"role": "assistant", "content": "Here is the answer.", "score_target": True}],
    )

    asyncio.run(sanity_dispatcher._append_observations([state], "run", 1))

    assert state.messages[-1] == {
        "role": "user",
        "content": "Observation: No bash command found in assistant message.",
    }
    assert not state.stopped


def test_trajectory_ignores_short_heuristic_for_bash_command():
    state = sanity_dispatcher._TrajectoryState(
        sample_id="sanity-fallback:0",
        prompt="initial prompt",
        messages=[{"role": "user", "content": "Fix it."}],
        turns=[],
    )

    sanity_dispatcher._apply_turn_result(
        [state],
        {
            "responses": ["```bash\nls -la\n```"],
            "heuristics": [{"passed": False, "reason": "too short (4 tokens, min=5)"}],
        },
    )

    assert state.heuristic_reason == ""


def test_trajectory_keeps_heuristic_for_non_command_output():
    state = sanity_dispatcher._TrajectoryState(
        sample_id="sanity-fallback:0",
        prompt="initial prompt",
        messages=[{"role": "user", "content": "Fix it."}],
        turns=[],
    )

    sanity_dispatcher._apply_turn_result(
        [state],
        {
            "responses": [""],
            "heuristics": [{"passed": False, "reason": "empty response"}],
        },
    )

    assert state.heuristic_reason == "empty response"


def test_heuristics_passes_varied_code_responses():
    responses = [
        "def add(a, b): return a + b here",
        "the function computes a sum of two numbers correctly",
        "import math then call the helper to finish now",
    ]
    out = _heuristics(responses, _req())
    assert all(v["passed"] for v in out), out


# ── _strip_thinking fallback (Option B) ────────────────────────────────────────


_CURSED_RAW = (
    "<think>\n"
    "THOUGHT: grep the repo for the version constant then patch it.\n"
    "ACTION: grep -rn 'VERSION' src/\n"
    "</think is missing so this is unclosed"
)

_CURSED_RAW_WITH_CODE = (
    "<think>\n"
    "THOUGHT: The fix is a one-line sed to bump the version.\n"
    "ACTION: sed -i 's/3.10.0/3.11.0/' pkg/version/version.go\n"
)


def test_heuristics_passes_cursed_think_output():
    # Unclosed <think> content fed directly as the answer should pass all checks
    # because the raw output contains code keywords and is non-empty/non-repetitive.
    responses = [
        _CURSED_RAW_WITH_CODE,
        "<think>\nTHOUGHT: read the file first\nACTION: cat pkg/version/version.go\n",
        "<think>\nTHOUGHT: verify the change\nACTION: grep VERSION pkg/version/version.go\n",
    ]
    out = _heuristics(responses, _req())
    assert all(v["passed"] for v in out), out


def test_heuristics_empty_vllm_still_fails():
    # A truly empty vLLM response (model produced nothing) must still be caught.
    out = _heuristics(["", "", ""], _req())
    assert not any(v["passed"] for v in out)
    assert all(v["reason"] == "empty response" for v in out)


def test_run_prompts_falls_back_to_raw_on_unclosed_think(monkeypatch):
    # When vLLM returns an unclosed <think> block, _strip_thinking returns "".
    # The `or raw` fallback must return the raw string so heuristics see real content.
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"text": _CURSED_RAW, "finish_reason": "length"}]}

    class _Client:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return _Response()

    monkeypatch.setattr("sanity_remote.worker.httpx.AsyncClient", _Client)
    engine = VllmEngine.__new__(VllmEngine)
    engine._s = type(
        "Settings",
        (),
        {
            "vllm_port": 1234,
            "gen_temperature": 0.7,
            "gen_top_p": 0.8,
            "gen_top_k": 20,
            "gen_min_p": 0.0,
            "gen_read_timeout_s": 300.0,
        },
    )()

    out = asyncio.run(engine._run_prompts("model-name", ["prompt"], 77))
    # Must return the raw string, not an empty string.
    assert out == [_CURSED_RAW]


def test_run_prompts_uses_raw_completions(monkeypatch):
    captured = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"text": "completion text", "finish_reason": "stop"}]}

    class _Client:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return _Response()

    monkeypatch.setattr("sanity_remote.worker.httpx.AsyncClient", _Client)
    engine = VllmEngine.__new__(VllmEngine)
    engine._s = type(
        "Settings",
        (),
        {
            "vllm_port": 1234,
            "gen_temperature": 0.7,
            "gen_top_p": 0.8,
            "gen_top_k": 20,
            "gen_min_p": 0.0,
            "gen_read_timeout_s": 300.0,
        },
    )()

    out = asyncio.run(engine._run_prompts("model-name", ["raw transcript"], 77))

    assert out == ["completion text"]
    assert captured["url"] == "http://localhost:1234/v1/completions"
    assert captured["json"] == {
        "model": "model-name",
        "prompt": "raw transcript",
        "max_tokens": 77,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "stop_token_ids": [248046],
    }


def test_run_prompts_timeout_has_retryable_fault_code(monkeypatch):
    class _Client:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            raise sanity_worker.httpx.ReadTimeout("too slow")

    monkeypatch.setattr("sanity_remote.worker.httpx.AsyncClient", _Client)
    engine = VllmEngine.__new__(VllmEngine)
    engine._s = type(
        "Settings",
        (),
        {
            "vllm_port": 1234,
            "gen_temperature": 0.7,
            "gen_top_p": 0.8,
            "gen_top_k": 20,
            "gen_min_p": 0.0,
            "gen_read_timeout_s": 300.0,
        },
    )()

    with pytest.raises(WorkerFault) as err:
        asyncio.run(engine._run_prompts("model-name", ["raw transcript"], 77))

    assert err.value.code == "generation_timeout"
    assert err.value.retryable is True
