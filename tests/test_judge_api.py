from __future__ import annotations

import asyncio
import json

from albedo_eval_service.judge_api import (
    JudgeSample,
    QuestionPrepStore,
    QuestionService,
    ScoreBatchRequest,
    _evaluator_provider,
    _score_samples,
)
from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_core import JUDGE_MODELS
from albedo_eval_service.judge_openrouter import JudgeRawResponse, OpenRouterJudgeClient


class FakeClient:
    """Evaluator returns N questions; judges answer all-1 for the challenger, all-0 for the king."""

    def __init__(self, n_questions: int = 3):
        self.n_questions = n_questions

    async def complete(self, *, model, messages, temperature=None, max_tokens=None, provider=None, response_schema=None, accept=None):
        questions = [{"text": f"q{i}?", "example_bad": "bad"} for i in range(self.n_questions)]
        return JudgeRawResponse(model=model, provider="fake", raw=json.dumps({"questions": questions}))

    async def score(self, *, model, messages, response_schema=None, schema_name="", max_tokens=None, provider=None, accept=None):
        ids = response_schema["properties"]["answers"]["items"]["properties"]["id"]["enum"]
        answer = 1 if "CHAL" in messages[1]["content"] else 0
        raw = json.dumps({"answers": [{"id": qid, "answer": answer, "explanation": "e"} for qid in ids]})
        return JudgeRawResponse(model=model, provider="fake", raw=raw)


def test_evaluator_provider_is_always_fp8():
    # order list -> fallbacks off; failover is the retry-rotation over the list.
    settings = JudgeSettings(evaluator_providers="prov-a, prov-b")
    provider = _evaluator_provider(settings)
    assert provider == {"allow_fallbacks": False, "quantizations": ["fp8"], "order": ["prov-a", "prov-b"]}
    # no providers listed -> still fp8 + allow_fallbacks (OpenRouter fails over across fp8 providers)
    bare = _evaluator_provider(JudgeSettings(evaluator_providers=""))
    assert bare == {"allow_fallbacks": True, "quantizations": ["fp8"]}


def test_scoring_scores_both_sides_independently():
    settings = JudgeSettings(num_questions=3)
    fake = FakeClient(n_questions=3)
    store = QuestionPrepStore(settings, QuestionService(settings, fake))
    request = ScoreBatchRequest(
        eval_run_id="run-1",
        batch_id="score-0001",
        total_sample_count=1,
        judge_models=list(JUDGE_MODELS[:3]),
        samples=[
            JudgeSample(
                sample_id="s1",
                prompt="task",
                previous_king_output="KING answer",
                challenger_output="CHAL answer",
            )
        ],
    )
    records = asyncio.run(
        _score_samples(client=fake, request=request, settings=settings, prep_store=store)
    )
    record = records[0]
    assert record["scoring_mode"] == "binary"
    assert record["scored"] is True
    assert record["challenger_score"] == 1.0
    assert record["king_score"] == 0.0
    # 2 sides x 3 judges
    assert len(record["judge_results"]) == 6
    assert {r["side"] for r in record["judge_results"]} == {"previous_king", "challenger"}


def test_call_retries_until_accept_passes():
    # _score_with_retries returns bad, bad, good; accept passes only on "good" -> 3rd call wins.
    settings = JudgeSettings(openrouter_api_key="x", parse_retries=3)
    client = OpenRouterJudgeClient(settings)
    calls = {"n": 0}

    async def fake_swr(**kwargs):
        calls["n"] += 1
        return JudgeRawResponse(model="m", provider="p", raw="good" if calls["n"] == 3 else "bad")

    client._score_with_retries = fake_swr

    async def run():
        r = await client._call(model="m", messages=[], accept=lambda raw: raw == "good")
        await client.aclose()
        return r

    result = asyncio.run(run())
    assert result.raw == "good"
    assert calls["n"] == 3


def test_call_gives_up_after_parse_retries():
    settings = JudgeSettings(openrouter_api_key="x", parse_retries=3)
    client = OpenRouterJudgeClient(settings)
    calls = {"n": 0}

    async def fake_swr(**kwargs):
        calls["n"] += 1
        return JudgeRawResponse(model="m", provider="p", raw="bad")

    client._score_with_retries = fake_swr

    async def run():
        r = await client._call(model="m", messages=[], accept=lambda raw: raw == "good")
        await client.aclose()
        return r

    result = asyncio.run(run())
    assert result.raw == "bad"        # returns last attempt
    assert calls["n"] == 3            # bounded at parse_retries


class OneJudgeBrokenClient:
    """Evaluator ok; JUDGE_MODELS[0] always returns unparseable output; the other judges are fine."""

    def __init__(self, n_questions=3):
        self.n_questions = n_questions

    async def complete(self, *, model, messages, temperature=None, max_tokens=None, provider=None, response_schema=None, accept=None):
        qs = [{"text": f"q{i}", "example_bad": "b"} for i in range(self.n_questions)]
        return JudgeRawResponse(model=model, provider="fake", raw=json.dumps({"questions": qs}))

    async def score(self, *, model, messages, response_schema=None, schema_name="", max_tokens=None, provider=None, accept=None):
        ids = response_schema["properties"]["answers"]["items"]["properties"]["id"]["enum"]
        if model == JUDGE_MODELS[0]:
            raw = "garbage, not json"
        else:
            raw = json.dumps({"answers": [{"id": i, "answer": 1, "explanation": "e"} for i in ids]})
        return JudgeRawResponse(model=model, provider="fake", raw=raw)


def test_sample_unscored_if_a_judge_never_parses():
    settings = JudgeSettings(num_questions=3)
    fake = OneJudgeBrokenClient(n_questions=3)
    store = QuestionPrepStore(settings, QuestionService(settings, fake))
    request = ScoreBatchRequest(
        eval_run_id="r", batch_id="b", total_sample_count=1, judge_models=list(JUDGE_MODELS[:3]),
        samples=[JudgeSample(sample_id="s1", prompt="task", previous_king_output="KING", challenger_output="CHAL")],
    )
    records = asyncio.run(_score_samples(client=fake, request=request, settings=settings, prep_store=store))
    # one judge never parsed -> sample invalid (all 3 judges required per side)
    assert records[0]["scored"] is False
