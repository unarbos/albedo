from __future__ import annotations

import asyncio
import json

from albedo_eval_service.judge_api import (
    BASE_PROMPT,
    FORMAT_MINI_CODER,
    FORMAT_SWE_ZERO,
    JudgeSample,
    ObservationSimulationService,
    QuestionPrepStore,
    QuestionService,
    ScoreBatchRequest,
    SimulateObservationRequest,
    _empty_simulation_output,
    _evaluator_provider,
    _score_samples,
    _simulation_system_prompt,
    _simulation_transcript,
    _valid_simulation_output,
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


def test_simulation_transcript_uses_section_markers():
    transcript = _simulation_transcript(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ],
        prompt="unused",
        assistant_output="```bash\nls\n```",
    )

    assert transcript == "### system\nsys\n\n### user\ntask\n\n### assistant\n```bash\nls\n```"


def test_simulation_system_prompt_selects_dataset_format():
    swe = _simulation_system_prompt("swe-zero/data/train-00000.parquet:0:0")
    mini = _simulation_system_prompt("mini-coder/data/train-00000.parquet:0:0")

    assert BASE_PROMPT in swe and FORMAT_SWE_ZERO in swe
    assert BASE_PROMPT in mini and FORMAT_MINI_CODER in mini
    assert "only the FIRST block is" in swe
    assert "Respect pipe limits exactly" in swe
    assert "Anchor on evidence" in swe
    assert _valid_simulation_output("Observation:", "swe-zero/x:0:0") is True
    assert _valid_simulation_output("No observation", "swe-zero/x:0:0") is False
    assert _valid_simulation_output(
        "<returncode>0</returncode>\n<output>\nok\n</output>", "mini-coder/x:0:0"
    ) is True
    assert _valid_simulation_output("Observation: ok", "mini-coder/x:0:0") is False


def test_observation_simulation_uses_glm_and_retries_without_observation_prefix():
    class SimClient:
        async def complete(self, **kwargs):
            self.kwargs = kwargs
            return JudgeRawResponse(
                model=kwargs["model"], provider="fake", raw="Observation: ok"
            )

    settings = JudgeSettings(evaluator_model="z-ai/glm-5.2", simulation_max_tokens=123)
    client = SimClient()
    service = ObservationSimulationService(settings, client)
    observation = asyncio.run(
        service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="sample",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output="```bash\npwd\n```",
            )
        )
    )

    assert observation == "Observation: ok"
    assert client.kwargs["model"] == "z-ai/glm-5.2"
    assert client.kwargs["max_tokens"] == 123
    assert client.kwargs["provider"]["quantizations"] == ["fp8"]
    assert client.kwargs["accept"]("Observation: ok") is True
    assert client.kwargs["accept"]("not an observation") is False


def test_observation_simulation_falls_back_on_invalid_format():
    class BadSimClient:
        async def complete(self, **kwargs):
            return JudgeRawResponse(model=kwargs["model"], provider="fake", raw="not an observation")

    async def run(sample_id):
        service = ObservationSimulationService(
            JudgeSettings(evaluator_model="z-ai/glm-5.2"),
            BadSimClient(),
        )
        return await service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id=sample_id,
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output="```bash\ntrue\n```",
            )
        )

    swe = asyncio.run(run("swe-zero/x:0:0"))
    mini = asyncio.run(run("mini-coder/x:0:0"))

    assert swe == "Observation:"
    assert mini == "<returncode>0</returncode>\n<output>\n</output>"
    assert _valid_simulation_output(_empty_simulation_output("swe-zero/x:0:0"), "swe-zero/x:0:0")
    assert _valid_simulation_output(_empty_simulation_output("mini-coder/x:0:0"), "mini-coder/x:0:0")


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


def test_scoring_regenerates_questions_when_async_prep_failed():
    class PrepFailsOnceClient(FakeClient):
        def __init__(self):
            super().__init__(n_questions=3)
            self.complete_calls = 0

        async def complete(self, **kwargs):
            self.complete_calls += 1
            if self.complete_calls == 1:
                raise RuntimeError("prep broke")
            return await super().complete(**kwargs)

    settings = JudgeSettings(num_questions=3)
    fake = PrepFailsOnceClient()
    store = QuestionPrepStore(settings, QuestionService(settings, fake))

    async def run():
        prep_id = store.start(
            type(
                "Req",
                (),
                {
                    "eval_run_id": "run",
                    "samples": [
                        JudgeSample(
                            sample_id="s1",
                            prompt="task",
                            previous_king_output="",
                            challenger_output="",
                        )
                    ],
                },
            )()
        )
        request = ScoreBatchRequest(
            eval_run_id="run",
            batch_id="score-0001",
            total_sample_count=1,
            category_prep_id=prep_id,
            judge_models=list(JUDGE_MODELS[:1]),
            samples=[
                JudgeSample(
                    sample_id="s1",
                    prompt="task",
                    previous_king_output="KING",
                    challenger_output="CHAL",
                )
            ],
        )
        return await _score_samples(
            client=fake, request=request, settings=settings, prep_store=store
        )

    records = asyncio.run(run())

    assert records[0]["scored"] is True
    assert fake.complete_calls == 2
