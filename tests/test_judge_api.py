from __future__ import annotations

import asyncio
import json

import pytest

from albedo_config import JudgeSettings
from albedo_config.models import JUDGE_MODELS
from albedo_eval_service.judge_api import (
    JudgeSample,
    ObservationSimulationService,
    QuestionPrepStore,
    QuestionService,
    RepoContextClient,
    ScoreBatchRequest,
    SimulateObservationRequest,
    _collapse_looping,
    _evaluator_provider,
    _looping_output,
    _role_violation,
    _score_samples,
    _simulation_transcript,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient, JudgeRawResponse
from albedo_eval_service.shared.observation_format import (
    OPENHANDS,
    RETURNCODE,
    SWE_AGENT,
    empty_output,
    truncation_notice,
    valid_output,
)
from albedo_eval_service.simulator.prompt_simulator import (
    BASE_PROMPT,
    FORMAT_MINI_CODER,
    FORMAT_OPENHANDS,
    FORMAT_SWE_AGENT,
    simulation_system_prompt,
)

_RC_OBSERVATION = "<returncode>0</returncode>\n<output>\nok\n</output>"
_RC_PREFIX = [
    {"role": "user", "content": "task"},
    {"role": "assistant", "content": "```bash\nls\n```"},
    {"role": "user", "content": "<returncode>0</returncode>\n<output>\nfile.txt\n</output>"},
]


class FakeClient:
    def __init__(self, n_questions: int = 3):
        self.n_questions = n_questions

    async def complete(
        self,
        *,
        model,
        messages,
        temperature=None,
        max_tokens=None,
        provider=None,
        response_schema=None,
        accept=None,
        purpose="",
        eval_run_id="",
    ):
        questions = [{"text": f"q{i}?", "example_bad": "bad"} for i in range(self.n_questions)]
        return JudgeRawResponse(
            model=model, provider="fake", raw=json.dumps({"questions": questions})
        )

    async def score(
        self,
        *,
        model,
        messages,
        response_schema=None,
        schema_name="",
        max_tokens=None,
        provider=None,
        accept=None,
        purpose="",
    ):
        ids = response_schema["properties"]["answers"]["items"]["properties"]["id"]["enum"]
        content = messages[1]["content"]
        answer = 0 if "KING" in content and "CHAL" not in content else 1
        raw = json.dumps(
            {"answers": [{"id": qid, "answer": answer, "explanation": "e"} for qid in ids]}
        )
        return JudgeRawResponse(model=model, provider="fake", raw=raw)


def _reference_backed_service(settings, fake):
    from albedo_eval_service.judge_api import ReferenceTrajectoryService

    simulator = ObservationSimulationService(settings, fake)
    return QuestionService(settings, fake, ReferenceTrajectoryService(settings, fake, simulator))


_MESSAGES = [{"role": "user", "content": "fix the bug"}]


def test_evaluator_provider_is_always_fp8():
    settings = JudgeSettings(evaluator_providers="prov-a, prov-b")
    provider = _evaluator_provider(settings)
    assert provider == {
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
        "order": ["prov-a", "prov-b"],
    }
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


def test_simulation_system_prompt_carries_the_formats_block():
    openhands = simulation_system_prompt(OPENHANDS)
    mini = simulation_system_prompt(RETURNCODE)
    swe_agent = simulation_system_prompt(SWE_AGENT)

    assert BASE_PROMPT in openhands and FORMAT_OPENHANDS in openhands
    assert BASE_PROMPT in mini and FORMAT_MINI_CODER in mini
    assert BASE_PROMPT in swe_agent and FORMAT_SWE_AGENT in swe_agent
    assert "only the FIRST block is" in openhands
    assert "Respect pipe limits exactly" in openhands
    assert "Anchor on evidence" in openhands


def test_observation_simulation_primary_capped_then_fallback():
    class SimClient:
        def __init__(self, fail_first=False):
            self.calls = []
            self.fail_first = fail_first

        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            if self.fail_first and len(self.calls) == 1:
                return JudgeRawResponse(
                    model=kwargs["model"], provider="fake", raw="", error="blocked"
                )
            return JudgeRawResponse(model=kwargs["model"], provider="fake", raw=_RC_OBSERVATION)

    settings = JudgeSettings(
        evaluator_model="z-ai/glm-5.2",
        simulation_model="openai/gpt-5.6-luna",
        simulation_providers="openai",
        simulation_loop_reruns=0,
        simulation_max_tokens=123,
    )
    request = SimulateObservationRequest(
        eval_run_id="run",
        sample_id="sample",
        prompt="task",
        messages=_RC_PREFIX,
        assistant_output="```bash\npwd\n```",
    )

    client = SimClient()
    service = ObservationSimulationService(settings, client)
    observation = asyncio.run(service.simulate(request))
    assert observation == _RC_OBSERVATION
    (primary_call,) = client.calls
    assert primary_call["model"] == "openai/gpt-5.6-luna"
    assert primary_call["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert primary_call["parse_retries"] == 2
    assert primary_call["retry_count"] == 1
    assert primary_call["max_tokens"] == 123
    assert primary_call["accept"](_RC_OBSERVATION) is True
    assert primary_call["accept"]("Observation: ok") is False

    client = SimClient(fail_first=True)
    service = ObservationSimulationService(settings, client)
    observation = asyncio.run(service.simulate(request))
    assert observation == _RC_OBSERVATION
    assert [c["model"] for c in client.calls] == ["openai/gpt-5.6-luna", "z-ai/glm-5.2"]
    fallback_call = client.calls[1]
    assert fallback_call["provider"]["quantizations"] == ["fp8"]
    assert "parse_retries" not in fallback_call
    assert "retry_count" not in fallback_call


def test_observation_simulation_falls_back_on_invalid_format():
    class BadSimClient:
        async def complete(self, **kwargs):
            return JudgeRawResponse(model=kwargs["model"], provider="fake", raw="Observation: nope")

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

    openhands = asyncio.run(run("open-swe-traces/x:0:0"))
    mini = asyncio.run(run("mini-coder/x:0:0"))

    assert openhands == (
        "\n[The command completed with exit code 0.]\n[Command finished with exit code 0]"
    )
    assert mini == "<returncode>0</returncode>\n<output>\n</output>"
    for fmt in (RETURNCODE, SWE_AGENT, OPENHANDS):
        assert valid_output(empty_output(fmt), fmt)


def test_scoring_scores_both_sides_independently():
    settings = JudgeSettings(num_questions=3, sota_trajectory_turns=1)
    fake = FakeClient(n_questions=8)
    store = QuestionPrepStore(settings, _reference_backed_service(settings, fake))
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
                messages=_MESSAGES,
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
    assert len(record["judge_results"]) == 2 * len(JUDGE_MODELS)
    assert {r["side"] for r in record["judge_results"]} == {"previous_king", "challenger"}


def test_call_retries_until_accept_passes():
    settings = JudgeSettings(openrouter_api_key="x", parse_retries=3)
    client = JudgeLLMClient(settings)
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
    client = JudgeLLMClient(settings)
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
    assert result.raw == "bad"
    assert calls["n"] == 3


class OneJudgeBrokenClient:
    def __init__(self, n_questions=3):
        self.n_questions = n_questions

    async def complete(
        self,
        *,
        model,
        messages,
        temperature=None,
        max_tokens=None,
        provider=None,
        response_schema=None,
        accept=None,
        purpose="",
        eval_run_id="",
    ):
        qs = [{"text": f"q{i}", "example_bad": "b"} for i in range(self.n_questions)]
        return JudgeRawResponse(model=model, provider="fake", raw=json.dumps({"questions": qs}))

    async def score(
        self,
        *,
        model,
        messages,
        response_schema=None,
        schema_name="",
        max_tokens=None,
        provider=None,
        accept=None,
        purpose="",
    ):
        ids = response_schema["properties"]["answers"]["items"]["properties"]["id"]["enum"]
        if model == JUDGE_MODELS[0]:
            raw = "garbage, not json"
        else:
            raw = json.dumps({"answers": [{"id": i, "answer": 1, "explanation": "e"} for i in ids]})
        return JudgeRawResponse(model=model, provider="fake", raw=raw)


def test_sample_unscored_if_a_judge_never_parses():
    settings = JudgeSettings(num_questions=3, sota_trajectory_turns=1)
    fake = OneJudgeBrokenClient(n_questions=8)
    store = QuestionPrepStore(settings, _reference_backed_service(settings, fake))
    request = ScoreBatchRequest(
        eval_run_id="r",
        batch_id="b",
        total_sample_count=1,
        judge_models=list(JUDGE_MODELS[:3]),
        samples=[
            JudgeSample(
                sample_id="s1",
                prompt="task",
                previous_king_output="KING",
                challenger_output="CHAL",
                messages=_MESSAGES,
            )
        ],
    )
    records = asyncio.run(
        _score_samples(client=fake, request=request, settings=settings, prep_store=store)
    )
    assert records[0]["scored"] is False


def test_truncated_side_scores_zero_without_calling_the_judge():
    class RecordingClient(FakeClient):
        def __init__(self):
            super().__init__(n_questions=8)
            self.judged = []

        async def score(self, **kwargs):
            content = kwargs["messages"][1]["content"]
            if "KING" in content or "CHAL" in content:
                self.judged.append(content)
            return await super().score(**kwargs)

    settings = JudgeSettings(num_questions=3, sota_trajectory_turns=1)
    fake = RecordingClient()
    store = QuestionPrepStore(settings, _reference_backed_service(settings, fake))
    request = ScoreBatchRequest(
        eval_run_id="r",
        batch_id="b",
        total_sample_count=1,
        judge_models=list(JUDGE_MODELS[:1]),
        samples=[
            JudgeSample(
                sample_id="s1",
                prompt="task",
                previous_king_output="KING",
                challenger_output=f"CANDIDATE OUTPUT 1:\n{truncation_notice(16384)}",
                messages=_MESSAGES,
            )
        ],
    )
    record = asyncio.run(
        _score_samples(client=fake, request=request, settings=settings, prep_store=store)
    )[0]

    assert record["scored"] is True
    assert record["challenger_score"] == 0.0
    assert record["king_score"] is not None

    challenger_results = [r for r in record["judge_results"] if r["side"] == "challenger"]
    assert challenger_results
    assert all(r["corrupted"] for r in challenger_results)
    assert all(r["parse_ok"] for r in challenger_results)
    assert all(r["yes_rate"] == 0.0 for r in challenger_results)

    assert len(fake.judged) == 1
    assert all("KING" in judged for judged in fake.judged)


def test_scoring_regenerates_questions_when_async_prep_failed():
    class PrepFailsOnceClient(FakeClient):
        def __init__(self):
            super().__init__(n_questions=8)
            self.complete_calls = 0

        async def complete(self, **kwargs):
            self.complete_calls += 1
            if self.complete_calls == 1:
                raise RuntimeError("prep broke")
            return await super().complete(**kwargs)

    settings = JudgeSettings(num_questions=3, sota_trajectory_turns=1)
    fake = PrepFailsOnceClient()
    store = QuestionPrepStore(settings, _reference_backed_service(settings, fake))

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
                            messages=_MESSAGES,
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
                    messages=_MESSAGES,
                )
            ],
        )
        return await _score_samples(
            client=fake, request=request, settings=settings, prep_store=store
        )

    records = asyncio.run(run())

    assert records[0]["scored"] is True
    assert fake.complete_calls == 4


class _AnchorFakeClient:
    def __init__(self, n_questions: int = 30, fail_reference: bool = False):
        self.n_questions = n_questions
        self.fail_reference = fail_reference
        self.saw_reference_prompt = False

    async def complete(
        self,
        *,
        model,
        messages,
        temperature=None,
        max_tokens=None,
        provider=None,
        response_schema=None,
        accept=None,
        purpose="",
        parse_retries=None,
        retry_count=None,
        eval_run_id="",
    ):
        if response_schema is None:
            if self.fail_reference:
                return JudgeRawResponse(model=model, provider="fake", raw="", error="boom")
            if messages[0]["role"] == "system" and "ENVIRONMENT" in messages[0]["content"]:
                return JudgeRawResponse(model=model, provider="fake", raw="Observation: ok")
            return JudgeRawResponse(
                model=model,
                provider="fake",
                raw="THOUGHT: fix lib/x.py\n\n```bash\nsed -i 's/a/b/' lib/x.py\n```",
            )
        if "REFERENCE TRAJECTORY" in messages[1]["content"]:
            self.saw_reference_prompt = True
        questions = [
            {"text": f"q{i} gate{i}?", "example_bad": "bad"} for i in range(self.n_questions)
        ]
        questions.append(
            {
                "text": "Does it avoid re-running the grep the reference already ran?",
                "example_bad": "bad",
            }
        )
        return JudgeRawResponse(
            model=model, provider="fake", raw=json.dumps({"questions": questions})
        )


def _anchor_service(fake):
    from albedo_eval_service.judge_api import ReferenceTrajectoryService

    settings = JudgeSettings(openrouter_api_key="k", num_questions=50)
    simulator = ObservationSimulationService(settings, fake)
    return QuestionService(settings, fake, ReferenceTrajectoryService(settings, fake, simulator))


def test_prepare_anchors_on_reference_and_filters_leaks():
    from albedo_eval_service.judge_api import QuestionPrepSample

    fake = _AnchorFakeClient()
    service = _anchor_service(fake)
    sample = QuestionPrepSample(
        sample_id="swe-zero/data/train-0.parquet:1:1",
        prompt="TASK",
        messages=[{"role": "user", "content": "fix the bug"}],
        assistant_turns=2,
    )
    result = asyncio.run(service.prepare(sample, eval_run_id="run-1"))
    assert fake.saw_reference_prompt
    assert result.source["question_mode"] == "sota_anchored"
    assert result.source["reference_model"] == "z-ai/glm-5.2"
    assert "REFERENCE STEP" in result.source["reference_trajectory"]
    assert all("the reference" not in q["text"].casefold() for q in result.questions)
    behavior_tags = {
        q["tag"]
        for q in result.questions
        if q["requires"] == "action" and q["tag"].startswith("behavior:")
    }
    assert behavior_tags == {
        "behavior:precision_reads",
        "behavior:issue_anchored_narrowing",
        "behavior:convergence_and_orientation",
    }

    behavior_tags = {
        q["tag"]
        for q in result.questions
        if q["requires"] == "action" and q["tag"].startswith("behavior:")
    }
    assert behavior_tags == {
        "behavior:precision_reads",
        "behavior:issue_anchored_narrowing",
        "behavior:convergence_and_orientation",
    }


def test_prepare_raises_when_reference_generation_and_reroll_both_fail():
    from albedo_eval_service.judge_api import QuestionPrepSample, QuestionScoringUnavailable

    fake = _AnchorFakeClient(fail_reference=True)
    service = _anchor_service(fake)
    sample = QuestionPrepSample(
        sample_id="s:1:1",
        prompt="TASK",
        messages=[{"role": "user", "content": "fix"}],
        assistant_turns=2,
    )
    with pytest.raises(QuestionScoringUnavailable):
        asyncio.run(service.prepare(sample, eval_run_id="run-1"))
    assert not fake.saw_reference_prompt


def test_prepare_raises_when_sample_has_no_messages():
    from albedo_eval_service.judge_api import QuestionScoringUnavailable

    fake = _AnchorFakeClient()
    service = _anchor_service(fake)
    with pytest.raises(QuestionScoringUnavailable):
        asyncio.run(
            service.prepare(
                JudgeSample(
                    sample_id="s:1:1",
                    prompt="TASK",
                    previous_king_output="k",
                    challenger_output="c",
                )
            )
        )
    assert not fake.saw_reference_prompt


def test_simulation_transcript_strips_thought_from_assistant_turns():
    from albedo_eval_service.judge_api import _simulation_transcript

    transcript = _simulation_transcript(
        messages=[
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "THOUGHT: files X and Y were already shown\n\n```bash\ncat a.py\n```",
            },
            {"role": "user", "content": "Observation: ..."},
        ],
        prompt="fix the bug",
        assistant_output="THOUGHT: the fix is verified and tests pass\n\n```bash\nsed -n '1,5p' a.py\n```",  # noqa: E501
    )
    assert "already shown" not in transcript
    assert "tests pass" not in transcript
    assert "cat a.py" in transcript
    assert "sed -n '1,5p' a.py" in transcript


def test_simulation_transcript_keeps_text_without_command_block():
    from albedo_eval_service.judge_api import _simulation_transcript

    transcript = _simulation_transcript(
        messages=None,
        prompt="task",
        assistant_output="no fenced block here",
    )
    assert "no fenced block here" in transcript


def test_simulation_system_prompt_grounded_and_fallback():
    assert simulation_system_prompt(OPENHANDS) == f"{BASE_PROMPT}\n{FORMAT_OPENHANDS}"
    assert simulation_system_prompt(OPENHANDS, None) == f"{BASE_PROMPT}\n{FORMAT_OPENHANDS}"
    assert simulation_system_prompt(OPENHANDS, "") == f"{BASE_PROMPT}\n{FORMAT_OPENHANDS}"

    grounded = simulation_system_prompt(OPENHANDS, "GROUNDING BLOCK")
    assert grounded == f"{BASE_PROMPT}\nGROUNDING BLOCK\n{FORMAT_OPENHANDS}"
    assert FORMAT_MINI_CODER in simulation_system_prompt(RETURNCODE, "GROUNDING BLOCK")


def test_observation_simulation_uses_repo_context_when_available():
    class SimClient:
        async def complete(self, **kwargs):
            self.kwargs = kwargs
            return JudgeRawResponse(model=kwargs["model"], provider="fake", raw="ok")

    class FakeRepoContext:
        def __init__(self, block):
            self.block = block
            self.calls = []

        async def context_for(self, sample_id, assistant_output):
            self.calls.append((sample_id, assistant_output))
            return self.block

    async def run(repo_context):
        client = SimClient()
        service = ObservationSimulationService(
            JudgeSettings(simulation_model=""), client, repo_context
        )
        await service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="swe-zero/x:0:0",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output="```bash\nls\n```",
            )
        )
        return client.kwargs["messages"][0]["content"]

    ctx = FakeRepoContext("REAL LISTING")
    system_prompt = asyncio.run(run(ctx))
    assert "REAL LISTING" in system_prompt
    assert BASE_PROMPT in system_prompt
    assert ctx.calls == [("swe-zero/x:0:0", "```bash\nls\n```")]

    assert asyncio.run(run(FakeRepoContext(None))) == f"{BASE_PROMPT}\n{FORMAT_OPENHANDS}"
    assert asyncio.run(run(None)) == f"{BASE_PROMPT}\n{FORMAT_OPENHANDS}"


def test_looping_output_detection():
    assert _looping_output("Observation:\n" + "same line\n" * 100) is True
    assert _looping_output("Observation:\n" + "alpha beta\ngamma delta\n" * 60) is True
    listing = "\n".join(f"./pkg/module_{i}.py" for i in range(300))
    assert _looping_output(f"Observation:\n{listing}") is False
    assert _looping_output("Observation:\n" + "PASSED\n" * 10 + "done\n") is False
    assert _looping_output("Observation:") is False


def test_collapse_looping_keeps_prefix_and_marks_repetition():
    text = "Observation:\nreal output line\n" + "same line\n" * 100
    collapsed = _collapse_looping(text)
    assert collapsed.count("same line") < 30
    assert "real output line" in collapsed
    assert "... (output repeats)" in collapsed
    assert len(collapsed) < len(text) / 2

    cycle = "Observation:\nhead\n" + "alpha beta\ngamma delta\n" * 60
    collapsed_cycle = _collapse_looping(cycle)
    assert "head" in collapsed_cycle
    assert collapsed_cycle.count("alpha beta") <= 3
    assert "... (output repeats)" in collapsed_cycle


def test_observation_simulation_reruns_looping_then_collapses():
    looping_raw = "ok start\n" + "loop line\n" * 100

    class ScriptedClient:
        def __init__(self, raws):
            self.raws = list(raws)
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            raw = self.raws.pop(0) if len(self.raws) > 1 else self.raws[0]
            return JudgeRawResponse(model=kwargs["model"], provider="fake", raw=raw)

    settings = JudgeSettings(simulation_model="", simulation_loop_reruns=2)

    def run(client):
        service = ObservationSimulationService(settings, client)
        return asyncio.run(
            service.simulate(
                SimulateObservationRequest(
                    eval_run_id="run",
                    sample_id="swe-zero/x:0:0",
                    prompt="task",
                    messages=[{"role": "user", "content": "task"}],
                    assistant_output="```bash\nls\n```",
                )
            )
        )

    recovering = ScriptedClient([looping_raw, "clean"])
    assert run(recovering) == "clean"
    assert recovering.calls == 2

    stuck = ScriptedClient([looping_raw])
    observation = run(stuck)
    assert stuck.calls == 1 + settings.simulation_loop_reruns
    assert observation.startswith("ok start")
    assert observation.count("loop line") < 30
    assert "... (output repeats)" in observation


def test_simulation_primary_model_falls_back_to_evaluator():
    ROLE_LEAK = "ls output\n### assistant\n```bash\nfind .\n```"

    class Scripted:
        def __init__(self, by_model):
            self.by_model = by_model
            self.calls = []

        async def complete(self, **kw):
            model = kw["model"]
            self.calls.append(model)
            queue = self.by_model[model]
            raw = queue.pop(0) if len(queue) > 1 else queue[0]
            return JudgeRawResponse(model=model, provider="fake", raw=raw)

    settings = JudgeSettings(
        evaluator_model="z-ai/glm-5.2",
        simulation_model="xiaomi/mimo-v2.5-pro",
        simulation_loop_reruns=1,
    )

    def run(client):
        service = ObservationSimulationService(settings, client)
        return asyncio.run(
            service.simulate(
                SimulateObservationRequest(
                    eval_run_id="run",
                    sample_id="swe-zero/x:0:0",
                    prompt="task",
                    messages=[{"role": "user", "content": "task"}],
                    assistant_output="```bash\nls\n```",
                )
            )
        )

    good = Scripted({"xiaomi/mimo-v2.5-pro": ["ok"], "z-ai/glm-5.2": ["unused"]})
    assert run(good) == "ok"
    assert good.calls == ["xiaomi/mimo-v2.5-pro"]

    recovers = Scripted(
        {
            "xiaomi/mimo-v2.5-pro": [ROLE_LEAK, "recovered"],
            "z-ai/glm-5.2": ["unused"],
        }
    )
    assert run(recovers) == "recovered"
    assert recovers.calls == ["xiaomi/mimo-v2.5-pro"] * 2

    stuck = Scripted(
        {
            "xiaomi/mimo-v2.5-pro": [ROLE_LEAK],
            "z-ai/glm-5.2": ["from the fallback"],
        }
    )
    assert run(stuck) == "from the fallback"
    assert stuck.calls == ["xiaomi/mimo-v2.5-pro"] * 4 + ["z-ai/glm-5.2"]

    bad_format = Scripted(
        {
            "xiaomi/mimo-v2.5-pro": ["Observation: wrong dialect"],
            "z-ai/glm-5.2": ["from the fallback"],
        }
    )
    assert run(bad_format) == "from the fallback"
    assert bad_format.calls[-1] == "z-ai/glm-5.2"

    solo = Scripted({"z-ai/glm-5.2": ["ok"]})
    service = ObservationSimulationService(
        JudgeSettings(evaluator_model="z-ai/glm-5.2", simulation_model=""), solo
    )
    asyncio.run(
        service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="swe-zero/x:0:0",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output="```bash\nls\n```",
            )
        )
    )
    assert solo.calls == ["z-ai/glm-5.2"]


def test_role_violation_detection():
    assert _role_violation("Observation: ok") is False
    assert _role_violation("Observation: ls\n### assistant\n```bash\nls\n```") is True
    assert _role_violation("THOUGHT: let me check the files") is True
    assert _role_violation("Observation:\nTHOUGHT: next I will") is True
    assert _role_violation("Observation: docs/thought.md: THOUGHTS are cheap") is False


def test_same_prompt_used_for_primary_and_fallback():
    class Recorder:
        def __init__(self):
            self.systems = []

        async def complete(self, **kw):
            self.systems.append((kw["model"], kw["messages"][0]["content"]))
            raw = "ok" if kw["model"] == "z-ai/glm-5.2" else "x\n### assistant\nbad"
            return JudgeRawResponse(model=kw["model"], provider="fake", raw=raw)

    class Ctx:
        async def context_for(self, sample_id, assistant_output):
            return "GROUNDING BLOCK"

    client = Recorder()
    settings = JudgeSettings(
        evaluator_model="z-ai/glm-5.2",
        simulation_model="xiaomi/mimo-v2.5-pro",
        simulation_loop_reruns=0,
    )
    service = ObservationSimulationService(settings, client, Ctx())
    observation = asyncio.run(
        service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="swe-zero/x:0:0",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output="```bash\nls\n```",
            )
        )
    )
    assert observation == "ok"
    primary_system = [s for m, s in client.systems if m == "xiaomi/mimo-v2.5-pro"][0]
    fallback_system = [s for m, s in client.systems if m == "z-ai/glm-5.2"][0]
    assert primary_system == fallback_system
    assert BASE_PROMPT in fallback_system
    assert "GROUNDING BLOCK" in fallback_system


def test_repo_context_client_degrades_to_none_on_error():
    settings = JudgeSettings(
        repo_context_url="http://127.0.0.1:9", repo_context_timeout_seconds=0.5
    )
    client = RepoContextClient(settings)
    try:
        assert asyncio.run(client.context_for("swe-zero/x:0:0", "```bash\nls\n```")) is None
    finally:
        asyncio.run(client.aclose())
