from __future__ import annotations

import asyncio
import json

import httpx

from albedo_eval_service.chutes_glm import GLMProviderClient
from albedo_eval_service.judge_api import JudgeSample, ScoreBatchRequest, _score_samples_with_categories
from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_core import (
    build_category_pairwise_messages,
    parse_category_verdict,
    validate_category_payload,
)
from albedo_eval_service.judge_openrouter import JudgeRawResponse


def test_validate_category_payload_requires_exact_shape():
    raw = json.dumps(
        {
            "categories": [
                {
                    "id": f"cat_{idx:02d}",
                    "name": f"Category {idx}",
                    "description": "desc",
                    "scoring_guidance": "guidance",
                }
                for idx in range(1, 6)
            ]
        }
    )
    categories, digest = validate_category_payload(raw)
    assert len(categories) == 5
    assert categories[0]["id"] == "cat_01"
    assert digest.startswith("sha256:")




def _category_payload_list():
    return [
        {
            "id": f"cat_{idx:02d}",
            "name": f"Category {idx}",
            "description": "desc",
            "scoring_guidance": "guidance",
        }
        for idx in range(1, 6)
    ]


def test_validate_category_payload_accepts_common_wrappers():
    direct, _ = validate_category_payload(json.dumps(_category_payload_list()))
    assert direct[0]["id"] == "cat_01"

    fenced, _ = validate_category_payload("```json\n" + json.dumps(_category_payload_list()) + "\n```")
    assert len(fenced) == 5

    nested, _ = validate_category_payload(json.dumps({"result": {"categories": _category_payload_list()}}))
    assert nested[-1]["id"] == "cat_05"

def test_parse_category_verdict_maps_dynamic_ids_for_challenger_order():
    categories = [
        {"id": f"cat_{idx:02d}", "name": f"c{idx}", "description": "d", "scoring_guidance": "g"}
        for idx in range(1, 6)
    ]
    verdict = parse_category_verdict(
        json.dumps({category["id"]: 1 for category in categories}),
        categories=categories,
        challenger_position=1,
    )
    assert verdict.parse_ok is True
    assert verdict.judge_mean == 1.0


def test_category_pairwise_prompt_uses_categories_not_glm_response():
    categories = [
        {"id": f"cat_{idx:02d}", "name": f"c{idx}", "description": "d", "scoring_guidance": "g"}
        for idx in range(1, 6)
    ]
    messages = build_category_pairwise_messages(
        context_prompt="task",
        previous_king_output="king",
        challenger_output="challenger",
        challenger_first=False,
        categories=categories,
    )
    content = messages[1]["content"]
    assert "cat_01" in content
    assert "GLM RESPONSE" not in content


def test_glm_openrouter_fallback_uses_fp8_provider_filter():
    payloads = []

    def chutes_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    def openrouter_handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings = JudgeSettings(
        chutes_api_key="ck",
        openrouter_api_key="ok",
        glm_retry_count=0,
        chutes_utilization_check_enabled=False,
    )
    client = GLMProviderClient(settings)

    async def run():
        assert client._chutes_client is not None
        assert client._openrouter_client is not None
        await client._chutes_client.aclose()
        await client._openrouter_client.aclose()
        client._chutes_client = httpx.AsyncClient(
            base_url=settings.chutes_base_url.rstrip("/"), transport=httpx.MockTransport(chutes_handler)
        )
        client._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(openrouter_handler)
        )
        try:
            result = await client.complete(messages=[{"role": "user", "content": "x"}])
        finally:
            await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.provider == "openrouter-fp8"
    assert payloads[0]["model"] == "z-ai/glm-5.2"
    assert payloads[0]["provider"]["quantizations"] == ["fp8"]
    assert payloads[0]["provider"]["allow_fallbacks"] is True
    assert payloads[0]["provider"]["require_parameters"] is True




def test_glm_fallback_disables_chutes_after_three_errors():
    calls = {"chutes": 0, "openrouter": 0}

    def chutes_handler(request: httpx.Request) -> httpx.Response:
        calls["chutes"] += 1
        return httpx.Response(503, json={"error": "down"})

    def openrouter_handler(request: httpx.Request) -> httpx.Response:
        calls["openrouter"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings = JudgeSettings(
        chutes_api_key="ck",
        openrouter_api_key="ok",
        glm_retry_count=0,
        chutes_disable_after_errors=3,
        chutes_utilization_check_enabled=False,
    )
    client = GLMProviderClient(settings)

    async def run():
        assert client._chutes_client is not None
        assert client._openrouter_client is not None
        await client._chutes_client.aclose()
        await client._openrouter_client.aclose()
        client._chutes_client = httpx.AsyncClient(
            base_url=settings.chutes_base_url.rstrip("/"), transport=httpx.MockTransport(chutes_handler)
        )
        client._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(openrouter_handler)
        )
        try:
            results = [
                await client.complete(messages=[{"role": "user", "content": f"x-{idx}"}])
                for idx in range(4)
            ]
        finally:
            await client.aclose()
        return results

    results = asyncio.run(run())
    assert [result.provider for result in results] == ["openrouter-fp8"] * 4
    assert calls["chutes"] == 3
    assert calls["openrouter"] == 4



def test_glm_skips_chutes_when_15m_utilization_is_high():
    calls = {"utilization": 0, "chutes": 0, "openrouter": 0}

    def utilization_handler(request: httpx.Request) -> httpx.Response:
        calls["utilization"] += 1
        return httpx.Response(
            200,
            json=[
                {
                    "chute_id": "08901219-159f-55a7-87cf-9d0d02744668",
                    "utilization_15m": 0.76,
                    "instance_count": 4,
                }
            ],
        )

    def chutes_handler(request: httpx.Request) -> httpx.Response:
        calls["chutes"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "chutes"}}]})

    def openrouter_handler(request: httpx.Request) -> httpx.Response:
        calls["openrouter"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "openrouter"}}]})

    settings = JudgeSettings(
        chutes_api_key="ck",
        openrouter_api_key="ok",
        glm_retry_count=0,
        chutes_utilization_15m_threshold=0.75,
        chutes_utilization_cache_seconds=60,
    )
    client = GLMProviderClient(settings)

    async def run():
        assert client._chutes_client is not None
        assert client._openrouter_client is not None
        assert client._utilization_client is not None
        await client._chutes_client.aclose()
        await client._openrouter_client.aclose()
        await client._utilization_client.aclose()
        client._chutes_client = httpx.AsyncClient(
            base_url=settings.chutes_base_url.rstrip("/"), transport=httpx.MockTransport(chutes_handler)
        )
        client._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(openrouter_handler)
        )
        client._utilization_client = httpx.AsyncClient(transport=httpx.MockTransport(utilization_handler))
        try:
            first = await client.complete(messages=[{"role": "user", "content": "x"}])
            second = await client.complete(messages=[{"role": "user", "content": "y"}])
        finally:
            await client.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert first.provider == "openrouter-fp8"
    assert second.provider == "openrouter-fp8"
    assert calls == {"utilization": 1, "chutes": 0, "openrouter": 2}


def test_glm_skips_chutes_when_instance_count_is_three_or_less():
    calls = {"utilization": 0, "chutes": 0, "openrouter": 0}

    def utilization_handler(request: httpx.Request) -> httpx.Response:
        calls["utilization"] += 1
        return httpx.Response(
            200,
            json=[
                {
                    "chute_id": "08901219-159f-55a7-87cf-9d0d02744668",
                    "utilization_15m": 0.50,
                    "instance_count": 3,
                }
            ],
        )

    def chutes_handler(request: httpx.Request) -> httpx.Response:
        calls["chutes"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "chutes"}}]})

    def openrouter_handler(request: httpx.Request) -> httpx.Response:
        calls["openrouter"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "openrouter"}}]})

    settings = JudgeSettings(
        chutes_api_key="ck",
        openrouter_api_key="ok",
        glm_retry_count=0,
        chutes_utilization_15m_threshold=0.75,
        chutes_utilization_cache_seconds=60,
    )
    client = GLMProviderClient(settings)

    async def run():
        assert client._chutes_client is not None
        assert client._openrouter_client is not None
        assert client._utilization_client is not None
        await client._chutes_client.aclose()
        await client._openrouter_client.aclose()
        await client._utilization_client.aclose()
        client._chutes_client = httpx.AsyncClient(
            base_url=settings.chutes_base_url.rstrip("/"), transport=httpx.MockTransport(chutes_handler)
        )
        client._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(openrouter_handler)
        )
        client._utilization_client = httpx.AsyncClient(transport=httpx.MockTransport(utilization_handler))
        try:
            result = await client.complete(messages=[{"role": "user", "content": "x"}])
        finally:
            await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.provider == "openrouter-fp8"
    assert calls == {"utilization": 1, "chutes": 0, "openrouter": 1}


def test_glm_uses_chutes_when_15m_utilization_is_at_threshold():
    calls = {"utilization": 0, "chutes": 0, "openrouter": 0}

    def utilization_handler(request: httpx.Request) -> httpx.Response:
        calls["utilization"] += 1
        return httpx.Response(
            200,
            json=[
                {
                    "chute_id": "08901219-159f-55a7-87cf-9d0d02744668",
                    "utilization_15m": 0.75,
                }
            ],
        )

    def chutes_handler(request: httpx.Request) -> httpx.Response:
        calls["chutes"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "chutes"}}]})

    def openrouter_handler(request: httpx.Request) -> httpx.Response:
        calls["openrouter"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "openrouter"}}]})

    settings = JudgeSettings(
        chutes_api_key="ck",
        openrouter_api_key="ok",
        glm_retry_count=0,
        chutes_utilization_15m_threshold=0.75,
    )
    client = GLMProviderClient(settings)

    async def run():
        assert client._chutes_client is not None
        assert client._openrouter_client is not None
        assert client._utilization_client is not None
        await client._chutes_client.aclose()
        await client._openrouter_client.aclose()
        await client._utilization_client.aclose()
        client._chutes_client = httpx.AsyncClient(
            base_url=settings.chutes_base_url.rstrip("/"), transport=httpx.MockTransport(chutes_handler)
        )
        client._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(openrouter_handler)
        )
        client._utilization_client = httpx.AsyncClient(transport=httpx.MockTransport(utilization_handler))
        try:
            result = await client.complete(messages=[{"role": "user", "content": "x"}])
        finally:
            await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.provider == "chutes"
    assert calls == {"utilization": 1, "chutes": 1, "openrouter": 0}

class FailingCategoryService:
    async def prepare(self, sample):
        raise RuntimeError("category fail")


class EmptyPrepStore:
    async def get(self, prep_id, sample):
        return None


class FakeJudgeClient:
    async def score(self, *, model, messages, response_schema=None, schema_name=""):
        return JudgeRawResponse(model=model, provider="fake", raw=json.dumps({
            "correctness": 2,
            "grounding": 2,
            "progress": 2,
            "protocol": 2,
            "efficiency": 2,
        }))


def test_category_scoring_failure_can_be_caught_for_fixed_metric_fallback():
    request = ScoreBatchRequest(
        eval_run_id="run",
        batch_id="score-1",
        total_sample_count=1,
        judge_models=["z-ai/glm-5.1"],
        samples=[
            JudgeSample(
                sample_id="s1",
                prompt="task",
                previous_king_output="king",
                challenger_output="challenger",
                sample_index=0,
            )
        ],
    )
    try:
        asyncio.run(
            _score_samples_with_categories(
                client=FakeJudgeClient(),
                request=request,
                category_service=FailingCategoryService(),
                prep_store=EmptyPrepStore(),
            )
        )
    except RuntimeError as exc:
        assert "category fail" in str(exc)
    else:
        raise AssertionError("expected category failure")
