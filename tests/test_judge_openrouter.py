from __future__ import annotations

import asyncio
import json

import httpx

from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_openrouter import OpenRouterJudgeClient


def test_openrouter_payload_respects_provider_structured_output_support():
    payloads = asyncio.run(_capture_payloads())

    plain_payload = payloads[0]
    assert plain_payload["model"] == "z-ai/glm-5.2"
    assert "order" not in plain_payload["provider"]
    assert plain_payload["provider"]["quantizations"] == ["fp8"]
    assert plain_payload["provider"]["allow_fallbacks"] is True
    assert plain_payload["provider"]["require_parameters"] is True
    assert "response_format" not in plain_payload

    schema_payload = payloads[1]
    assert schema_payload["model"] == "qwen/qwen3.5-397b-a17b"
    assert "order" not in schema_payload["provider"]
    assert schema_payload["provider"]["quantizations"] == ["fp8"]
    assert schema_payload["provider"]["allow_fallbacks"] is True
    assert schema_payload["provider"]["require_parameters"] is True
    assert schema_payload["response_format"]["type"] == "json_schema"


async def _capture_payloads():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode()))
        raw = json.dumps({"answers": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    settings = JudgeSettings(openrouter_api_key="test-key")
    client = OpenRouterJudgeClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.score(model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}])
        await client.score(
            model="qwen/qwen3.5-397b-a17b",
            messages=[{"role": "user", "content": "x"}],
            response_schema={"type": "object", "properties": {"answers": {"type": "array"}}},
        )
    finally:
        await client.aclose()
    return payloads


def test_provider_order_rotates_across_retries():
    orders = asyncio.run(_capture_orders_under_failures())
    assert orders[0] == ["A", "B", "C"]
    assert orders[1] == ["B", "C", "A"]
    assert orders[2] == ["C", "A", "B"]


async def _capture_orders_under_failures():
    orders = []

    def handler(request: httpx.Request) -> httpx.Response:
        orders.append(json.loads(request.content.decode())["provider"]["order"])
        if len(orders) < 3:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings = JudgeSettings(openrouter_api_key="test-key", retry_count=2, retry_backoff_seconds=0)
    client = OpenRouterJudgeClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.complete(
            model="z-ai/glm-5.2",
            messages=[{"role": "user", "content": "x"}],
            provider={"order": ["A", "B", "C"], "allow_fallbacks": True, "quantizations": ["fp8"]},
        )
    finally:
        await client.aclose()
    return orders


# --------------------------------------------------------------- engy routing for SOTA reference


def _engy_client(handler, **overrides):
    """A client whose OpenRouter and engy legs both hit MockTransport handlers."""
    settings = JudgeSettings(
        openrouter_api_key="test-key", engy_api_key="engy-key",
        retry_count=0, retry_backoff_seconds=0, parse_retries=1, **overrides,
    )
    client = OpenRouterJudgeClient(settings)
    return settings, client


async def _swap_transports(client, settings, hits):
    def make(tag):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            hits.append((tag, body["model"]))
            if tag == "engy" and getattr(client, "_engy_should_fail", False):
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return handler

    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(make("openrouter")),
    )
    await client._engy.aclose()
    client._engy = httpx.AsyncClient(
        base_url=settings.engy_base_url.rstrip("/"),
        transport=httpx.MockTransport(make("engy")),
    )


def test_reference_goes_to_engy_with_bare_model_name():
    hits = asyncio.run(_route("reference"))
    assert hits == [("engy", "glm-5.2")]


def test_non_reference_purposes_stay_on_openrouter():
    assert asyncio.run(_route("judge")) == [("openrouter", "z-ai/glm-5.2")]
    assert asyncio.run(_route("questions")) == [("openrouter", "z-ai/glm-5.2")]
    assert asyncio.run(_route("simulate")) == [("openrouter", "z-ai/glm-5.2")]


def test_model_outside_engy_models_stays_on_openrouter():
    hits = asyncio.run(_route("reference", model="deepseek/deepseek-v3.2"))
    assert hits == [("openrouter", "deepseek/deepseek-v3.2")]


async def _route(purpose, model="z-ai/glm-5.2"):
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None)
    await _swap_transports(client, settings, hits)
    try:
        await client.complete(
            model=model, messages=[{"role": "user", "content": "x"}],
            purpose=purpose, eval_run_id="eval-1",
        )
    finally:
        await client.aclose()
    return hits


def test_engy_falls_back_to_openrouter_after_max_errors_within_one_eval():
    hits = asyncio.run(_exhaust_engy_budget())
    engy_attempts = [h for h in hits if h[0] == "engy"]
    # budget is 2, so engy is tried twice then abandoned for the rest of the eval
    assert len(engy_attempts) == 2
    assert hits[-1][0] == "openrouter"


async def _exhaust_engy_budget():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=2)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for _ in range(4):
            await client.complete(
                model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}],
                purpose="reference", eval_run_id="eval-1",
            )
    finally:
        await client.aclose()
    return hits


def test_budget_is_per_eval_so_overlapping_evaluations_still_abandon_engy():
    """Two evaluations in flight must not reset each other's budget."""
    hits = asyncio.run(_interleaved_evals())
    # retry_count=0 here, so a failed engy call returns an error and the divert
    # shows on that eval's NEXT call. Each eval spends its own budget of 1.
    # With a single shared counter this was ["engy"] * 4 -- engy never abandoned.
    assert [h[0] for h in hits] == [
        "engy",         # eval-X spends its budget
        "engy",         # eval-Y spends its own, without resetting X's
        "openrouter",   # eval-X abandoned engy
        "openrouter",   # eval-Y likewise
    ]


async def _interleaved_evals():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=1)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for eval_id in ("eval-X", "eval-Y", "eval-X", "eval-Y"):
            await client.complete(
                model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}],
                purpose="reference", eval_run_id=eval_id,
            )
    finally:
        await client.aclose()
    return hits


def test_error_budget_resets_on_next_evaluation():
    hits = asyncio.run(_budget_across_evals())
    assert [h[0] for h in hits] == ["engy", "openrouter", "engy", "openrouter"]


async def _budget_across_evals():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=1)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for eval_id in ("eval-1", "eval-2"):
            for _ in range(2):
                await client.complete(
                    model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}],
                    purpose="reference", eval_run_id=eval_id,
                )
    finally:
        await client.aclose()
    return hits
