from __future__ import annotations

import asyncio
import json

import httpx

from albedo_config import JudgeSettings
from albedo_eval_service.judge_llm_client import JudgeLLMClient


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
    assert schema_payload["model"] == "z-ai/glm-5.2"
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
    client = JudgeLLMClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.score(model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}])
        await client.score(
            model="z-ai/glm-5.2",
            messages=[{"role": "user", "content": "x"}],
            response_schema={"type": "object", "properties": {"answers": {"type": "array"}}},
        )
    finally:
        await client.aclose()
    return payloads


def test_provider_order_rotates_across_parse_attempts():
    """Regression: with 2 pinned providers a stride of transport_budget+1 was a
    multiple of the provider count, so rejected content retried the same provider."""
    orders = asyncio.run(_capture_orders_under_rejection())
    assert orders == [["deepseek", "cloudflare"], ["cloudflare", "deepseek"]]


async def _capture_orders_under_rejection():
    orders = []

    def handler(request: httpx.Request) -> httpx.Response:
        orders.append(json.loads(request.content.decode())["provider"]["order"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})

    settings = JudgeSettings(openrouter_api_key="test-key", retry_backoff_seconds=0)
    client = JudgeLLMClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.complete(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "x"}],
            provider={"order": ["deepseek", "cloudflare"], "allow_fallbacks": False},
            parse_retries=2,
            retry_count=0,
            accept=lambda raw: raw == "ok",
        )
    finally:
        await client.aclose()
    return orders


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
    client = JudgeLLMClient(settings)
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
        openrouter_api_key="test-key",
        engy_api_key="engy-key",
        retry_count=0,
        retry_backoff_seconds=0,
        parse_retries=1,
        **overrides,
    )
    client = JudgeLLMClient(settings)
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


def test_decode_bound_purposes_stay_on_openrouter():
    assert asyncio.run(_route("judge")) == [("openrouter", "z-ai/glm-5.2")]
    assert asyncio.run(_route("questions")) == [("openrouter", "z-ai/glm-5.2")]


def test_simulate_routes_only_the_primary_simulation_model_to_engy():
    assert asyncio.run(_route("simulate", model="deepseek/deepseek-v4-flash-0731")) == [
        ("engy", "deepseek-v4-flash-0731")
    ]
    assert asyncio.run(_route("simulate")) == [("openrouter", "z-ai/glm-5.2")]


def test_engy_transport_error_rescues_the_same_call_on_openrouter():
    hits, result = asyncio.run(_rescued_call())
    assert hits == [
        ("engy", "deepseek-v4-flash-0731"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ]
    assert result.error is None
    assert result.raw == "ok"


async def _rescued_call():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        result = await client.complete(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "x"}],
            purpose="simulate",
            eval_run_id="eval-1",
        )
    finally:
        await client.aclose()
    return hits, result


def test_model_outside_engy_models_stays_on_openrouter():
    hits = asyncio.run(_route("reference", model="deepseek/deepseek-v3.2"))
    assert hits == [("openrouter", "deepseek/deepseek-v3.2")]


async def _route(purpose, model="z-ai/glm-5.2"):
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None)
    await _swap_transports(client, settings, hits)
    try:
        await client.complete(
            model=model,
            messages=[{"role": "user", "content": "x"}],
            purpose=purpose,
            eval_run_id="eval-1",
        )
    finally:
        await client.aclose()
    return hits


def test_engy_falls_back_to_openrouter_after_max_errors_within_one_eval():
    hits = asyncio.run(_exhaust_engy_budget())
    # each failing engy attempt is rescued on OpenRouter in the same call; after
    # the budget (2) is spent, engy is not tried again for the rest of the eval
    assert [h[0] for h in hits] == [
        "engy",
        "openrouter",  # call 1: engy fails, rescued
        "engy",
        "openrouter",  # call 2: engy fails, budget spent
        "openrouter",  # call 3: engy skipped
        "openrouter",  # call 4
    ]


async def _exhaust_engy_budget():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=2)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for _ in range(4):
            await client.complete(
                model="z-ai/glm-5.2",
                messages=[{"role": "user", "content": "x"}],
                purpose="reference",
                eval_run_id="eval-1",
            )
    finally:
        await client.aclose()
    return hits


def test_budget_is_per_eval_so_overlapping_evaluations_still_abandon_engy():
    """Two evaluations in flight must not reset each other's budget."""
    hits = asyncio.run(_interleaved_evals())
    # each failing engy call is rescued on OpenRouter in-call; each eval spends its
    # OWN budget of 1. With a single shared counter engy was never abandoned.
    assert [h[0] for h in hits] == [
        "engy",
        "openrouter",  # eval-X spends its budget (rescued in-call)
        "engy",
        "openrouter",  # eval-Y spends its own, without resetting X's
        "openrouter",  # eval-X abandoned engy
        "openrouter",  # eval-Y likewise
    ]


async def _interleaved_evals():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=1)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for eval_id in ("eval-X", "eval-Y", "eval-X", "eval-Y"):
            await client.complete(
                model="z-ai/glm-5.2",
                messages=[{"role": "user", "content": "x"}],
                purpose="reference",
                eval_run_id=eval_id,
            )
    finally:
        await client.aclose()
    return hits


def test_error_budget_resets_on_next_evaluation():
    hits = asyncio.run(_budget_across_evals())
    assert [h[0] for h in hits] == [
        "engy",
        "openrouter",  # eval-1 call 1: engy fails, rescued in-call, budget (1) spent
        "openrouter",  # eval-1 call 2: engy skipped
        "engy",
        "openrouter",  # eval-2 call 1: fresh budget, tries engy again
        "openrouter",  # eval-2 call 2
    ]


async def _budget_across_evals():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=1)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for eval_id in ("eval-1", "eval-2"):
            for _ in range(2):
                await client.complete(
                    model="z-ai/glm-5.2",
                    messages=[{"role": "user", "content": "x"}],
                    purpose="reference",
                    eval_run_id=eval_id,
                )
    finally:
        await client.aclose()
    return hits


async def _swap_content_transports(client, settings, hits, engy_body):
    """OpenRouter answers 'ok'; engy answers 200 with `engy_body` (dict) verbatim."""

    def make(tag):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            hits.append((tag, body["model"]))
            if tag == "engy":
                return httpx.Response(200, json=engy_body)
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


def test_engy_empty_200_counts_and_retries_on_openrouter():
    """The miner-restart failure mode: 200 with no choices/usage."""
    hits, result, client_errors = asyncio.run(_content_failure({}))
    assert hits == [
        ("engy", "deepseek-v4-flash-0731"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ]
    assert result.error is None and result.raw == "ok"
    assert client_errors == {("eval-1", "deepseek/deepseek-v4-flash-0731"): 1}


def test_engy_garbage_rejected_by_accept_retries_but_is_not_charged_to_engy():
    """A delivered-but-rejected answer is the model's fault, not engy's."""
    garbage = {"choices": [{"message": {"content": "### assistant I think we should..."}}]}
    hits, result, client_errors = asyncio.run(
        _content_failure(garbage, accept=lambda raw: raw == "ok")
    )
    assert hits == [
        ("engy", "deepseek-v4-flash-0731"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ]
    assert result.raw == "ok"
    assert client_errors == {}


async def _content_failure(engy_body, accept=None):
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None)
    await _swap_content_transports(client, settings, hits, engy_body)
    try:
        result = await client.complete(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "x"}],
            purpose="simulate",
            eval_run_id="eval-1",
            accept=accept,
        )
    finally:
        await client.aclose()
    return hits, result, dict(client._engy_errors)


def test_empty_engy_responses_alone_exhaust_the_engy_budget():
    hits = asyncio.run(_empty_responses_exhaust())
    # budget 2: calls 1-2 try engy (empty) and rescue on OR; calls 3-4 skip engy
    assert [h[0] for h in hits] == [
        "engy",
        "openrouter",
        "engy",
        "openrouter",
        "openrouter",
        "openrouter",
    ]


async def _empty_responses_exhaust():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=2)
    await _swap_content_transports(client, settings, hits, {})
    try:
        for _ in range(4):
            await client.complete(
                model="deepseek/deepseek-v4-flash-0731",
                messages=[{"role": "user", "content": "x"}],
                purpose="simulate",
                eval_run_id="eval-1",
            )
    finally:
        await client.aclose()
    return hits


def test_rejected_content_never_exhausts_the_engy_budget():
    """Regression: contract rejections used to kill engy minutes into every eval."""
    hits = asyncio.run(_rejections_never_exhaust())
    # engy is healthy, so every one of the 4 calls still gets to try it
    assert [h[0] for h in hits] == ["engy", "openrouter"] * 4


async def _rejections_never_exhaust():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=2)
    garbage = {"choices": [{"message": {"content": "nope"}}]}
    await _swap_content_transports(client, settings, hits, garbage)
    try:
        for _ in range(4):
            await client.complete(
                model="deepseek/deepseek-v4-flash-0731",
                messages=[{"role": "user", "content": "x"}],
                purpose="simulate",
                eval_run_id="eval-1",
                accept=lambda raw: raw == "ok",
            )
    finally:
        await client.aclose()
    return hits


def test_engy_gets_one_shot_per_call_then_the_call_stays_on_openrouter():
    """Re-asking engy the same prompt at temperature 0 just repeats the rejection,
    so later parse attempts of the SAME call must walk the pinned providers instead."""
    hits = asyncio.run(_rejection_across_parse_attempts())
    assert hits == [
        ("engy", "deepseek-v4-flash-0731", "deepseek"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731", "deepseek"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731", "cloudflare"),
    ]


async def _rejection_across_parse_attempts():
    hits: list[tuple[str, str, str]] = []

    def make(tag):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            order = (body.get("provider") or {}).get("order") or ["-"]
            hits.append((tag, body["model"], order[0]))
            return httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})

        return handler

    settings, client = _engy_client(None)
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
    try:
        await client.complete(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "x"}],
            purpose="simulate",
            eval_run_id="eval-1",
            provider={"order": ["deepseek", "cloudflare"], "allow_fallbacks": False},
            parse_retries=2,
            accept=lambda raw: raw == "ok",
        )
    finally:
        await client.aclose()
    return hits


def test_reference_and_simulate_budgets_are_independent():
    """glm reference failures must not disable deepseek for simulate, or vice versa."""
    hits = asyncio.run(_two_models_one_eval())
    # glm burns its own budget of 1, deepseek still gets its first engy try afterwards
    assert hits == [
        ("engy", "glm-5.2"),
        ("openrouter", "z-ai/glm-5.2"),
        ("openrouter", "z-ai/glm-5.2"),
        ("engy", "deepseek-v4-flash-0731"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ]


async def _two_models_one_eval():
    hits: list[tuple[str, str]] = []
    settings, client = _engy_client(None, engy_max_errors=1)
    await _swap_transports(client, settings, hits)
    client._engy_should_fail = True
    try:
        for _ in range(2):
            await client.complete(
                model="z-ai/glm-5.2",
                messages=[{"role": "user", "content": "x"}],
                purpose="reference",
                eval_run_id="eval-1",
            )
        await client.complete(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "x"}],
            purpose="simulate",
            eval_run_id="eval-1",
        )
    finally:
        await client.aclose()
    return hits
