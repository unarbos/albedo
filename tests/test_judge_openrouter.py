from __future__ import annotations

import asyncio
import json

import httpx

from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_core import METRIC_KEYS
from albedo_eval_service.judge_openrouter import OpenRouterJudgeClient


def test_openrouter_payload_respects_provider_structured_output_support():
    payloads = asyncio.run(_capture_payloads())

    glm_payload = payloads[0]
    assert glm_payload["model"] == "z-ai/glm-5.1"
    assert glm_payload["provider"]["order"] == ["baidu"]
    assert glm_payload["provider"]["quantizations"] == ["fp8"]
    assert glm_payload["provider"]["allow_fallbacks"] is False
    assert "response_format" not in glm_payload

    qwen_payload = payloads[1]
    assert qwen_payload["model"] == "qwen/qwen3.5-397b-a17b"
    assert qwen_payload["provider"]["order"] == ["deepinfra"]
    assert qwen_payload["response_format"]["type"] == "json_schema"


async def _capture_payloads():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode()))
        raw = json.dumps({metric: 0 for metric in METRIC_KEYS})
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    settings = JudgeSettings(openrouter_api_key="test-key")
    client = OpenRouterJudgeClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.score(model="z-ai/glm-5.1", messages=[{"role": "user", "content": "x"}])
        await client.score(model="qwen/qwen3.5-397b-a17b", messages=[{"role": "user", "content": "x"}])
    finally:
        await client.aclose()
    return payloads
