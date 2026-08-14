from __future__ import annotations

import asyncio

from loguru import logger

from albedo_config import JudgeSettings, get_judge_settings
from albedo_eval_service.judge_llm_client import JudgeLLMClient, JudgeRawResponse

SANITY_DEFAULT_JUDGE_MODELS: tuple[str, ...] = ("z-ai/glm-5.2",)


def make_client(settings: JudgeSettings | None = None) -> JudgeLLMClient:
    return JudgeLLMClient(settings or get_judge_settings())


async def query_panel(
    client: JudgeLLMClient,
    system: str,
    user: str,
    models: tuple[str, ...] = SANITY_DEFAULT_JUDGE_MODELS,
    temperature: float | None = None,
) -> list[JudgeRawResponse]:
    logger.info("[sanity/panel] querying {} judges: {}", len(models), list(models))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _one(model: str) -> JudgeRawResponse:
        try:
            result = await client.complete(model=model, messages=messages, temperature=temperature)
            logger.info("[sanity/panel] {} ok chars={}", model, len(result.raw or ""))
            return result
        except Exception as exc:
            logger.warning("[sanity/panel] {} failed: {}: {}", model, type(exc).__name__, exc)
            return JudgeRawResponse(
                model=model, provider=None, raw="", error=f"{type(exc).__name__}: {exc}"
            )

    results = list(await asyncio.gather(*[_one(m) for m in models]))
    resolved = sum(1 for r in results if not r.error)
    logger.info("[sanity/panel] done resolved={}/{}", resolved, len(models))
    return results
