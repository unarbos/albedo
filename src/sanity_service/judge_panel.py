"""Sanity judge panel - runs a single-model prompt across the reused eval judge ensemble."""

from __future__ import annotations

import asyncio

from loguru import logger

from albedo_eval_service.judge_config import JudgeSettings, get_judge_settings
from albedo_eval_service.judge_core import JUDGE_MODELS
from albedo_eval_service.judge_openrouter import JudgeRawResponse, OpenRouterJudgeClient


def make_client(settings: JudgeSettings | None = None) -> OpenRouterJudgeClient:
    # Builds a judge client from the shared ALBEDO_JUDGE_* settings; caller owns its lifecycle.
    return OpenRouterJudgeClient(settings or get_judge_settings())


async def query_panel(client: OpenRouterJudgeClient, system: str, user: str, models: tuple[str, ...] = JUDGE_MODELS, temperature: float | None = None,) -> list[JudgeRawResponse]:
    # Sends the prompt to all judges concurrently; one response per model, errors captured.
    # temperature overrides the judge default per call (used by the injection re-check for variance).
    logger.info("[sanity/panel] querying {} judges: {}", len(models), list(models))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _one(model: str) -> JudgeRawResponse:
        try:
            result = await client.complete(model=model, messages=messages, temperature=temperature)
            logger.info("[sanity/panel] {} ok chars={}", model, len(result.raw or ""))
            return result
        except Exception as exc:  # noqa: BLE001 - a dead judge must not abort the panel
            logger.warning("[sanity/panel] {} failed: {}: {}", model, type(exc).__name__, exc)
            return JudgeRawResponse(model=model, provider=None, raw="", error=f"{type(exc).__name__}: {exc}")

    results = list(await asyncio.gather(*[_one(m) for m in models]))
    resolved = sum(1 for r in results if not r.error)
    logger.info("[sanity/panel] done resolved={}/{}", resolved, len(models))
    return results
