"""Sanity judge panel - runs a single-model prompt across the reused eval judge ensemble."""
from __future__ import annotations

import asyncio

from albedo_eval_service.judge_config import JudgeSettings, get_judge_settings
from albedo_eval_service.judge_core import JUDGE_MODELS
from albedo_eval_service.judge_openrouter import JudgeRawResponse, OpenRouterJudgeClient


def make_client(settings: JudgeSettings | None = None) -> OpenRouterJudgeClient:
    # Builds a judge client from the shared ALBEDO_JUDGE_* settings; caller owns its lifecycle.
    return OpenRouterJudgeClient(settings or get_judge_settings())


async def query_panel(
    client: OpenRouterJudgeClient,
    system: str,
    user: str,
    models: tuple[str, ...] = JUDGE_MODELS,
) -> list[JudgeRawResponse]:
    # Sends the prompt to all judges concurrently; one response per model, errors captured.
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _one(model: str) -> JudgeRawResponse:
        try:
            return await client.complete(model=model, messages=messages)
        except Exception as exc:  # noqa: BLE001 - a dead judge must not abort the panel
            return JudgeRawResponse(
                model=model, provider=None, raw="", error=f"{type(exc).__name__}: {exc}"
            )

    return list(await asyncio.gather(*[_one(m) for m in models]))
