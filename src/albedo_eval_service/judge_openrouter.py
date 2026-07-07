from __future__ import annotations

import asyncio
import email.utils
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from loguru import logger

from .judge_config import JudgeSettings
from .judge_core import JUDGE_MODELS, JUDGE_PROVIDER_PINS


@dataclass(frozen=True)
class JudgeRawResponse:
    model: str
    provider: str | None
    raw: str
    error: str | None = None


class OpenRouterJudgeClient:
    def __init__(self, settings: JudgeSettings):
        if not settings.openrouter_api_key:
            raise ValueError("ALBEDO_JUDGE_OPENROUTER_API_KEY is required")
        self.settings = settings
        # Size the connection pool to the per-model concurrency across all models + the evaluator,
        # so raising max_concurrency_per_model actually parallelizes instead of queueing on sockets.
        pool = max(64, (len(JUDGE_MODELS) + 1) * settings.max_concurrency_per_model)
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterJudgeClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def score(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "albedo_answers",
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
        accept: Callable[[str], bool] | None = None,
    ) -> JudgeRawResponse:
        return await self._call(
            model=model, messages=messages, response_schema=response_schema,
            schema_name=schema_name, max_tokens=max_tokens, provider=provider, accept=accept,
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        accept: Callable[[str], bool] | None = None,
    ) -> JudgeRawResponse:
        # Generic completion (e.g. the question evaluator): `response_schema` forces JSON, `provider`
        # overrides the per-model pins.
        return await self._call(
            model=model, messages=messages, response_schema=response_schema,
            temperature=temperature, max_tokens=max_tokens, provider=provider, accept=accept,
        )

    async def _call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "albedo_answers",
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
        accept: Callable[[str], bool] | None = None,
    ) -> JudgeRawResponse:
        sem = self._semaphores.setdefault(
            model, asyncio.Semaphore(max(1, self.settings.max_concurrency_per_model))
        )
        async with sem:
            # Retry a 200-that-doesn't-parse (accept=False) up to parse_retries times; each retry is a
            # fresh call that may re-route to a different provider via allow_fallbacks.
            last: JudgeRawResponse | None = None
            for _ in range(max(1, self.settings.parse_retries)):
                last = await self._score_with_retries(
                    model=model, messages=messages, response_schema=response_schema,
                    schema_name=schema_name, temperature=temperature, max_tokens=max_tokens,
                    provider=provider,
                )
                if last.error is None and (accept is None or accept(last.raw)):
                    return last
            return last

    async def _score_with_retries(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "albedo_answers",
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
    ) -> JudgeRawResponse:
        last_error = ""
        for attempt in range(self.settings.retry_count + 1):
            try:
                return await self._score_once(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    schema_name=schema_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.settings.retry_count:
                    break
                await asyncio.sleep(
                    _retry_sleep_seconds(exc, attempt, self.settings.retry_backoff_seconds)
                )
        logger.warning(
            f"[judge-openrouter] retries exhausted model={model} "
            f"attempts={self.settings.retry_count + 1}, returning error: {last_error}"
        )
        return JudgeRawResponse(
            model=model, provider=_provider_name(model), raw="", error=last_error
        )

    async def _score_once(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "albedo_answers",
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
    ) -> JudgeRawResponse:
        provider_block = provider if provider is not None else JUDGE_PROVIDER_PINS.get(model, {})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "reasoning": {"enabled": False, "exclude": True},
            "provider": {**provider_block, "require_parameters": True},
        }
        # Force JSON whenever a schema is given; require_parameters routes only to providers that honor it.
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": response_schema},
            }
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        raw = _message_content(body.get("choices", []))
        provider = _provider_name(model)
        return JudgeRawResponse(model=model, provider=provider, raw=raw)


def _provider_name(model: str) -> str | None:
    order = JUDGE_PROVIDER_PINS.get(model, {}).get("order")
    if isinstance(order, list) and order:
        return str(order[0])
    return None


def _retry_sleep_seconds(exc: Exception, attempt: int, base_backoff_seconds: float) -> float:
    backoff = base_backoff_seconds * (2**attempt) * random.uniform(0.8, 1.2)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return max(backoff, _retry_after_seconds(exc.response.headers.get("retry-after")))
    return backoff


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _message_content(choices: list[dict[str, Any]]) -> str:
    if not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""
