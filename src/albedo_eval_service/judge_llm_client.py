from __future__ import annotations

import asyncio
import collections
import email.utils
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from loguru import logger

from albedo_config import JudgeSettings
from albedo_config.models import JUDGE_MODELS, JUDGE_PROVIDER_PINS


@dataclass(frozen=True)
class JudgeRawResponse:
    model: str
    provider: str | None
    raw: str
    error: str | None = None


ENGY_PURPOSES = frozenset({"reference", "simulate"})


class JudgeLLMClient:
    def __init__(self, settings: JudgeSettings):
        if not settings.openrouter_api_key:
            raise ValueError("ALBEDO_JUDGE_OPENROUTER_API_KEY is required")
        self.settings = settings
        pool = max(64, (len(JUDGE_MODELS) + 1) * settings.max_concurrency_per_model)
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._engy = (
            httpx.AsyncClient(
                base_url=settings.engy_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.engy_api_key}"},
                timeout=httpx.Timeout(settings.request_timeout_seconds),
                limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
            )
            if settings.engy_api_key
            else None
        )
        self._engy_models = {m.strip() for m in settings.engy_models.split(",") if m.strip()}
        self._engy_errors: collections.Counter[tuple[str, str]] = collections.Counter()
        logger.info(
            f"[judge-llm] engy routing for purposes={sorted(ENGY_PURPOSES)}: "
            + (
                f"ON models={sorted(self._engy_models)} url={settings.engy_base_url} "
                f"max_errors={settings.engy_max_errors}"
                if self._engy is not None
                else "OFF (no engy api key)"
            )
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._engy is not None:
            await self._engy.aclose()

    def _use_engy(self, purpose: str, model: str, eval_run_id: str) -> bool:
        if self._engy is None or purpose not in ENGY_PURPOSES or model not in self._engy_models:
            return False
        if purpose == "simulate" and model != self.settings.simulation_model:
            return False
        return self._engy_errors[(eval_run_id, model)] < self.settings.engy_max_errors

    async def __aenter__(self) -> "JudgeLLMClient":
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
        purpose: str = "judge",
    ) -> JudgeRawResponse:
        return await self._call(
            model=model,
            messages=messages,
            response_schema=response_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            provider=provider,
            accept=accept,
            purpose=purpose,
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
        purpose: str = "other",
        parse_retries: int | None = None,
        retry_count: int | None = None,
        eval_run_id: str = "",
    ) -> JudgeRawResponse:
        return await self._call(
            model=model,
            messages=messages,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            accept=accept,
            purpose=purpose,
            parse_retries=parse_retries,
            retry_count=retry_count,
            eval_run_id=eval_run_id,
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
        purpose: str = "other",
        parse_retries: int | None = None,
        retry_count: int | None = None,
        eval_run_id: str = "",
    ) -> JudgeRawResponse:
        sem = self._semaphores.setdefault(
            model, asyncio.Semaphore(max(1, self.settings.max_concurrency_per_model))
        )
        parse_budget = self.settings.parse_retries if parse_retries is None else parse_retries
        transport_budget = self.settings.retry_count if retry_count is None else retry_count
        async with sem:
            last: JudgeRawResponse | None = None
            engy_spent = False  # engy gets one shot per call; re-asking at t=0 repeats itself
            for parse_attempt in range(max(1, parse_budget)):
                last = await self._score_with_retries(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    schema_name=schema_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                    base_shift=parse_attempt,
                    purpose=purpose,
                    retry_count=transport_budget,
                    eval_run_id=eval_run_id,
                    force_openrouter=engy_spent,
                )
                if _usable(last, accept):
                    return last
                if last.provider == "engy":
                    engy_spent = True
                    logger.warning(
                        f"[judge-llm] engy content rejected (not charged to engy health) "
                        f"eval_run_id={eval_run_id} model={model} purpose={purpose}, "
                        f"retrying this call on openrouter"
                    )
                    last = await self._score_with_retries(
                        model=model,
                        messages=messages,
                        response_schema=response_schema,
                        schema_name=schema_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        provider=provider,
                        base_shift=parse_attempt,
                        purpose=purpose,
                        retry_count=transport_budget,
                        eval_run_id=eval_run_id,
                        force_openrouter=True,
                    )
                    if _usable(last, accept):
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
        base_shift: int = 0,
        purpose: str = "other",
        retry_count: int | None = None,
        eval_run_id: str = "",
        force_openrouter: bool = False,
    ) -> JudgeRawResponse:
        transport_budget = self.settings.retry_count if retry_count is None else retry_count
        last_error = ""
        for attempt in range(transport_budget + 1):
            try:
                return await self._score_once(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    schema_name=schema_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                    provider_shift=base_shift + attempt,
                    purpose=purpose,
                    eval_run_id=eval_run_id,
                    force_openrouter=force_openrouter,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= transport_budget:
                    break
                await asyncio.sleep(
                    _retry_sleep_seconds(exc, attempt, self.settings.retry_backoff_seconds)
                )
        logger.warning(
            f"[judge-llm] retries exhausted model={model} "
            f"attempts={transport_budget + 1}, returning error: {last_error}"
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
        provider_shift: int = 0,
        purpose: str = "other",
        eval_run_id: str = "",
        force_openrouter: bool = False,
    ) -> JudgeRawResponse:
        on_engy = not force_openrouter and self._use_engy(purpose, model, eval_run_id)
        client = self._engy if on_engy else self._client
        provider_block = provider if provider is not None else JUDGE_PROVIDER_PINS.get(model, {})
        provider_block = _rotate_order(provider_block, provider_shift)
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[-1] if on_engy else model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "reasoning": {"enabled": False, "exclude": True},
            "provider": {**provider_block, "require_parameters": True},
            "usage": {"include": True},
        }
        if model.startswith("openai/"):
            del payload["temperature"]
            payload["provider"] = dict(provider_block)
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": response_schema},
            }
        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except Exception as exc:
            if not on_engy:
                raise
            self._engy_errors[(eval_run_id, model)] += 1
            logger.warning(
                f"[judge-llm] engy error {self._engy_errors[(eval_run_id, model)]}/"
                f"{self.settings.engy_max_errors} eval_run_id={eval_run_id} "
                f"model={model}, retrying this call on openrouter: "
                f"{type(exc).__name__}: {exc}"
            )
            on_engy = False
            payload["model"] = model
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        logger.debug(
            f"[judge-llm] usage purpose={purpose} model={model} "
            f"prompt_tokens={usage.get('prompt_tokens')} cached_tokens={cached} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"reasoning_tokens={reasoning} "
            f"cost={float(usage.get('cost') or 0.0):.8f}"
        )
        raw = _message_content(body.get("choices", []))
        if on_engy and not raw.strip():
            self._engy_errors[(eval_run_id, model)] += 1
            logger.warning(
                f"[judge-llm] engy returned empty content "
                f"{self._engy_errors[(eval_run_id, model)]}/{self.settings.engy_max_errors} "
                f"eval_run_id={eval_run_id} model={model} body={str(body)[:200]}"
            )
        provider = "engy" if on_engy else _provider_name(model)
        return JudgeRawResponse(model=model, provider=provider, raw=raw)


def _usable(result: JudgeRawResponse, accept: Callable[[str], bool] | None) -> bool:
    if result.error is not None:
        return False
    if result.provider == "engy" and not result.raw.strip():
        return False
    return accept is None or accept(result.raw)


def _rotate_order(provider: dict[str, Any], shift: int) -> dict[str, Any]:
    order = provider.get("order")
    if not shift or not isinstance(order, list) or len(order) < 2:
        return provider
    k = shift % len(order)
    return {**provider, "order": order[k:] + order[:k]}


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
