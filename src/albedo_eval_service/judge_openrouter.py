from __future__ import annotations

import asyncio
import email.utils
import random
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import httpx
from loguru import logger

from .judge_config import JudgeSettings
from .judge_core import JUDGE_MODELS, JUDGE_PROVIDER_PINS


# Per-eval cost isolation for the shared judge-api microservice.
# Request handlers set `_current_eval_run_id` so nested client.score/complete
# calls attribute LiteLLM spend to the correct eval without plumbing the id
# through every call site. Buckets are keyed by eval_run_id and TTL-pruned.
_current_eval_run_id: ContextVar[str] = ContextVar("albedo_eval_run_id", default="")
_COST_TTL_SECONDS = 24 * 60 * 60


@dataclass
class _CostBucket:
    total_cost: float = 0.0
    request_count: int = 0
    per_model: dict[str, float] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.monotonic)


class _CostStore:
    def __init__(self, ttl_seconds: float = _COST_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._buckets: dict[str, _CostBucket] = {}
        self._lock = threading.Lock()

    def record(self, eval_run_id: str, model: str, cost: float) -> None:
        if not eval_run_id:
            logger.warning(
                "[judge-openrouter] cost not attributed (empty eval_run_id) "
                f"model={model} cost={cost:.8f}"
            )
            return
        with self._lock:
            self._sweep_locked()
            bucket = self._buckets.setdefault(eval_run_id, _CostBucket())
            bucket.total_cost += cost
            bucket.request_count += 1
            bucket.per_model[model] = round(bucket.per_model.get(model, 0.0) + cost, 8)
            bucket.updated_at = time.monotonic()

    def snapshot(self, eval_run_id: str) -> dict[str, object]:
        with self._lock:
            self._sweep_locked()
            bucket = self._buckets.get(eval_run_id)
            if bucket is None:
                return {
                    "eval_run_id": eval_run_id,
                    "total_cost": 0.0,
                    "request_count": 0,
                    "per_model": {},
                }
            return {
                "eval_run_id": eval_run_id,
                "total_cost": round(bucket.total_cost, 8),
                "request_count": bucket.request_count,
                "per_model": {
                    k: round(v, 8) for k, v in sorted(bucket.per_model.items())
                },
            }

    def reset(self, eval_run_id: str) -> None:
        with self._lock:
            self._buckets.pop(eval_run_id, None)

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        expired = [
            eid
            for eid, bucket in self._buckets.items()
            if now - bucket.updated_at > self._ttl_seconds
        ]
        for eid in expired:
            self._buckets.pop(eid, None)


_COST_STORE = _CostStore()


def get_eval_cost_snapshot(eval_run_id: str) -> dict[str, object]:
    """Return accumulated judge cost for one eval_run_id."""
    return _COST_STORE.snapshot(eval_run_id)


def reset_eval_cost(eval_run_id: str) -> None:
    """Drop a cost bucket after the eval Job has fetched its snapshot."""
    _COST_STORE.reset(eval_run_id)


@contextmanager
def eval_run_id_scope(eval_run_id: str) -> Iterator[None]:
    """Bind cost attribution to eval_run_id for the current async/task context."""
    token = _current_eval_run_id.set(eval_run_id or "")
    try:
        yield
    finally:
        _current_eval_run_id.reset(token)


def current_eval_run_id() -> str:
    return _current_eval_run_id.get()


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
        purpose: str = "judge",
    ) -> JudgeRawResponse:
        return await self._call(
            model=model, messages=messages, response_schema=response_schema,
            schema_name=schema_name, max_tokens=max_tokens, provider=provider, accept=accept,
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
    ) -> JudgeRawResponse:
        return await self._call(
            model=model, messages=messages, response_schema=response_schema,
            temperature=temperature, max_tokens=max_tokens, provider=provider, accept=accept,
            purpose=purpose, parse_retries=parse_retries, retry_count=retry_count,
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
    ) -> JudgeRawResponse:
        sem = self._semaphores.setdefault(
            model, asyncio.Semaphore(max(1, self.settings.max_concurrency_per_model))
        )
        parse_budget = self.settings.parse_retries if parse_retries is None else parse_retries
        transport_budget = self.settings.retry_count if retry_count is None else retry_count
        async with sem:
            last: JudgeRawResponse | None = None
            for parse_attempt in range(max(1, parse_budget)):
                last = await self._score_with_retries(
                    model=model, messages=messages, response_schema=response_schema,
                    schema_name=schema_name, temperature=temperature, max_tokens=max_tokens,
                    provider=provider,
                    base_shift=parse_attempt * (transport_budget + 1),
                    purpose=purpose,
                    retry_count=transport_budget,
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
        base_shift: int = 0,
        purpose: str = "other",
        retry_count: int | None = None,
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
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= transport_budget:
                    break
                await asyncio.sleep(
                    _retry_sleep_seconds(exc, attempt, self.settings.retry_backoff_seconds)
                )
        logger.warning(
            f"[judge-openrouter] retries exhausted model={model} "
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
    ) -> JudgeRawResponse:
        provider_block = provider if provider is not None else JUDGE_PROVIDER_PINS.get(model, {})
        provider_block = _rotate_order(provider_block, provider_shift)
        payload: dict[str, Any] = {
            "model": model,
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
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        # LiteLLM exposes the computed cost only in the x-litellm-response-cost
        # response header for non-streaming requests (it is NOT placed in the
        # JSON usage body). OpenRouter puts it in usage.cost. Prefer the header
        # when present so the same code works against both gateways.
        cost_str = response.headers.get("x-litellm-response-cost")
        if cost_str is None or cost_str == "":
            cost_str = usage.get("cost")
        try:
            cost = float(cost_str) if cost_str is not None else 0.0
        except (TypeError, ValueError):
            cost = 0.0
        # Attribute spend to the current eval_run_id (set by judge-api handlers
        # via eval_run_id_scope). Empty id is logged and skipped — never dumped
        # into a shared "default" bucket on the multi-tenant judge service.
        _COST_STORE.record(current_eval_run_id(), model, cost)
        logger.debug(
            f"[judge-openrouter] usage purpose={purpose} model={model} "
            f"eval_run_id={current_eval_run_id() or '-'} "
            f"prompt_tokens={usage.get('prompt_tokens')} cached_tokens={cached} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"reasoning_tokens={reasoning} "
            f"cost={cost:.8f}"
        )
        raw = _message_content(body.get("choices", []))
        provider = _provider_name(model)
        return JudgeRawResponse(model=model, provider=provider, raw=raw)


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
