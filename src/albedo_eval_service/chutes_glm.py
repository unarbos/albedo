from __future__ import annotations

import asyncio
import email.utils
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from .judge_config import JudgeSettings


@dataclass(frozen=True)
class GLMRawResponse:
    model: str
    provider: str
    raw: str
    error: str | None = None


class GLMProviderClient:
    """Chutes-primary GLM 5.2 client with OpenRouter FP8 fallback."""

    def __init__(self, settings: JudgeSettings):
        self.settings = settings
        self._chutes_client: httpx.AsyncClient | None = None
        self._openrouter_client: httpx.AsyncClient | None = None
        self._utilization_client: httpx.AsyncClient | None = None
        if settings.chutes_api_key:
            self._chutes_client = httpx.AsyncClient(
                base_url=settings.chutes_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.chutes_api_key}"},
                timeout=httpx.Timeout(settings.glm_request_timeout_seconds),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        if settings.openrouter_api_key:
            self._openrouter_client = httpx.AsyncClient(
                base_url=_openrouter_base_url(settings.openrouter_base_url),
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                timeout=httpx.Timeout(settings.glm_request_timeout_seconds),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        if settings.chutes_utilization_check_enabled and settings.chutes_utilization_url:
            self._utilization_client = httpx.AsyncClient(
                timeout=httpx.Timeout(min(settings.glm_request_timeout_seconds, 15.0)),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        self._semaphore = asyncio.Semaphore(max(1, settings.glm_max_concurrency))
        self._chutes_error_count = 0
        self._chutes_error_lock = asyncio.Lock()
        self._chutes_disabled_logged = False
        self._utilization_lock = asyncio.Lock()
        self._utilization_cache_expires_at = 0.0
        self._utilization_disabled_reason = ""

    async def aclose(self) -> None:
        if self._chutes_client is not None:
            await self._chutes_client.aclose()
        if self._openrouter_client is not None:
            await self._openrouter_client.aclose()
        if self._utilization_client is not None:
            await self._utilization_client.aclose()

    async def __aenter__(self) -> "GLMProviderClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GLMRawResponse:
        async with self._semaphore:
            chutes_error = ""
            chutes_disabled_reason = ""
            if self._chutes_client is not None:
                chutes_disabled_reason = await self._chutes_disabled_reason()
                if not chutes_disabled_reason:
                    response = await self._complete_with_retries(
                        provider="chutes",
                        model=self.settings.glm52_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if response.error is None:
                        return response
                    chutes_error = response.error or "chutes request failed"
                    await self._record_chutes_error(chutes_error)
                elif not self._chutes_disabled_logged:
                    self._chutes_disabled_logged = True
                    logger.warning(
                        "GLM 5.2 category generation routing directly to OpenRouter FP8: "
                        "primary_provider=chutes primary_model={} disabled_reason={} "
                        "fallback_model={}",
                        self.settings.glm52_model,
                        chutes_disabled_reason,
                        self.settings.openrouter_glm52_model,
                    )
            openrouter_error = ""
            if self._openrouter_client is not None:
                if chutes_error:
                    logger.warning(
                        "GLM 5.2 category generation provider fallback: primary_provider=chutes "
                        "primary_model={} fallback_provider=openrouter-fp8 fallback_model={} "
                        "reason={}",
                        self.settings.glm52_model,
                        self.settings.openrouter_glm52_model,
                        chutes_error,
                    )
                response = await self._complete_with_retries(
                    provider="openrouter-fp8",
                    model=self.settings.openrouter_glm52_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response.error is None:
                    return response
                openrouter_error = response.error or "openrouter request failed"
                logger.warning(
                    "GLM 5.2 category generation fallback failed: provider=openrouter-fp8 "
                    "model={} reason={}",
                    self.settings.openrouter_glm52_model,
                    openrouter_error,
                )
            error_parts = []
            if chutes_error:
                error_parts.append(f"chutes: {chutes_error}")
            elif chutes_disabled_reason:
                error_parts.append(f"chutes: {chutes_disabled_reason}")
            elif self._chutes_client is None:
                error_parts.append("chutes: missing CHUTES_API_KEY")
            if openrouter_error:
                error_parts.append(f"openrouter-fp8: {openrouter_error}")
            elif self._openrouter_client is None:
                error_parts.append("openrouter-fp8: missing OPENROUTER_API_KEY")
            return GLMRawResponse(
                model=self.settings.glm52_model,
                provider="glm-unavailable",
                raw="",
                error="; ".join(error_parts) or "no GLM provider configured",
            )

    async def _chutes_disabled_reason(self) -> str:
        threshold = max(1, self.settings.chutes_disable_after_errors)
        async with self._chutes_error_lock:
            if self._chutes_error_count >= threshold:
                return f"disabled after {self._chutes_error_count} errors"
        utilization_reason = await self._chutes_utilization_disabled_reason()
        if utilization_reason:
            return utilization_reason
        return ""

    async def _chutes_utilization_disabled_reason(self) -> str:
        if self._utilization_client is None:
            return ""
        now = time.monotonic()
        async with self._utilization_lock:
            if now < self._utilization_cache_expires_at:
                return self._utilization_disabled_reason
            self._utilization_cache_expires_at = (
                now + max(1.0, self.settings.chutes_utilization_cache_seconds)
            )
            self._utilization_disabled_reason = ""
            try:
                response = await self._utilization_client.get(
                    self.settings.chutes_utilization_url
                )
                response.raise_for_status()
                utilization, instance_count = _chute_utilization_snapshot(
                    response.json(), chute_id=self.settings.glm52_chute_id
                )
            except Exception as exc:
                logger.warning(
                    "GLM 5.2 Chutes utilization check failed; keeping Chutes enabled. reason={}",
                    f"{type(exc).__name__}: {exc}",
                )
                return ""
            threshold = self.settings.chutes_utilization_15m_threshold
            reasons = []
            if utilization is not None and utilization > threshold:
                reasons.append(f"utilization_15m={utilization:.3f} above threshold={threshold:.3f}")
            if instance_count is not None and instance_count <= 100:
                reasons.append(f"instance_count={instance_count} <= 100")
            if reasons:
                self._utilization_disabled_reason = "; ".join(reasons)
            return self._utilization_disabled_reason

    async def _record_chutes_error(self, error: str) -> None:
        threshold = max(1, self.settings.chutes_disable_after_errors)
        async with self._chutes_error_lock:
            self._chutes_error_count += 1
            error_count = self._chutes_error_count
        if error_count == threshold:
            logger.warning(
                "GLM 5.2 Chutes provider disabled after {} errors; "
                "routing subsequent requests to OpenRouter FP8. last_error={}",
                error_count,
                error,
            )

    async def _complete_with_retries(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> GLMRawResponse:
        last_error = ""
        for attempt in range(self.settings.glm_retry_count + 1):
            try:
                return await self._complete_once(
                    provider=provider,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.settings.glm_retry_count:
                    break
                await asyncio.sleep(
                    _retry_sleep_seconds(exc, attempt, self.settings.retry_backoff_seconds)
                )
        return GLMRawResponse(model=model, provider=provider, raw="", error=last_error)

    async def _complete_once(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> GLMRawResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.glm_temperature if temperature is None else temperature,
            "max_tokens": self.settings.glm_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if provider == "openrouter-fp8":
            payload["provider"] = {
                "quantizations": _split_csv(self.settings.openrouter_glm52_quantizations),
                "allow_fallbacks": True,
                "require_parameters": True,
            }
            client = self._openrouter_client
            path = "/chat/completions"
        else:
            client = self._chutes_client
            path = "/v1/chat/completions"
        if client is None:
            raise ValueError(f"{provider} client is not configured")
        response = await client.post(path, json=payload)
        response.raise_for_status()
        raw = _message_content(response.json().get("choices", []))
        return GLMRawResponse(model=model, provider=provider, raw=raw)


def _chute_utilization_snapshot(payload: Any, *, chute_id: str) -> tuple[float | None, int | None]:
    if not isinstance(payload, list):
        return None, None
    for item in payload:
        if not isinstance(item, dict) or item.get("chute_id") != chute_id:
            continue
        return _float_or_none(item.get("utilization_15m")), _int_or_none(
            item.get("instance_count")
            or item.get("active_instance_count")
            or item.get("total_instance_count")
        )
    return None, None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_csv(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["fp8"]


def _openrouter_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _message_content(choices: list[dict[str, Any]]) -> str:
    if not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


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
