from __future__ import annotations

import asyncio
import email.utils
import json
import random
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from uvicorn.importer import import_from_string


class ScoreBridgeClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ALBEDO_SCORE_BRIDGE_", extra="ignore")

    remote_ws_url: str = "ws://127.0.0.1:18090/score-bridge"
    remote_auth_token: str = ""
    judge_base_url: str = "http://127.0.0.1:8091"
    judge_auth_token: str = ""
    request_timeout_seconds: float = 1800.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    ping_interval_seconds: float = 20.0
    ping_timeout_seconds: float = 20.0
    websocket_max_size_bytes: int = 2048 * 1024 * 1024
    retry_count: int = 5
    retry_backoff_seconds: float = 1.5


async def run_bridge(settings: ScoreBridgeClientSettings) -> None:
    headers = {}
    if settings.remote_auth_token:
        headers["Authorization"] = f"Bearer {settings.remote_auth_token}"
    backoff = settings.reconnect_min_seconds
    while True:
        try:
            await _run_once(settings, headers=headers)
            backoff = settings.reconnect_min_seconds
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"score bridge disconnected: {type(exc).__name__}: {exc}", flush=True)
        sleep_for = min(settings.reconnect_max_seconds, backoff) * random.uniform(0.8, 1.2)
        await asyncio.sleep(sleep_for)
        backoff = min(settings.reconnect_max_seconds, backoff * 2)


async def _run_once(settings: ScoreBridgeClientSettings, *, headers: dict[str, str]) -> None:
    wsproto = import_from_string("websockets.asyncio.client:connect")
    judge_headers = {}
    if settings.judge_auth_token:
        judge_headers["Authorization"] = f"Bearer {settings.judge_auth_token}"
    async with httpx.AsyncClient(
        base_url=settings.judge_base_url.rstrip("/"),
        headers=judge_headers,
        timeout=httpx.Timeout(settings.request_timeout_seconds),
    ) as judge_client:
        async with wsproto(
            settings.remote_ws_url,
            additional_headers=headers,
            ping_interval=settings.ping_interval_seconds,
            ping_timeout=settings.ping_timeout_seconds,
            max_size=settings.websocket_max_size_bytes,
        ) as websocket:
            print(f"score bridge connected: {settings.remote_ws_url}", flush=True)
            async for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                except (ValueError, TypeError) as exc:
                    logger.warning(f"[score-bridge-client] dropping malformed frame: {exc}")
                    continue
                if message.get("type") != "score_request":
                    continue
                asyncio.create_task(_handle_score_request(settings, websocket, judge_client, message))


async def _handle_score_request(
    settings: ScoreBridgeClientSettings,
    websocket: Any,
    judge_client: httpx.AsyncClient,
    message: dict[str, Any],
) -> None:
    request_id = str(message.get("request_id") or "")
    payload = message.get("payload")
    endpoint = str(message.get("endpoint") or "/score-batch")
    if not request_id:
        return
    try:
        if not isinstance(payload, dict):
            raise ValueError("score_request payload must be an object")
        if endpoint not in {"/score-batch", "/category-prep", "/simulate-observation"}:
            raise ValueError(f"unsupported score bridge endpoint: {endpoint}")
        body = await _post_json_with_429_retry(
            judge_client,
            endpoint,
            payload,
            retry_count=settings.retry_count,
            base_backoff_seconds=settings.retry_backoff_seconds,
        )
        await websocket.send(json.dumps({"type": "score_response", "request_id": request_id, "body": body}))
    except Exception as exc:
        logger.exception(
            f"[score-bridge-client] score request failed request_id={request_id} endpoint={endpoint}: {exc}"
        )
        await websocket.send(
            json.dumps({"type": "score_response", "request_id": request_id, "error": f"{type(exc).__name__}: {exc}"})
        )


async def _post_json_with_429_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    *,
    retry_count: int,
    base_backoff_seconds: float,
) -> dict[str, Any]:
    for attempt in range(retry_count + 1):
        response = await client.post(endpoint, json=payload)
        if response.status_code != 429 or attempt >= retry_count:
            response.raise_for_status()
            return response.json()
        await asyncio.sleep(_retry_sleep_seconds(response, attempt, base_backoff_seconds))
    raise AssertionError("unreachable")


def _retry_sleep_seconds(
    response: httpx.Response,
    attempt: int,
    base_backoff_seconds: float,
) -> float:
    retry_after = _retry_after_seconds(response.headers.get("retry-after"))
    backoff = base_backoff_seconds * (2**attempt) * random.uniform(0.8, 1.2)
    return max(retry_after, backoff)


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


def main() -> None:
    asyncio.run(run_bridge(ScoreBridgeClientSettings()))


if __name__ == "__main__":
    main()
