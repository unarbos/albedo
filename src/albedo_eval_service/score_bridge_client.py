from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx
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
                message = json.loads(raw_message)
                if message.get("type") != "score_request":
                    continue
                asyncio.create_task(_handle_score_request(websocket, judge_client, message))


async def _handle_score_request(websocket: Any, judge_client: httpx.AsyncClient, message: dict[str, Any]) -> None:
    request_id = str(message.get("request_id") or "")
    payload = message.get("payload")
    endpoint = str(message.get("endpoint") or "/score-batch")
    if not request_id:
        return
    try:
        if not isinstance(payload, dict):
            raise ValueError("score_request payload must be an object")
        if endpoint not in {"/score-batch", "/category-prep"}:
            raise ValueError(f"unsupported score bridge endpoint: {endpoint}")
        response = await judge_client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        await websocket.send(json.dumps({"type": "score_response", "request_id": request_id, "body": body}))
    except Exception as exc:
        await websocket.send(
            json.dumps({"type": "score_response", "request_id": request_id, "error": f"{type(exc).__name__}: {exc}"})
        )


def main() -> None:
    asyncio.run(run_bridge(ScoreBridgeClientSettings()))


if __name__ == "__main__":
    main()
