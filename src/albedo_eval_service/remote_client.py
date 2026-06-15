from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .models import EvalRequest


class RemoteEvalClient:
    """HTTP client for the remote eval host tunnel API."""

    def __init__(self, *, base_url: str, auth_token: str = "", timeout_seconds: float = 30.0):
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ready(self) -> dict[str, Any]:
        response = await self._client.get("/ready")
        response.raise_for_status()
        return response.json()

    async def start_eval(self, request: EvalRequest) -> dict[str, Any]:
        response = await self._client.post("/eval-runs", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()

    async def get_eval(self, remote_run_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/eval-runs/{remote_run_id}")
        response.raise_for_status()
        return response.json()

    async def iter_events(self, remote_run_id: str) -> AsyncIterator[dict[str, Any]]:
        response = await self._client.get(f"/eval-runs/{remote_run_id}/events")
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events", payload if isinstance(payload, list) else [])
        for event in events:
            yield event
