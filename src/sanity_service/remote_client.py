"""HTTP client for the sanity GPU worker, reached over the SSH tunnel."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from sanity_remote.models import SanityRunRequest


class SanityRemoteClient:
    # Thin async client mirroring the eval RemoteEvalClient, against the /sanity-runs API.

    def __init__(
        self, *, base_url: str, auth_token: str = "", timeout_seconds: float = 30.0
    ) -> None:
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ready(self) -> dict[str, Any]:
        # Confirms the worker is up before dispatching.
        response = await self._client.get("/ready")
        response.raise_for_status()
        return response.json()

    async def start_run(self, request: SanityRunRequest) -> dict[str, Any]:
        # Submits a generation job (idempotent on run_id).
        response = await self._client.post("/sanity-runs", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        # Status snapshot (or the final result once done).
        response = await self._client.get(f"/sanity-runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def iter_events(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        # Yields the worker's events for this run.
        response = await self._client.get(f"/sanity-runs/{run_id}/events")
        response.raise_for_status()
        for event in response.json().get("events", []):
            yield event
