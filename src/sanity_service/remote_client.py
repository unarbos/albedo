"""HTTP client for the sanity GPU worker, reached over the SSH tunnel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from sanity_remote.models import SanityRunRequest

_RETRY_COUNT = 3
_RETRY_BACKOFF_S = 1.0


class SanityRemoteClient:
    # Thin async client mirroring the eval RemoteEvalClient, against the /sanity-runs API.

    def __init__(self, *, base_url: str, auth_token: str = "", timeout_seconds: float = 30.0) -> None:
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _fetch(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # Retries transient 5xx and connection errors with exponential backoff.
        last_exc: Exception | None = None
        for attempt in range(_RETRY_COUNT):
            try:
                r = await self._client.request(method, path, **kwargs)
                r.raise_for_status()
                return r
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt >= _RETRY_COUNT - 1:
                    raise
                last_exc = exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt >= _RETRY_COUNT - 1:
                    raise
                last_exc = exc
            delay = _RETRY_BACKOFF_S * (2**attempt)
            logger.warning(
                "[sanity-client] transient error attempt={}/{} path={} retrying in {}s: {}",
                attempt + 1,
                _RETRY_COUNT,
                path,
                delay,
                last_exc,
            )
            await asyncio.sleep(delay)
        raise RuntimeError("unreachable")  # noqa: EM101 - loop always raises before here

    async def ready(self) -> dict[str, Any]:
        # Confirms the worker is up before dispatching.
        r = await self._fetch("GET", "/ready")
        data = r.json()
        logger.info("[sanity-client] worker ready host_id={} role={} active_runs={}", data.get("host_id"), data.get("role"), data.get("active_runs"),)
        return data

    async def start_run(self, request: SanityRunRequest) -> dict[str, Any]:
        # Submits a generation job (idempotent on run_id).
        logger.info("[sanity-client] submitting run={} digest={:.16}", request.run_id, request.digest)
        r = await self._fetch("POST", "/sanity-runs", json=request.model_dump(mode="json"))
        data = r.json()
        logger.info("[sanity-client] run submitted state={}", data.get("state"))
        return data

    async def get_run(self, run_id: str) -> dict[str, Any]:
        # Status snapshot (or the final result once done).
        r = await self._fetch("GET", f"/sanity-runs/{run_id}")
        return r.json()

    async def iter_events(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        # Yields the worker's events for this run.
        r = await self._fetch("GET", f"/sanity-runs/{run_id}/events")
        for event in r.json().get("events", []):
            yield event
