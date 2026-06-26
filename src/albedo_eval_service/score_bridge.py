from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect


class ScoreBridgeUnavailable(RuntimeError):
    pass


class ScoreBridgeHub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: WebSocket | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._websocket is not None and self._loop is not None

    async def attach(self, websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        old: WebSocket | None = None
        with self._lock:
            old = self._websocket
            self._loop = loop
            self._websocket = websocket
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ScoreBridgeUnavailable("score bridge client replaced"))
            self._pending.clear()
        if old is not None and old is not websocket:
            try:
                await old.close(code=1012)
            except Exception:
                pass
        try:
            while True:
                message = await websocket.receive_json()
                await self._handle_message(message)
        except WebSocketDisconnect:
            pass
        finally:
            with self._lock:
                if self._websocket is websocket:
                    self._websocket = None
                    self._loop = None
                    for future in self._pending.values():
                        if not future.done():
                            future.set_exception(ScoreBridgeUnavailable("score bridge disconnected"))
                    self._pending.clear()

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        endpoint: str = "/score-batch",
    ) -> dict[str, Any]:
        # Wait up to 120s for the bridge to (re)connect - handles transient disconnects during vLLM cleanup.
        _deadline = time.monotonic() + 120.0
        while True:
            with self._lock:
                loop = self._loop
                websocket = self._websocket
            if loop is not None and websocket is not None:
                break
            remaining = _deadline - time.monotonic()
            if remaining <= 0:
                raise ScoreBridgeUnavailable("no score bridge client connected")
            time.sleep(min(2.0, remaining))
        future = asyncio.run_coroutine_threadsafe(
            self._request_on_loop(
                websocket,
                payload,
                timeout_seconds=timeout_seconds,
                endpoint=endpoint,
            ),
            loop,
        )
        return future.result(timeout=timeout_seconds + 5.0)

    async def _request_on_loop(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        endpoint: str,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        with self._lock:
            if self._websocket is not websocket:
                raise ScoreBridgeUnavailable("score bridge client changed")
            self._pending[request_id] = response_future
        try:
            await websocket.send_json(
                {
                    "type": "score_request",
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "payload": payload,
                }
            )
            return await asyncio.wait_for(response_future, timeout=timeout_seconds)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "score_response":
            return
        request_id = str(message.get("request_id") or "")
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or future.done():
            return
        error = message.get("error")
        if error:
            future.set_exception(RuntimeError(str(error)))
            return
        body = message.get("body")
        if not isinstance(body, dict):
            future.set_exception(RuntimeError("score bridge response body must be an object"))
            return
        future.set_result(body)


score_bridge_hub = ScoreBridgeHub()
