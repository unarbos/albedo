"""Always-on king control plane in front of a local vLLM OpenAI server.

Runs in the albedo-king pod alongside `vllm serve` (internal port). Exposes:
  GET  /ready|/health  — kube probes + challenger wait
  POST /v1/completions — proxy to vLLM, or 503 king_changing while reloading

State file (JSON) at KING_STATE_PATH controls readiness:
  {"status":"ready","king_generation_id":"1","king_model_uri":"..."}
  {"status":"changing","king_generation_id":"1","king_model_uri":"..."}

Ops mark changing before unloading weights, then ready after the new model loads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

_STATE_PATH = Path(os.environ.get("KING_STATE_PATH", "/tmp/king-state.json"))
_VLLM_URL = os.environ.get("KING_VLLM_URL", "http://127.0.0.1:8001").rstrip("/")
_MODEL_URI = os.environ.get("KING_MODEL_URI", "")
_GENERATION_ID = os.environ.get("KING_GENERATION_ID", "1")
_HOST = os.environ.get("KING_API_HOST", "0.0.0.0")
_PORT = int(os.environ.get("KING_API_PORT", "8000"))


def _default_state() -> dict[str, Any]:
    return {
        "status": "ready",
        "king_generation_id": _GENERATION_ID,
        "king_model_uri": _MODEL_URI,
    }


def _read_state() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        state = _default_state()
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state) + "\n")
        return state
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception:
        return _default_state()


def create_app() -> FastAPI:
    app = FastAPI(title="albedo-king")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        state = _read_state()
        status = str(state.get("status") or "ready").lower()
        payload = {
            "status": status,
            "king_generation_id": str(state.get("king_generation_id") or _GENERATION_ID),
            "king_model_uri": str(state.get("king_model_uri") or _MODEL_URI),
        }
        if status in ("changing", "king_changing"):
            return JSONResponse(
                status_code=503,
                content={
                    **payload,
                    "fault_code": "king_changing",
                },
            )
        return JSONResponse(payload)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy_v1(path: str, request: Request) -> Response:
        state = _read_state()
        status = str(state.get("status") or "ready").lower()
        if status in ("changing", "king_changing"):
            raise HTTPException(
                status_code=503,
                detail={
                    "fault_code": "king_changing",
                    "king_generation_id": str(
                        state.get("king_generation_id") or _GENERATION_ID
                    ),
                    "king_model_uri": str(state.get("king_model_uri") or _MODEL_URI),
                },
            )
        url = f"{_VLLM_URL}/v1/{path}"
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in ("host", "content-length")
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0)) as client:
            upstream = await client.request(
                request.method,
                url,
                params=request.query_params,
                content=body,
                headers=headers,
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return app


def main() -> None:
    uvicorn.run(create_app(), host=_HOST, port=_PORT, log_level="info")


if __name__ == "__main__":
    main()
