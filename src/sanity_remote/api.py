"""Stateless sanity worker API - the dispatcher POSTs a job and polls events; no DB, no judges."""

from __future__ import annotations

from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from loguru import logger

from sanity_remote.config import SanityRemoteSettings, get_remote_settings
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_remote.worker import generate

app = FastAPI(title="Albedo Sanity Remote Worker", version="0.1.0")
store = SanityRunStore()


def require_auth(authorization: Annotated[str | None, Header()] = None, settings: SanityRemoteSettings = Depends(get_remote_settings),) -> None:
    # Bearer-token gate; open when no token is configured (local/dev).
    if not settings.auth_token:
        return
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status_code=401, detail="invalid remote auth token")


@app.get("/health")
def health() -> dict[str, str]:
    # Liveness probe (unauthenticated).
    return {"status": "ok"}


@app.get("/ready")
def ready(settings: SanityRemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)) -> dict[str, object]:
    # Reports readiness + host identity for the dispatcher's host selection.
    return {
        "ready": settings.ready,
        "host_id": settings.host_id,
        "role": settings.host_role,
        "active_runs": len(store.list_active()),
    }


@app.get("/capacity")
def capacity(settings: SanityRemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)) -> dict[str, object]:
    # Current load so the dispatcher avoids piling work on a busy host.
    return {
        "host_id": settings.host_id,
        "role": settings.host_role,
        "active_runs": len(store.list_active()),
    }


@app.post("/sanity-runs")
async def start_run(request: SanityRunRequest, background_tasks: BackgroundTasks, settings: SanityRemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth),) -> dict[str, str]:
    # Accepts a generation job (idempotent on run_id) and runs it in the background.
    if not settings.ready:
        raise HTTPException(status_code=503, detail="sanity worker is not ready")
    run = store.start(request)
    queued = store.mark_worker_started(run.run_id)
    if queued is not None:
        logger.info("[sanity-remote-api] queuing run={} digest={:.16}", run.run_id, request.digest)
        background_tasks.add_task(generate, queued)
    else:
        logger.info("[sanity-remote-api] duplicate run_id={} state={}", run.run_id, run.state)
    return {"run_id": run.run_id, "state": run.state}


@app.get("/sanity-runs/{run_id}")
def get_run(run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    # Status snapshot (or the final result once done).
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.as_status()


@app.get("/sanity-runs/{run_id}/events")
def get_run_events(run_id: str, _: None = Depends(require_auth)) -> dict[str, list[dict[str, object]]]:
    # Full event list for the dispatcher to poll until a result appears.
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"events": run.events}


@app.post("/sanity-runs/{run_id}/cancel")
def cancel_run(run_id: str, _: None = Depends(require_auth)) -> dict[str, str]:
    # Marks a run failed/retryable on dispatcher request.
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    run.fail(fault_code="run_cancelled", fault_message="cancelled by dispatcher", retryable=True)
    logger.info("[sanity-remote-api] cancelled run={}", run_id)
    return {"run_id": run_id, "state": run.state}


def main() -> None:
    # Console entrypoint: serve the worker API on the configured port.
    import os

    import uvicorn

    settings = get_remote_settings()
    # hippius_validation.config reads ALBEDO_MODEL_CACHE_DIR at first import (inside _materialize).
    # Propagate our setting now so the lazy import picks up the right cache root.
    os.environ.setdefault("ALBEDO_MODEL_CACHE_DIR", settings.model_cache_dir)
    uvicorn.run(
        "sanity_remote.api:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info",
    )
