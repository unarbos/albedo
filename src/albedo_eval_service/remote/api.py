from __future__ import annotations

import threading
import time
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, WebSocket
from loguru import logger
from pydantic import BaseModel

from albedo_config import RemoteSettings, get_remote_settings

from ..modelstore.resolver import ModelArtifactResolver
from ..scoring.score_bridge import score_bridge_hub
from ..shared.models import EvalRequest
from .state import RemoteRun, RemoteRunStore
from .worker import RemoteEvalWorker

app = FastAPI(title="Albedo Remote Eval API", version="0.1.0")
store = RemoteRunStore()


def require_auth(
    authorization: Annotated[str | None, Header()] = None,
    settings: RemoteSettings = Depends(get_remote_settings),
) -> None:
    if not settings.auth_token:
        return
    expected = f"Bearer {settings.auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid remote auth token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(
    settings: RemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)
) -> dict[str, object]:
    warnings = []
    if not settings.dataset_root and not settings.mock_auto_verdict:
        warnings.append("ALBEDO_REMOTE_DATASET_ROOT is not set")
    return {
        "ready": settings.ready,
        "host_id": settings.host_id,
        "role": settings.host_role,
        "accelerator_type": settings.accelerator_type,
        "gpu_count": settings.gpu_count,
        "free_gpu_count": settings.free_gpu_count,
        "generation_backend": settings.generation_backend,
        "warnings": warnings,
        "score_bridge_connected": score_bridge_hub.connected,
    }


@app.get("/capacity")
def capacity(
    settings: RemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)
) -> dict[str, object]:
    return {
        "host_id": settings.host_id,
        "role": settings.host_role,
        "gpu_count": settings.gpu_count,
        "free_gpu_count": settings.free_gpu_count,
        "active_runs": len(store.list_active()),
        "accelerator_type": settings.accelerator_type,
        "generation_backend": settings.generation_backend,
        "score_bridge_connected": score_bridge_hub.connected,
    }


@app.websocket("/score-bridge")
async def score_bridge(
    websocket: WebSocket, settings: RemoteSettings = Depends(get_remote_settings)
) -> None:
    if settings.auth_token:
        expected = f"Bearer {settings.auth_token}"
        if websocket.headers.get("authorization") != expected:
            await websocket.close(code=1008)
            return
    await score_bridge_hub.attach(websocket)


@app.post("/eval-runs")
def start_eval_run(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
    settings: RemoteSettings = Depends(get_remote_settings),
    _: None = Depends(require_auth),
) -> dict[str, str]:
    if not settings.ready:
        raise HTTPException(status_code=503, detail="remote eval host is not ready")
    run = store.start(
        request,
        challenger_won=settings.mock_challenger_won,
        auto_verdict=settings.mock_auto_verdict,
    )
    if not settings.mock_auto_verdict:
        queued_run = store.mark_worker_started(run.remote_run_id)
        if queued_run:
            background_tasks.add_task(_execute_remote_run, queued_run, settings)
    return {"remote_run_id": run.remote_run_id, "state": run.state}


class ModelPrefetchRequest(BaseModel):
    model_uri: str


_prefetch_inflight: set[str] = set()
_prefetch_inflight_guard = threading.Lock()
_prefetch_failed_at: dict[str, float] = {}
_PREFETCH_FAILURE_COOLDOWN_S = 900.0


@app.post("/model-prefetch")
def prefetch_model(
    request: ModelPrefetchRequest,
    background_tasks: BackgroundTasks,
    settings: RemoteSettings = Depends(get_remote_settings),
    _: None = Depends(require_auth),
) -> dict[str, str]:
    model_uri = request.model_uri.strip()
    if not model_uri:
        raise HTTPException(status_code=422, detail="model_uri must be non-empty")
    with _prefetch_inflight_guard:
        if model_uri in _prefetch_inflight:
            return {"model_uri": model_uri, "state": "in_progress"}
        failed_at = _prefetch_failed_at.get(model_uri)
        if failed_at is not None and time.monotonic() - failed_at < _PREFETCH_FAILURE_COOLDOWN_S:
            return {"model_uri": model_uri, "state": "failure_cooldown"}
        _prefetch_inflight.add(model_uri)
    background_tasks.add_task(_prefetch_model_artifact, model_uri, settings)
    return {"model_uri": model_uri, "state": "started"}


def _prefetch_model_artifact(model_uri: str, settings: RemoteSettings) -> None:
    try:
        resolved = ModelArtifactResolver(settings).resolve(model_uri)
        with _prefetch_inflight_guard:
            _prefetch_failed_at.pop(model_uri, None)
        print(f"model_prefetch_done ref={model_uri} cache_hit={resolved.cache_hit}", flush=True)
    except Exception as exc:
        with _prefetch_inflight_guard:
            _prefetch_failed_at[model_uri] = time.monotonic()
        logger.warning(f"[remote-api] model prefetch failed ref={model_uri}: {exc}")
    finally:
        with _prefetch_inflight_guard:
            _prefetch_inflight.discard(model_uri)


@app.get("/eval-runs/{remote_run_id}")
def get_eval_run(remote_run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    return run.as_status()


@app.get("/eval-runs/{remote_run_id}/events")
def get_eval_run_events(
    remote_run_id: str, _: None = Depends(require_auth)
) -> dict[str, list[dict[str, object]]]:
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    return {"events": run.events}


@app.post("/eval-runs/{remote_run_id}/cancel")
def cancel_eval_run(remote_run_id: str, _: None = Depends(require_auth)) -> dict[str, str]:
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    run.fail(fault_code="remote_run_cancelled", fault_message="Remote run cancelled by backend")
    return {"remote_run_id": remote_run_id, "state": run.state}


def _execute_remote_run(run: RemoteRun, settings: RemoteSettings) -> None:
    RemoteEvalWorker(settings).execute(run)


def main() -> None:
    import uvicorn

    settings = get_remote_settings()
    uvicorn.run(
        "albedo_eval_service.remote.api:app",
        host=settings.eval_api_host,
        port=settings.eval_api_port,
    )
