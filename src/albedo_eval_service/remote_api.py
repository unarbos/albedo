from __future__ import annotations

from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException

from .models import EvalRequest
from .remote_config import RemoteSettings, get_remote_settings
from .remote_state import RemoteRun, RemoteRunStore
from .remote_worker import RemoteEvalWorker


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
def ready(settings: RemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)) -> dict[str, object]:
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
    }


@app.get("/capacity")
def capacity(settings: RemoteSettings = Depends(get_remote_settings), _: None = Depends(require_auth)) -> dict[str, object]:
    return {
        "host_id": settings.host_id,
        "role": settings.host_role,
        "gpu_count": settings.gpu_count,
        "free_gpu_count": settings.free_gpu_count,
        "active_runs": len(store.list_active()),
        "accelerator_type": settings.accelerator_type,
        "generation_backend": settings.generation_backend,
    }


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


@app.get("/eval-runs/{remote_run_id}")
def get_eval_run(remote_run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    return run.as_status()


@app.get("/eval-runs/{remote_run_id}/events")
def get_eval_run_events(remote_run_id: str, _: None = Depends(require_auth)) -> dict[str, list[dict[str, object]]]:
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

    uvicorn.run("albedo_eval_service.remote_api:app", host="0.0.0.0", port=8090)
