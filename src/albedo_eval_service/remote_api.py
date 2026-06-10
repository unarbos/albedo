from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

from .models import EvalRequest
from .remote_config import RemoteSettings, get_remote_settings
from .remote_state import RemoteRunStore


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
    return {
        "ready": settings.ready,
        "host_id": settings.host_id,
        "role": settings.host_role,
        "accelerator_type": settings.accelerator_type,
        "gpu_count": settings.gpu_count,
        "free_gpu_count": settings.free_gpu_count,
        "warnings": [],
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
    }


@app.post("/eval-runs")
def start_eval_run(
    request: EvalRequest,
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
    run.state = "failed"
    run.events.append(
        {
            "type": "verdict",
            "eval_run_id": str(run.request.eval_run_id),
            "state": "failed",
            "fault_class": "REMOTE_EVAL_FAULT",
            "fault_code": "remote_run_cancelled",
            "fault_message": "Remote run cancelled by backend",
            "retryable": True,
            "artifacts": {},
        }
    )
    return {"remote_run_id": remote_run_id, "state": run.state}


def main() -> None:
    import uvicorn

    uvicorn.run("albedo_eval_service.remote_api:app", host="0.0.0.0", port=8090)
