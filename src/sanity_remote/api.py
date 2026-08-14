from __future__ import annotations

from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from loguru import logger

from albedo_config import SanityRemoteSettings, get_sanity_remote_settings
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_remote.worker import generate, teardown

app = FastAPI(title="Albedo Sanity Remote Worker", version="0.1.0")
store = SanityRunStore()


def require_auth(
    authorization: Annotated[str | None, Header()] = None,
    settings: SanityRemoteSettings = Depends(get_sanity_remote_settings),
) -> None:
    if not settings.auth_token:
        return
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status_code=401, detail="invalid remote auth token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(
    settings: SanityRemoteSettings = Depends(get_sanity_remote_settings),
    _: None = Depends(require_auth),
) -> dict[str, object]:
    return {
        "ready": settings.ready,
        "host_id": settings.host_id,
        "role": settings.host_role,
        "active_runs": len(store.list_active()),
    }


@app.get("/capacity")
def capacity(
    settings: SanityRemoteSettings = Depends(get_sanity_remote_settings),
    _: None = Depends(require_auth),
) -> dict[str, object]:
    return {
        "host_id": settings.host_id,
        "role": settings.host_role,
        "active_runs": len(store.list_active()),
    }


@app.post("/sanity-runs")
async def start_run(
    request: SanityRunRequest,
    background_tasks: BackgroundTasks,
    settings: SanityRemoteSettings = Depends(get_sanity_remote_settings),
    _: None = Depends(require_auth),
) -> dict[str, str]:
    if not settings.ready:
        raise HTTPException(status_code=503, detail="sanity worker is not ready")
    active = store.list_active()
    incoming = getattr(request, "run_id", None)
    if active and (incoming is None or all(r.run_id != incoming for r in active)):
        raise HTTPException(
            status_code=409, detail=f"sanity worker busy: {len(active)} active run(s)"
        )
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
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.as_status()


@app.get("/sanity-runs/{run_id}/events")
def get_run_events(
    run_id: str, _: None = Depends(require_auth)
) -> dict[str, list[dict[str, object]]]:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"events": run.events}


@app.post("/sanity-runs/{run_id}/cancel")
def cancel_run(run_id: str, _: None = Depends(require_auth)) -> dict[str, str]:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    run.fail(fault_code="run_cancelled", fault_message="cancelled by dispatcher", retryable=True)
    logger.info("[sanity-remote-api] cancelled run={}", run_id)
    return {"run_id": run_id, "state": run.state}


@app.post("/teardown")
async def teardown_worker(_: None = Depends(require_auth)) -> dict[str, str]:
    await teardown()
    return {"state": "ok"}


def main() -> None:
    import uvicorn

    settings = get_sanity_remote_settings()
    uvicorn.run(
        "sanity_remote.api:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
