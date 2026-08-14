from __future__ import annotations

from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException

from albedo_config import Settings, get_eval_settings

from ..shared.models import SubmissionStatus
from .repository import EvalRepository


def get_repository(settings: Settings = Depends(get_eval_settings)) -> EvalRepository:
    return EvalRepository(settings.database_url)


app = FastAPI(title="Albedo Eval Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(settings: Settings = Depends(get_eval_settings)) -> dict[str, object]:
    missing = []
    for field_name in (
        "database_url",
        "dataset_version",
        "dataset_manifest_uri",
        "judge_config_hash",
    ):
        if not getattr(settings, field_name):
            missing.append(field_name)
    return {"ready": not missing, "missing": missing}


@app.get("/submissions/{submission_id}", response_model=SubmissionStatus)
def submission_status(
    submission_id: UUID,
    repository: EvalRepository = Depends(get_repository),
) -> SubmissionStatus:
    status = repository.get_submission(submission_id)
    if status is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return status


def main() -> None:
    import uvicorn

    settings = get_eval_settings()
    uvicorn.run(
        "albedo_eval_service.control.api:app", host=settings.api_host, port=settings.api_port
    )
