from uuid import uuid4

from fastapi.testclient import TestClient

from albedo_eval_service.models import Challenger, DatasetConfig, EvalRequest, PreviousKing, ScoringConfig
from albedo_eval_service.remote_api import app, store
from albedo_eval_service.remote_config import RemoteSettings, get_remote_settings


def _settings() -> RemoteSettings:
    return RemoteSettings(auth_token="secret", mock_auto_verdict=True, mock_challenger_won=False)


def _request() -> dict:
    request = EvalRequest(
        eval_run_id=uuid4(),
        submission_id=uuid4(),
        challenger=Challenger(model_uri="s3://models/challenger", model_hash="sha256:chal"),
        previous_king=PreviousKing(model_uri="s3://models/king", model_hash="sha256:king", king_version=1),
        dataset=DatasetConfig(
            version="AlienKevin/SWE-ZERO-12M-trajectories",
            manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
            manifest_hash="982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f",
            sample_count=4,
            max_turns_per_sample=2,
            sample_seed="0xabc",
            sampling_algo="swe-zero-manifest-sample-v1",
            sample_ids=["data/train-00000.parquet:0:0"],
        ),
        scoring=ScoringConfig(judge_config_hash="sha256:judge"),
        artifact_prefix="s3://albedo-artifacts/submissions/sub/eval/run",
    )
    return request.model_dump(mode="json")


def setup_function():
    app.dependency_overrides[get_remote_settings] = _settings
    store._runs.clear()


def teardown_function():
    app.dependency_overrides.clear()
    store._runs.clear()


def test_remote_api_requires_auth():
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 401


def test_remote_api_starts_idempotent_run_and_replays_verdict():
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    request = _request()

    first = client.post("/eval-runs", json=request, headers=headers)
    second = client.post("/eval-runs", json=request, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["remote_run_id"] == second.json()["remote_run_id"]

    events = client.get(f"/eval-runs/{first.json()['remote_run_id']}/events", headers=headers).json()["events"]
    verdict = events[-1]
    assert verdict["type"] == "verdict"
    assert verdict["state"] == "succeeded"
    assert verdict["artifacts"] == {}


def test_remote_api_capacity_reports_active_runs():
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    response = client.get("/capacity", headers=headers)

    assert response.status_code == 200
    assert response.json()["gpu_count"] == 8
    assert response.json()["active_runs"] == 0
