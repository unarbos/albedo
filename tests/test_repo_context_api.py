from __future__ import annotations

from fastapi.testclient import TestClient

from albedo_config import RepoContextSettings
from repo_context_service.api import create_app
from repo_context_service.core import GroundingContext


class FakeService:
    def __init__(self, result: GroundingContext | Exception):
        self.result = result
        self.prefetched: list[str] | None = None

    def context_for(self, sample_id, assistant_output):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def prefetch(self, sample_ids):
        self.prefetched = list(sample_ids)
        return {"samples": len(sample_ids), "instances": 0, "ready": 0}

    def _auth_headers(self):
        return {}

    def close(self):
        pass


def make_client(result, service: FakeService | None = None) -> TestClient:
    settings = RepoContextSettings(_env_file=None, cache_dir="/tmp/unused")
    return TestClient(create_app(settings, service=service or FakeService(result)))


def test_repo_context_happy_path():
    client = make_client(GroundingContext(context="BLOCK", kind="repo"))
    response = client.post("/repo-context", json={"sample_id": "swe-zero/data/train-0.parquet:0:0"})
    assert response.status_code == 200
    assert response.json() == {
        "sample_id": "swe-zero/data/train-0.parquet:0:0",
        "context": "BLOCK",
        "kind": "repo",
    }


def test_repo_context_returns_none_kind_on_failure():
    client = make_client(RuntimeError("boom"))
    response = client.post("/repo-context", json={"sample_id": "x", "assistant_output": "y"})
    assert response.status_code == 200
    assert response.json() == {"sample_id": "x", "context": None, "kind": "none"}


def test_prefetch_endpoint_accepts_and_runs_in_background():
    service = FakeService(GroundingContext(context=None, kind="none"))
    client = make_client(None, service=service)
    response = client.post("/prefetch", json={"sample_ids": ["a:0:0", "b:1:0"]})
    assert response.status_code == 200
    assert response.json() == {"accepted": 2}
    assert service.prefetched == ["a:0:0", "b:1:0"]


def test_healthz_reports_configuration():
    client = make_client(GroundingContext(context=None, kind="none"))
    payload = client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["cache_dir"] == "/tmp/unused"
    assert payload["manifest_configured"] is False
    assert payload["github_token_set"] is False
