from __future__ import annotations

import asyncio

import pytest
from uuid import uuid4

from albedo_eval_service.models import Challenger, DatasetConfig, EvalRequest, PreviousKing, ScoringConfig
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_dataset import EvalSample
from albedo_eval_service.remote_scoring import build_scorer
from albedo_eval_service.score_bridge import ScoreBridgeHub, ScoreBridgeUnavailable
from albedo_eval_service.score_bridge_client import ScoreBridgeClientSettings, run_bridge


def test_build_scorer_supports_websocket_backend():
    scorer = build_scorer(RemoteSettings(scoring_backend="websocket", scoring_timeout_seconds=1))

    assert scorer.__class__.__name__ == "WebSocketScoringClient"


def test_score_bridge_hub_reports_unavailable_without_client():
    hub = ScoreBridgeHub()

    with pytest.raises(ScoreBridgeUnavailable):
        hub.request({"hello": "world"}, timeout_seconds=0.01)


def test_score_bridge_client_reconnects_after_disconnect(monkeypatch):
    attempts = []

    async def fake_run_once(settings, *, headers):
        attempts.append(headers)
        raise RuntimeError("socket dropped")

    async def fake_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr("albedo_eval_service.score_bridge_client._run_once", fake_run_once)
    monkeypatch.setattr("albedo_eval_service.score_bridge_client.asyncio.sleep", fake_sleep)

    settings = ScoreBridgeClientSettings(remote_auth_token="remote-token", reconnect_min_seconds=0.01)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_bridge(settings))

    assert attempts == [{"Authorization": "Bearer remote-token"}]

def _eval_request() -> EvalRequest:
    return EvalRequest(
        eval_run_id=uuid4(),
        submission_id=uuid4(),
        challenger=Challenger(model_uri="challenger", model_hash="sha256:chal"),
        previous_king=PreviousKing(model_uri="king", model_hash="sha256:king", king_version=1),
        dataset=DatasetConfig(
            version="AlienKevin/SWE-ZERO-12M-trajectories",
            manifest_uri="hf://dataset",
            manifest_hash="sha256:manifest",
            sample_count=1,
            sample_seed="seed",
            sampling_algo="explicit",
        ),
        scoring=ScoringConfig(judge_config_hash="sha256:judge"),
        artifact_prefix="local://run",
    )


def test_websocket_scorer_starts_category_prep_over_bridge(monkeypatch):
    calls = []

    def fake_request(payload, *, timeout_seconds, endpoint="/score-batch"):
        calls.append({"payload": payload, "timeout_seconds": timeout_seconds, "endpoint": endpoint})
        return {"category_prep_id": "prep-123"}

    monkeypatch.setattr("albedo_eval_service.remote_scoring.score_bridge_hub.request", fake_request)
    scorer = build_scorer(RemoteSettings(scoring_backend="websocket", scoring_timeout_seconds=7))

    prep_id = scorer.start_category_prep(
        request=_eval_request(),
        samples=[EvalSample(sample_id="data/train-00000.parquet:0:0", prompt="Prompt")],
    )

    assert prep_id == "prep-123"
    assert calls[0]["endpoint"] == "/category-prep"
    assert calls[0]["timeout_seconds"] == 7
    assert calls[0]["payload"]["samples"] == [
        {"sample_id": "data/train-00000.parquet:0:0", "prompt": "Prompt", "sample_index": 0}
    ]

