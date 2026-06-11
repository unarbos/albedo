from __future__ import annotations

import asyncio

import pytest

from albedo_eval_service.remote_config import RemoteSettings
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
