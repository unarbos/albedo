from __future__ import annotations

import asyncio
from uuid import uuid4

from albedo_config import JudgeSettings
from albedo_eval_service.judge_api import ObservationSimulationService, SimulateObservationRequest
from albedo_eval_service.judge_llm_client import JudgeRawResponse


class StubClient:
    """Always returns unusable content, recording the provider ladder."""

    def __init__(self):
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return JudgeRawResponse(model=kwargs["model"], provider="stub", raw="")


def _request() -> SimulateObservationRequest:
    return SimulateObservationRequest(
        eval_run_id=str(uuid4()),
        sample_id="mini-coder/data/train-00000.parquet:0:1",
        prompt="fix the bug",
        assistant_output="I will look around.\n```bash\ncat src/main.py\n```",
        messages=[{"role": "user", "content": "fix the bug"}],
    )


def test_simulation_ladder_tries_each_provider_before_expensive_fallback():
    settings = JudgeSettings(
        engy_api_key="",
        simulation_model="deepseek/deepseek-v4-flash-0731",
        simulation_providers="deepseek,cloudflare",
    )
    client = StubClient()
    service = ObservationSimulationService(settings, client, None)

    observation = asyncio.run(service.simulate(_request()))

    models = [c["model"] for c in client.calls]
    orders = [(c["provider"] or {}).get("order") for c in client.calls]
    assert models == [
        "deepseek/deepseek-v4-flash-0731",  # rung 1: deepseek provider first
        "deepseek/deepseek-v4-flash-0731",  # rung 2: cloudflare provider first
        settings.evaluator_model,  # rung 3: expensive judge model last
    ]
    assert orders[0][0] == "deepseek"
    assert orders[1][0] == "cloudflare"
    assert orders[2][0] == settings.evaluator_providers.split(",")[0]
    # everything failed -> deterministic empty observation, never an exception
    assert "returncode" in observation


def test_simulation_single_model_uses_evaluator_provider():
    settings = JudgeSettings(engy_api_key="", simulation_model="")
    client = StubClient()
    service = ObservationSimulationService(settings, client, None)

    asyncio.run(service.simulate(_request()))

    assert len(client.calls) == 1
    assert client.calls[0]["model"] == settings.evaluator_model
    assert (client.calls[0]["provider"] or {}).get("order") == [
        p.strip() for p in settings.evaluator_providers.split(",")
    ]
