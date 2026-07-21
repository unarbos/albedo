from __future__ import annotations

import threading
import time
import types
from uuid import uuid4

import httpx
import pytest

import albedo_eval_service.remote_scoring as remote_scoring_module
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_dataset import EvalSample
from albedo_eval_service.remote_generation import GenerationResult
from albedo_eval_service.remote_scoring import (
    WebSocketScoringClient,
    _category_prep_payload,
    _collect_score_batches,
    _post_json_with_429_retry,
    _score_batch_payloads,
    _simulate_observation_payload,
)


def _samples(counts: dict[str, int]) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for source, n in counts.items():
        for row in range(n):
            sid = f"{source}/data/train-00000.parquet:{row}:1"
            samples.append(EvalSample(sample_id=sid, prompt=f"task {sid}"))
    return samples


def test_category_prep_payload_carries_only_id_and_prompt():
    # Flat prep is task-only and order-free: no sample_index counterbalancing.
    samples = _samples({"swe-zero": 4, "mini-coder": 2})
    request = types.SimpleNamespace(eval_run_id=uuid4())
    payload = _category_prep_payload(request, samples)
    assert payload["total_sample_count"] == len(samples)
    for entry in payload["samples"]:
        assert set(entry) == {"sample_id", "prompt"}


def test_score_batch_payload_carries_both_outputs_no_index():
    samples = _samples({"swe-zero": 4, "mini-coder": 2})
    king = [GenerationResult(sample_id=s.sample_id, text="king") for s in samples]
    challenger = [GenerationResult(sample_id=s.sample_id, text="challenger out") for s in samples]
    request = types.SimpleNamespace(
        eval_run_id=uuid4(),
        scoring=types.SimpleNamespace(judge_count=3),
        dataset=types.SimpleNamespace(scoring_batch_size=100),
    )
    payloads = _score_batch_payloads(request, samples, king, challenger, category_prep_id="prep-1")
    assert payloads[0]["total_sample_count"] == len(samples)
    assert payloads[0]["category_prep_id"] == "prep-1"
    for entry in payloads[0]["samples"]:
        assert set(entry) == {"sample_id", "prompt", "previous_king_output", "challenger_output"}


def test_simulate_observation_payload_carries_messages():
    sample = EvalSample(
        sample_id="data/train-00000.parquet:0:0",
        prompt="formatted prompt",
        messages=[{"role": "user", "content": "Fix it"}],
    )
    request = types.SimpleNamespace(eval_run_id=uuid4())

    payload = _simulate_observation_payload(request, sample, "THOUGHT...\n```bash\nls\n```")

    assert payload == {
        "eval_run_id": str(request.eval_run_id),
        "sample_id": sample.sample_id,
        "prompt": "formatted prompt",
        "messages": [{"role": "user", "content": "Fix it"}],
        "assistant_output": "THOUGHT...\n```bash\nls\n```",
    }


def test_collect_score_batches_preserves_payload_order():
    payloads = [{"batch_id": f"score-{i:04d}"} for i in range(1, 7)]
    delays = {payload["batch_id"]: (6 - i) * 0.01 for i, payload in enumerate(payloads)}

    def send(payload):
        time.sleep(delays[payload["batch_id"]])  # earlier batches finish later
        return {
            "scoring_records": [{"sample_id": f"{payload['batch_id']}-r"}],
            "summary": {"batch_id": payload["batch_id"]},
        }

    records, summaries = _collect_score_batches(payloads, send, max_concurrency=4)

    assert [r["sample_id"] for r in records] == [f"score-{i:04d}-r" for i in range(1, 7)]
    assert [s["batch_id"] for s in summaries] == [f"score-{i:04d}" for i in range(1, 7)]


def test_collect_score_batches_propagates_batch_failure():
    payloads = [{"batch_id": "score-0001"}, {"batch_id": "score-0002"}]

    def send(payload):
        if payload["batch_id"] == "score-0002":
            raise RuntimeError("boom")
        return {"scoring_records": [], "summary": {}}

    with pytest.raises(RuntimeError, match="boom"):
        _collect_score_batches(payloads, send, max_concurrency=2)


def test_remote_scoring_defaults_to_128_concurrent_requests():
    assert RemoteSettings(_env_file=None).scoring_batch_concurrency == 128


def test_post_json_retries_429(monkeypatch):
    sleeps = []
    calls = 0

    def fake_sleep(seconds):
        sleeps.append(seconds)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.25"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    monkeypatch.setattr(remote_scoring_module.time, "sleep", fake_sleep)
    with httpx.Client(base_url="http://judge", transport=httpx.MockTransport(handler)) as client:
        body = _post_json_with_429_retry(
            client,
            "/score-batch",
            {"batch_id": "score-0001"},
            retry_count=2,
            base_backoff_seconds=0.01,
        )

    assert body == {"ok": True}
    assert calls == 2
    assert sleeps == [0.25]


def test_websocket_scoring_client_sends_batches_concurrently(monkeypatch):
    samples = _samples({"swe-zero": 6})
    king = [GenerationResult(sample_id=s.sample_id, text="king") for s in samples]
    challenger = [GenerationResult(sample_id=s.sample_id, text="challenger") for s in samples]
    request = types.SimpleNamespace(
        eval_run_id=uuid4(),
        scoring=types.SimpleNamespace(judge_count=3),
        dataset=types.SimpleNamespace(scoring_batch_size=2),
    )
    # 3 batches must be in flight at once or the barrier breaks and the test fails.
    barrier = threading.Barrier(3, timeout=5)

    class StubHub:
        def request(self, payload, *, timeout_seconds, endpoint="/score-batch"):
            barrier.wait()
            return {
                "scoring_records": [
                    {
                        "sample_id": entry["sample_id"],
                        "scored": True,
                        "king_score": 0.5,
                        "challenger_score": 0.5,
                        "judge_results": [],
                    }
                    for entry in payload["samples"]
                ],
                "summary": {"batch_id": payload["batch_id"]},
            }

    monkeypatch.setattr(remote_scoring_module, "score_bridge_hub", StubHub())
    client = WebSocketScoringClient(
        RemoteSettings(scoring_backend="websocket", scoring_batch_concurrency=3)
    )

    result = client.score(
        request=request, samples=samples, king_results=king, challenger_results=challenger
    )

    assert [r["sample_id"] for r in result.records] == [s.sample_id for s in samples]
    assert [s["batch_id"] for s in result.summary["batch_summaries"]] == [
        "score-0001",
        "score-0002",
        "score-0003",
    ]


def test_score_batch_payload_skips_errored_generations():
    samples = _samples({"swe-zero": 3})
    king = [GenerationResult(sample_id=s.sample_id, text="king") for s in samples]
    challenger = [
        GenerationResult(sample_id=samples[0].sample_id, text="ok"),
        GenerationResult(sample_id=samples[1].sample_id, text="", error="vllm_timeout"),
        GenerationResult(sample_id=samples[2].sample_id, text="ok"),
    ]
    request = types.SimpleNamespace(
        eval_run_id=uuid4(),
        scoring=types.SimpleNamespace(judge_count=3),
        dataset=types.SimpleNamespace(scoring_batch_size=100),
    )
    payloads = _score_batch_payloads(request, samples, king, challenger)
    emitted = {entry["sample_id"] for payload in payloads for entry in payload["samples"]}
    assert samples[1].sample_id not in emitted  # errored pair dropped
    assert len(emitted) == 2
