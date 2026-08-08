"""Unit tests for remote HttpOpenAIGenerator + king_changing handling."""

from __future__ import annotations

import httpx
import pytest

from albedo_eval_service.remote_generation import HttpOpenAIGenerator, KingChangedError


class _Sample:
    def __init__(self, sample_id: str, prompt: str):
        self.sample_id = sample_id
        self.prompt = prompt


def test_http_openai_generator_maps_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"text": "hello", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 2},
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None):
            calls.append({"url": url, "json": json})
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    gen = HttpOpenAIGenerator(
        base_url="http://king:8000",
        model="test-model",
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
    )
    out = gen.generate([_Sample("s1", "prompt-1")])
    assert len(out) == 1
    assert out[0].sample_id == "s1"
    assert out[0].text == "hello"
    assert out[0].error is None
    assert calls[0]["url"] == "http://king:8000/v1/completions"
    assert calls[0]["json"]["model"] == "test-model"


def test_http_openai_generator_raises_king_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 503

        def json(self):
            return {
                "fault_code": "king_changing",
                "king_generation_id": "9",
            }

        @property
        def text(self):
            return "changing"

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    gen = HttpOpenAIGenerator(
        base_url="http://king:8000",
        model="test-model",
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
    )
    with pytest.raises(KingChangedError) as excinfo:
        gen.generate([_Sample("s1", "prompt-1")])
    assert excinfo.value.fault_code == "king_changed"
    assert excinfo.value.king_generation_id == "9"


def test_remote_topology_requires_challenger_four() -> None:
    import uuid

    from albedo_eval_service.models import (
        Challenger,
        DatasetConfig,
        EvalRequest,
        GpuRequest,
        PreviousKing,
        ScoringConfig,
    )
    from albedo_eval_service.remote_config import RemoteSettings
    from albedo_eval_service.remote_worker import RemoteEvalWorker

    settings = RemoteSettings(
        king_base_url="http://albedo-king.albedo-poc.svc:8000",
        previous_king_gpu_ids="",
        challenger_gpu_ids="0,1,2,3",
        scoring_backend="mock",
        upload_artifacts=False,
    )
    worker = RemoteEvalWorker(settings)
    request = EvalRequest(
        eval_run_id=uuid.uuid4(),
        submission_id=uuid.uuid4(),
        challenger=Challenger(model_uri="m", model_hash="h"),
        previous_king=PreviousKing(model_uri="k", model_hash="h", king_version=1),
        dataset=DatasetConfig(
            version="v",
            manifest_uri="s3://x/m",
            manifest_hash="h",
            sample_count=1,
            sample_seed="s",
            sampling_algo="a",
        ),
        scoring=ScoringConfig(judge_config_hash="h", judge_count=1),
        gpu_request=GpuRequest(
            accelerator="H200",
            min_gpus=4,
            preferred_gpus=4,
            previous_king_gpu_count=0,
            challenger_gpu_count=4,
            tensor_parallel_size_per_model=4,
        ),
        artifact_prefix="s3://b/p",
    )
    topo = worker._topology(request)
    assert topo.previous_king == []
    assert topo.challenger == ["0", "1", "2", "3"]


def test_cleanup_stale_vllm_disabled_when_king_remote() -> None:
    from albedo_eval_service.remote_config import RemoteSettings
    from albedo_eval_service.remote_worker import RemoteEvalWorker

    worker = RemoteEvalWorker(
        RemoteSettings(
            king_base_url="http://king:8000",
            cleanup_stale_vllm=True,
            scoring_backend="mock",
            upload_artifacts=False,
        )
    )
    assert worker._should_cleanup_stale_vllm() is False


def test_king_changed_fails_run_without_verdict(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import uuid

    from albedo_eval_service.models import (
        Challenger,
        DatasetConfig,
        EvalRequest,
        GpuRequest,
        PreviousKing,
        ScoringConfig,
    )
    from albedo_eval_service.remote_config import RemoteSettings
    from albedo_eval_service.remote_dataset import EvalSample
    from albedo_eval_service.remote_models import ResolvedModel
    from albedo_eval_service.remote_state import RemoteRun
    from albedo_eval_service.remote_worker import RemoteEvalWorker

    settings = RemoteSettings(
        king_base_url="http://king:8000",
        previous_king_gpu_ids="",
        challenger_gpu_ids="0,1,2,3",
        artifact_spool_dir=str(tmp_path / "artifacts"),
        upload_artifacts=False,
        scoring_backend="mock",
        trajectory_assistant_turns=1,
    )
    request = EvalRequest(
        eval_run_id=uuid.uuid4(),
        submission_id=uuid.uuid4(),
        challenger=Challenger(model_uri="m", model_hash="h"),
        previous_king=PreviousKing(model_uri="k", model_hash="h", king_version=1),
        dataset=DatasetConfig(
            version="v",
            manifest_uri="s3://x/m",
            manifest_hash="h",
            sample_count=1,
            sample_seed="s",
            sampling_algo="a",
            sample_ids=["s1"],
        ),
        scoring=ScoringConfig(judge_config_hash="h", judge_count=1),
        gpu_request=GpuRequest(
            accelerator="H200",
            min_gpus=4,
            preferred_gpus=4,
            previous_king_gpu_count=0,
            challenger_gpu_count=4,
            tensor_parallel_size_per_model=4,
        ),
        artifact_prefix="s3://b/p",
    )
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    worker = RemoteEvalWorker(settings)

    monkeypatch.setattr(
        worker,
        "_load_samples",
        lambda *_args, **_kwargs: [EvalSample(sample_id="s1", prompt="p")],
    )
    monkeypatch.setattr(worker, "_prefetch_repo_context", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_start_category_prep", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_wait_king_ready", lambda: None)

    class _Boom:
        def generate(self, samples):
            raise KingChangedError("king_changing", king_generation_id="2")

        def close(self):
            return None

    monkeypatch.setattr(
        worker,
        "_generator_factory",
        lambda side, gpu_ids, model: _Boom(),
    )
    monkeypatch.setattr(
        worker,
        "_resolve_model_for_side",
        lambda *_a, **_k: ResolvedModel("m", "/tmp/m", "test", True, 0, 0),
    )

    worker.execute(run)
    assert run.state == "failed"
    verdict = run.final_verdict()
    assert verdict is not None
    assert verdict["fault_code"] == "king_changed"
