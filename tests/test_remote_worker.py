from __future__ import annotations

import json
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.canonical_model_config import canonical_max_model_len
from albedo_eval_service.models import Challenger, DatasetConfig, EvalRequest, PreviousKing, ScoringConfig
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_generation import GenerationResult
from albedo_eval_service.remote_state import RemoteRun
from albedo_eval_service.remote_worker import RemoteEvalWorker


class RecordingGenerator:
    def __init__(self, *, side: str, calls: list[dict[str, object]]):
        self.side = side
        self.calls = calls

    def generate(self, samples):
        self.calls.append({"side": self.side, "sample_ids": [sample.sample_id for sample in samples]})
        suffix = " challenger output" if self.side == "challenger" else " king"
        return [GenerationResult(sample_id=sample.sample_id, text=sample.prompt + suffix) for sample in samples]


def _write_dataset(root):
    shard_dir = root / "data"
    shard_dir.mkdir()
    rows = []
    for idx in range(2):
        rows.append(
            json.dumps(
                [
                    {"role": "user", "content": f"Task {idx}"},
                    {"role": "assistant", "content": f"Answer {idx}"},
                ]
            )
        )
    pq.write_table(pa.table({"messages": rows}), shard_dir / "train-00000.parquet")


def _request():
    return EvalRequest(
        eval_run_id=uuid4(),
        submission_id=uuid4(),
        challenger=Challenger(model_uri="s3-or-hippius-uri/challenger", model_hash="sha256:chal"),
        previous_king=PreviousKing(model_uri="s3-or-hippius-uri/king", model_hash="sha256:king", king_version=7),
        dataset=DatasetConfig(
            version="AlienKevin/SWE-ZERO-12M-trajectories",
            manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
            manifest_hash="982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f",
            sample_count=2,
            max_turns_per_sample=1,
            sample_seed="0xabc",
            sampling_algo="swe-zero-manifest-sample-v1",
            generation_batch_size=1,
            scoring_batch_size=1,
            sample_ids=["data/train-00000.parquet:0:0", "data/train-00000.parquet:1:0"],
        ),
        scoring=ScoringConfig(judge_config_hash="sha256:judge"),
        artifact_prefix="s3://albedo-artifacts/submissions/sub/eval/run",
    )


def test_remote_worker_loads_parquet_and_runs_paired_generation(tmp_path):
    _write_dataset(tmp_path)
    calls: list[dict[str, object]] = []

    def factory(side, gpu_ids, model):
        calls.append({"side": side, "gpu_ids": gpu_ids, "model": model})
        return RecordingGenerator(side=side, calls=calls)

    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        generation_backend="vllm",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
    )

    RemoteEvalWorker(settings, generator_factory=factory).execute(run)

    assert run.state == "succeeded"
    verdict = run.final_verdict()
    assert verdict is not None
    assert set(verdict["artifacts"]) == {
        "generated_samples",
        "progress",
        "remote_logs",
        "request",
        "scoring_results",
        "transcript",
        "verdict",
    }
    assert verdict["artifact_metadata"]["generated_samples"]["sha256"].startswith("sha256:")
    assert verdict["valid_turns"] == 2
    assert verdict["gpu_topology"]["previous_king"] == ["0", "1", "2", "3"]
    assert verdict["gpu_topology"]["challenger"] == ["4", "5", "6", "7"]
    generation_events = [event for event in run.events if event["type"] == "generation_batch_done"]
    scoring_events = [event for event in run.events if event["type"] == "scoring_batch_done"]
    assert [event["batch_id"] for event in generation_events] == ["gen-0001", "gen-0002"]
    assert [event["batch_id"] for event in scoring_events] == ["score-0001", "score-0002"]
    assert {call["side"] for call in calls if "gpu_ids" in call} == {"previous_king", "challenger"}


def test_remote_worker_rejects_overlapping_gpu_groups(tmp_path):
    _write_dataset(tmp_path)
    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        previous_king_gpu_ids="0,1,2,3",
        challenger_gpu_ids="3,4,5,6",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
    )

    RemoteEvalWorker(settings, generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[])).execute(run)

    verdict = run.final_verdict()
    assert run.state == "failed"
    assert verdict is not None
    assert verdict["fault_code"] == "remote_worker_failed"
    assert "GPU groups overlap" in verdict["fault_message"]


def test_vllm_generator_uses_canonical_max_model_len_even_when_env_is_lower(tmp_path):
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        upload_artifacts=False,
        max_model_len=4096,
    )

    worker = RemoteEvalWorker(settings, generator_factory=None)
    generator = worker._vllm_generator("challenger", ["4", "5", "6", "7"], "/models/challenger")

    assert generator.max_model_len == canonical_max_model_len()
    assert generator.max_new_tokens == settings.max_new_tokens
