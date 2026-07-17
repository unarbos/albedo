from __future__ import annotations

import json
import queue
import sys
import types
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.canonical_model_config import canonical_max_model_len
from albedo_eval_service.models import (
    Challenger,
    DatasetConfig,
    EvalRequest,
    PreviousKing,
    ScoringConfig,
)
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_generation import (
    GenerationResult,
    VllmProcessGenerator,
    _vllm_worker,
    format_scored_trajectory,
)
from albedo_eval_service.remote_models import ResolvedModel
from albedo_eval_service.remote_scoring import ScoringResult
from albedo_eval_service.remote_state import RemoteRun
from albedo_eval_service.remote_worker import RemoteEvalWorker


class _Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(message["content"] for message in messages) + " assistant:"


class RecordingGenerator:
    def __init__(self, *, side: str, calls: list[dict[str, object]]):
        self.side = side
        self.calls = calls

    def generate(self, samples):
        self.calls.append(
            {"side": self.side, "sample_ids": [sample.sample_id for sample in samples]}
        )
        suffix = " challenger output" if self.side == "challenger" else " king"
        return [
            GenerationResult(sample_id=sample.sample_id, text=sample.prompt + suffix)
            for sample in samples
        ]

    def close(self):
        self.calls.append({"side": self.side, "closed": True})


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
        previous_king=PreviousKing(
            model_uri="s3-or-hippius-uri/king", model_hash="sha256:king", king_version=7
        ),
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


def test_scored_trajectory_marks_only_candidate_outputs():
    text = format_scored_trajectory(
        [
            {"role": "user", "content": "Fix it"},
            {"role": "assistant", "content": "first", "score_target": True},
            {"role": "user", "content": "Observation: ok", "environment_observation": True},
            {"role": "assistant", "content": "second", "score_target": True},
            {"role": "user", "content": "Observation: still ok", "environment_observation": True},
            {"role": "assistant", "content": "third", "score_target": True},
        ]
    )

    assert "Score ONLY CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT 3" in text
    assert "CONTEXT USER (do not score)" in text
    assert "CANDIDATE OUTPUT 1" in text
    assert "ENVIRONMENT OBSERVATION (context only, do not score)" in text
    assert "CANDIDATE OUTPUT 2" in text
    assert "CANDIDATE OUTPUT 3" in text


class _AliveProcess:
    exitcode = None

    def is_alive(self):
        return True


class _EmptyQueue:
    def get(self, *, timeout):
        raise queue.Empty

    def get_nowait(self):
        raise queue.Empty


def test_vllm_generator_times_out_when_worker_sends_no_payload():
    sample = types.SimpleNamespace(sample_id="sample-1", prompt="Fix it")
    generator = VllmProcessGenerator(
        model="m",
        gpu_ids=["0"],
        max_new_tokens=1,
        temperature=0,
        top_p=1,
        result_timeout_seconds=0.01,
    )
    generator._process = _AliveProcess()
    generator._result_queue = _EmptyQueue()

    payload = generator._wait_for_payload("1", [sample])

    assert payload["error"] == "vLLM process produced no result payload after 0.01s"


def test_remote_worker_loads_parquet_and_runs_paired_generation(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote_dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
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
        scoring_backend="mock",
    )

    RemoteEvalWorker(settings, generator_factory=factory).execute(run)

    assert run.state == "succeeded"
    verdict = run.final_verdict()
    assert verdict is not None
    assert set(verdict["artifacts"]) == {
        "generated_samples",
        "judge_results",
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
    generate_calls = [call for call in calls if "sample_ids" in call]
    assert [call["side"] for call in generate_calls].count("previous_king") == 2
    assert [call["side"] for call in generate_calls].count("challenger") == 2
    assert [call["side"] for call in calls if call.get("closed")].count("previous_king") == 1
    assert [call["side"] for call in calls if call.get("closed")].count("challenger") == 1


class RecordingModelResolver:
    def __init__(self, calls: list[object]):
        self.calls = calls

    def resolve(self, model_ref: str) -> ResolvedModel:
        self.calls.append(f"resolve:{model_ref}")
        return ResolvedModel(model_ref, model_ref, "test", True, 0, 0)


class RecordingScorer:
    def __init__(self, calls: list[object]):
        self.calls = calls

    def start_category_prep(self, *, request, samples):
        self.calls.append("category_prep")
        return "prep-1"

    def simulate_observation(self, *, request, sample, assistant_output):
        self.calls.append(f"simulate:{sample.sample_id}")
        return f"Observation: saw {assistant_output[-20:]}"

    def score(self, *, request, samples, king_results, challenger_results, category_prep_id=None):
        self.calls.append(f"score:{category_prep_id}")
        records = [
            {
                "sample_id": sample.sample_id,
                "order": ["previous_king", "challenger"],
                "judge_results": [],
                "judge_scores": [],
                "sample_score": 0.5,
                "scored": True,
                "scoring_mode": "test",
            }
            for sample in samples
        ]
        return ScoringResult(
            records=records,
            summary={
                "state": "succeeded",
                "score_challenger": 0.5,
                "score_king": 0.5,
                "challenger_won": False,
                "valid_turns": len(records),
                "total_turns": len(records),
                "judge_errors": 0,
                "scored_sample_count": len(records),
                "scoring_mode": "test",
            },
        )


def test_remote_worker_starts_category_prep_before_model_resolution(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote_dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
    calls: list[object] = []

    def factory(side, gpu_ids, model):
        calls.append({"side": side, "model": model})
        return RecordingGenerator(side=side, calls=calls)

    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        generation_backend="vllm",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
    )

    RemoteEvalWorker(
        settings,
        generator_factory=factory,
        model_resolver=RecordingModelResolver(calls),
        scorer=RecordingScorer(calls),
    ).execute(run)

    assert calls.index("category_prep") < calls.index("resolve:s3-or-hippius-uri/king")
    assert any(str(call).startswith("simulate:") for call in calls)


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
        scoring_backend="mock",
    )

    RemoteEvalWorker(
        settings,
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
    ).execute(run)

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
        scoring_backend="mock",
    )

    worker = RemoteEvalWorker(settings, generator_factory=None)
    generator = worker._vllm_generator("challenger", ["4", "5", "6", "7"], "/models/challenger")

    assert generator.max_model_len == canonical_max_model_len()
    assert generator.max_new_tokens == settings.max_new_tokens


def test_vllm_worker_stops_on_qwen_im_end(monkeypatch):
    captured = {}

    class _SamplingParams:
        def __init__(self, **kwargs):
            captured["params"] = kwargs

    class _LLM:
        def __init__(self, **kwargs):
            captured["llm"] = kwargs

        def generate(self, prompts, params):
            captured["prompts"] = prompts
            captured["params_obj"] = params
            choice = types.SimpleNamespace(text="done")
            return [types.SimpleNamespace(outputs=[choice])]

    class _Queue:
        payload = None

        def put(self, payload):
            self.payload = payload

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
    )
    queue = _Queue()

    _vllm_worker(
        model="/models/challenger",
        gpu_ids=["0"],
        prompts=["<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n"],
        sample_ids=["sample-1"],
        max_new_tokens=77,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_model_len=None,
        enforce_eager=False,
        queue=queue,
    )

    assert captured["params"]["stop_token_ids"] == [248046]
    assert captured["llm"]["enable_prefix_caching"] is True
    assert queue.payload == {"results": [{"sample_id": "sample-1", "text": "done", "error": None}]}
