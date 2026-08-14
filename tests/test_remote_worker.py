from __future__ import annotations

import json
import queue
import sys
import types
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_config import RemoteSettings
from albedo_eval_service.modelstore.canonical_model_config import canonical_max_model_len
from albedo_eval_service.modelstore.resolver import ResolvedModel
from albedo_eval_service.remote.generation import (
    GenerationResult,
    VllmProcessGenerator,
    _generate_payload,
    _vllm_worker,
    format_scored_trajectory,
)
from albedo_eval_service.remote.state import RemoteRun
from albedo_eval_service.remote.worker import (
    ObservationResult,
    RemoteEvalWorker,
    _completion_observation,
    _merge_trajectory_results,
    _missing_command_observation,
    _next_turn_samples,
)
from albedo_eval_service.scoring.scoring_client import ScoringResult
from albedo_eval_service.shared.models import (
    Challenger,
    DatasetConfig,
    EvalRequest,
    PreviousKing,
    ScoringConfig,
)
from albedo_eval_service.shared.observation_format import TRUNCATION_SENTINEL


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
        # a real turn always carries a bash block; without one the worker now short-circuits to a
        # missing-command observation instead of asking the simulator
        return [
            GenerationResult(
                sample_id=sample.sample_id,
                text=f"{sample.prompt}{suffix}\n```bash\nls\n```",
            )
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
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
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
        trajectory_assistant_turns=2,
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
    assert [call["side"] for call in generate_calls].count("previous_king") == 12
    assert [call["side"] for call in generate_calls].count("challenger") == 12
    king_calls = [call for call in generate_calls if call["side"] == "previous_king"]
    assert all(len(call["sample_ids"]) == 2 for call in king_calls[:8])
    assert all(len(call["sample_ids"]) == 1 for call in king_calls[8:])
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
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
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


def test_submit_echo_stops_future_trajectory_turns(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages", lambda messages, **_kwargs: "next"
    )
    sample = types.SimpleNamespace(
        sample_id="sample-1",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )
    observation = ObservationResult(
        "sample-1", "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )

    assert (
        _next_turn_samples(
            [sample],
            [
                GenerationResult(
                    "sample-1", "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                )
            ],
            {("challenger", "sample-1"): observation},
            side="challenger",
        )
        == []
    )

    merged = _merge_trajectory_results(
        [sample],
        [
            [
                GenerationResult(
                    "sample-1", "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                )
            ],
            [],
        ],
        [{("challenger", "sample-1"): observation}],
        side="challenger",
        token_limit=16384,
    )

    assert merged[0].error is None
    assert "CANDIDATE OUTPUT 2" not in merged[0].text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in merged[0].text


def test_generate_payload_flags_only_responses_that_hit_the_per_response_cap():
    class _LLM:
        def __init__(self, completions):
            self._completions = completions

        def generate(self, prompts, params):
            return [types.SimpleNamespace(outputs=[c]) for c in self._completions]

    at_cap = types.SimpleNamespace(text="a", finish_reason="length", token_ids=[0] * 16384)
    context_bound = types.SimpleNamespace(text="b", finish_reason="length", token_ids=[0] * 900)
    finished = types.SimpleNamespace(text="c", finish_reason="stop", token_ids=[0] * 12)

    payload = _generate_payload(
        _LLM([at_cap, context_bound, finished]),
        None,
        ["p1", "p2", "p3"],
        ["s1", "s2", "s3"],
        16384,
    )

    assert {r["sample_id"]: r["truncated"] for r in payload["results"]} == {
        "s1": True,
        "s2": False,
        "s3": False,
    }


def _trajectory_sample(sample_id: str = "sample-1"):
    return types.SimpleNamespace(
        sample_id=sample_id,
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )


def test_truncated_response_ends_trajectory_and_stays_valid(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages", lambda messages, **_kwargs: "next"
    )
    sample = _trajectory_sample()
    oversized = "x" * 200
    truncated = GenerationResult("sample-1", oversized, truncated=True)
    observation = ObservationResult("sample-1", "")

    assert (
        _next_turn_samples(
            [sample],
            [truncated],
            {("challenger", "sample-1"): observation},
            side="challenger",
        )
        == []
    )

    merged = _merge_trajectory_results(
        [sample],
        [[truncated], []],
        [{("challenger", "sample-1"): observation}],
        side="challenger",
        token_limit=16384,
    )

    assert merged[0].error is None
    assert merged[0].truncated is True
    assert TRUNCATION_SENTINEL in merged[0].text
    assert "16384" in merged[0].text
    assert oversized not in merged[0].text
    assert "CANDIDATE OUTPUT 2" not in merged[0].text


def _pairs_worker(scorer):
    return RemoteEvalWorker(
        RemoteSettings(
            scoring_backend="mock", upload_artifacts=False, resolve_model_artifacts=False
        ),
        scorer=scorer,
    )


def test_score_pairs_is_terminal_when_too_few_valid_pairs():
    class NeverCalled:
        def score(self, **_kwargs):
            raise AssertionError("scorer must not run when too few pairs are valid")

        def simulate_observation(self, **_kwargs):
            return ""

    samples = [_trajectory_sample(f"s{index}") for index in range(10)]
    king_results = [GenerationResult(sample.sample_id, "king") for sample in samples]
    challenger_results = [
        GenerationResult(sample.sample_id, "", "vllm timed out") for sample in samples[:9]
    ] + [GenerationResult("s9", "challenger")]

    summary = _pairs_worker(NeverCalled())._score_pairs(
        request=_request(),
        samples=samples,
        king_results=king_results,
        challenger_results=challenger_results,
    )["summary"]

    assert summary["state"] == "failed"
    assert summary["fault_class"] == "MINER_FAULT"
    assert summary["fault_code"] == "insufficient_valid_samples"
    assert summary["retryable"] is False
    assert summary["valid_turns"] == 1
    assert summary["total_turns"] == 10


def test_score_pairs_counts_truncated_pairs_as_valid():
    scored = {}

    class Recording:
        def score(
            self, *, request, samples, king_results, challenger_results, category_prep_id=None
        ):
            scored["samples"] = len(samples)
            return ScoringResult(records=[], summary={"state": "succeeded"})

        def simulate_observation(self, **_kwargs):
            return ""

    samples = [_trajectory_sample(f"s{index}") for index in range(10)]
    king_results = [GenerationResult(sample.sample_id, "king") for sample in samples]
    challenger_results = [
        GenerationResult(sample.sample_id, "notice", truncated=True) for sample in samples
    ]

    result = _pairs_worker(Recording())._score_pairs(
        request=_request(),
        samples=samples,
        king_results=king_results,
        challenger_results=challenger_results,
    )

    assert scored["samples"] == 10
    assert result["summary"]["state"] == "succeeded"


def test_submit_echo_bypasses_observation_simulator(tmp_path):
    class FailingScorer:
        def simulate_observation(self, **_kwargs):
            raise AssertionError("simulator should not run for submit echo")

    sample = types.SimpleNamespace(
        sample_id="mini-coder/data/train-00000.parquet:1:0",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )
    result = GenerationResult(
        sample.sample_id, "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
    )
    worker = RemoteEvalWorker(
        RemoteSettings(dataset_root=str(tmp_path), scoring_backend="mock"),
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
        scorer=FailingScorer(),
    )

    observations = worker._simulate_observations(
        request=_request(),
        samples_by_side={"challenger": [sample]},
        results_by_side={"challenger": [result]},
    )

    assert observations[("challenger", sample.sample_id)].observation == _completion_observation(
        sample
    )


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
            choice = types.SimpleNamespace(text="done", finish_reason="stop")
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
    assert queue.payload == {
        "results": [{"sample_id": "sample-1", "text": "done", "error": None, "truncated": False}]
    }


def test_prefetch_repo_context_fires_only_when_configured(monkeypatch):
    import threading

    import albedo_eval_service.remote.worker as remote_worker_module
    from albedo_eval_service.remote.dataset import EvalSample

    recorded: dict[str, object] = {}
    posted = threading.Event()

    def fake_post(url, json=None, timeout=None):
        recorded["url"] = url
        recorded["json"] = json
        posted.set()

    monkeypatch.setattr(remote_worker_module.httpx, "post", fake_post)
    samples = [EvalSample(sample_id="data/train-00000.parquet:0:0", prompt="p")]

    enabled = RemoteEvalWorker(
        RemoteSettings(
            repo_context_url="http://127.0.0.1:8093/",
            upload_artifacts=False,
            scoring_backend="mock",
        )
    )
    enabled._prefetch_repo_context(_request(), samples)
    assert posted.wait(2.0)
    assert recorded["url"] == "http://127.0.0.1:8093/prefetch"
    assert recorded["json"] == {"sample_ids": ["data/train-00000.parquet:0:0"]}

    posted.clear()
    disabled = RemoteEvalWorker(RemoteSettings(upload_artifacts=False, scoring_backend="mock"))
    disabled._prefetch_repo_context(_request(), samples)
    assert not posted.wait(0.2)


def test_missing_bash_command_bypasses_observation_simulator(tmp_path):
    """A turn with no bash block must get a real command error, never a simulated observation.

    Previously _command_only() fell back to the whole assistant message, so the simulator
    invented a filesystem for models that emit a JSON tool call instead of a bash fence.
    """

    class FailingScorer:
        def simulate_observation(self, **_kwargs):
            raise AssertionError("simulator must not run when there is no bash command")

    sample = types.SimpleNamespace(
        sample_id="mini-coder-rs/data/train-00000.parquet:1156:2",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )
    result = GenerationResult(
        sample.sample_id,
        'THOUGHT: read the file\n{"command": "sed -n \'590,670p\' /testbed/src/naive/date/mod.rs"}',
    )
    worker = RemoteEvalWorker(
        RemoteSettings(dataset_root=str(tmp_path), scoring_backend="mock"),
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
        scorer=FailingScorer(),
    )

    observations = worker._simulate_observations(
        request=_request(),
        samples_by_side={"challenger": [sample]},
        results_by_side={"challenger": [result]},
    )

    observation = observations[("challenger", sample.sample_id)].observation
    assert observation == _missing_command_observation(sample)
    assert "No bash command found" in observation
    assert "<returncode>2</returncode>" in observation
