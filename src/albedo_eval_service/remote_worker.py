from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, TypeVar

from .dataset_manifest import load_manifest_file
from .models import EvalRequest
from .remote_config import RemoteSettings
from .remote_dataset import EvalSample, load_swe_zero_samples
from .remote_generation import GenerationResult, Generator, VllmProcessGenerator
from .remote_state import RemoteRun
from .sampling import swe_zero_manifest_sample_ids

GeneratorFactory = Callable[[str, list[str], str], Generator]
T = TypeVar("T")


@dataclass(frozen=True)
class GpuTopology:
    accelerator: str
    previous_king: list[str]
    challenger: list[str]
    tensor_parallel_size_per_model: int

    def as_dict(self) -> dict[str, object]:
        return {
            "accelerator": self.accelerator,
            "previous_king": self.previous_king,
            "challenger": self.challenger,
            "tensor_parallel_size_per_model": self.tensor_parallel_size_per_model,
        }


class RemoteEvalWorker:
    def __init__(self, settings: RemoteSettings, *, generator_factory: GeneratorFactory | None = None):
        self.settings = settings
        self._generator_factory = generator_factory or self._vllm_generator

    def execute(self, run: RemoteRun) -> None:
        try:
            self._execute(run)
        except Exception as exc:
            run.fail(fault_code="remote_worker_failed", fault_message=f"{type(exc).__name__}: {exc}")

    def _execute(self, run: RemoteRun) -> None:
        request = run.request
        topology = self._topology(request)
        samples = self._load_samples(request)
        run.set_state("generating")
        run.append_event(
            {
                "type": "generation_started",
                "eval_run_id": str(request.eval_run_id),
                "gpu_topology": topology.as_dict(),
                "sample_count": len(samples),
                "generation_batch_size": request.dataset.generation_batch_size,
            }
        )

        king_model = self._model_for_side(request, side="previous_king")
        challenger_model = self._model_for_side(request, side="challenger")
        king_generator = self._generator_factory("previous_king", topology.previous_king, king_model)
        challenger_generator = self._generator_factory("challenger", topology.challenger, challenger_model)
        with ThreadPoolExecutor(max_workers=2) as executor:
            king_future = executor.submit(king_generator.generate, samples)
            challenger_future = executor.submit(challenger_generator.generate, samples)
            king_results = king_future.result()
            challenger_results = challenger_future.result()

        self._emit_generation_batches(run, request, samples, king_results, challenger_results, topology)
        run.set_state("scoring")
        run.append_event(
            {
                "type": "scoring_started",
                "eval_run_id": str(request.eval_run_id),
                "scoring_batch_size": request.dataset.scoring_batch_size,
            }
        )
        scoring_records = self._mock_score_pairs(
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            judge_count=request.scoring.judge_count,
        )
        self._emit_scoring_batches(run, request, scoring_records)
        verdict = self._mock_verdict(
            request=request,
            topology=topology,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            scoring_records=scoring_records,
        )
        run.append_event(verdict)
        run.set_state(str(verdict["state"]))

    def _load_samples(self, request: EvalRequest) -> list[EvalSample]:
        if not self.settings.dataset_root:
            raise ValueError("ALBEDO_REMOTE_DATASET_ROOT is required for SWE-ZERO parquet loading")

        sample_ids = list(request.dataset.sample_ids)
        if not sample_ids:
            manifest_path = Path(self.settings.dataset_root) / "manifest.json"
            manifest = load_manifest_file(manifest_path, expected_sha256=request.dataset.manifest_hash)
            sample_ids = swe_zero_manifest_sample_ids(
                manifest,
                block_hash=request.dataset.sample_seed,
                sample_count=request.dataset.sample_count,
                max_turns_per_sample=request.dataset.max_turns_per_sample,
            )
        return load_swe_zero_samples(dataset_root=self.settings.dataset_root, sample_ids=sample_ids)

    def _model_for_side(self, request: EvalRequest, *, side: str) -> str:
        if side == "previous_king":
            return self.settings.previous_king_model or request.previous_king.model_uri
        if side == "challenger":
            return self.settings.challenger_model or request.challenger.model_uri
        raise ValueError(f"unsupported model side: {side}")

    def _vllm_generator(self, side: str, gpu_ids: list[str], model: str) -> Generator:
        if self.settings.generation_backend != "vllm":
            raise ValueError(f"unsupported generation backend: {self.settings.generation_backend}")
        if not model:
            raise ValueError(f"missing model setting for {side}")
        return VllmProcessGenerator(
            model=model,
            gpu_ids=gpu_ids,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_model_len=self.settings.max_model_len,
            enforce_eager=self.settings.enforce_eager,
        )

    def _topology(self, request: EvalRequest) -> GpuTopology:
        previous_king = _parse_gpu_ids(self.settings.previous_king_gpu_ids)
        challenger = _parse_gpu_ids(self.settings.challenger_gpu_ids)
        if request.gpu_request.min_gpus != 8 or request.gpu_request.preferred_gpus != 8:
            raise ValueError("remote eval target requires an 8-GPU request")
        if request.gpu_request.tensor_parallel_size_per_model != 4:
            raise ValueError("remote eval requires tensor_parallel_size_per_model=4")
        if len(previous_king) != request.gpu_request.previous_king_gpu_count:
            raise ValueError("previous king GPU group does not match request")
        if len(challenger) != request.gpu_request.challenger_gpu_count:
            raise ValueError("challenger GPU group does not match request")
        if len(previous_king) != 4 or len(challenger) != 4:
            raise ValueError("remote eval requires fixed 4 GPU groups for both models")
        overlap = set(previous_king) & set(challenger)
        if overlap:
            raise ValueError(f"GPU groups overlap: {sorted(overlap)}")
        return GpuTopology(
            accelerator=self.settings.accelerator_type,
            previous_king=previous_king,
            challenger=challenger,
            tensor_parallel_size_per_model=request.gpu_request.tensor_parallel_size_per_model,
        )

    def _emit_generation_batches(
        self,
        run: RemoteRun,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        topology: GpuTopology,
    ) -> None:
        king_by_id = {result.sample_id: result for result in king_results}
        challenger_by_id = {result.sample_id: result for result in challenger_results}
        for batch_idx, batch in enumerate(_chunks(samples, request.dataset.generation_batch_size), start=1):
            sample_ids = [sample.sample_id for sample in batch]
            run.append_event(
                {
                    "type": "generation_batch_done",
                    "eval_run_id": str(request.eval_run_id),
                    "batch_id": f"gen-{batch_idx:04d}",
                    "sample_ids": sample_ids,
                    "models": ["challenger", "previous_king"],
                    "gpu_ids": topology.previous_king + topology.challenger,
                    "king_errors": sum(1 for sample_id in sample_ids if king_by_id[sample_id].error),
                    "chal_errors": sum(1 for sample_id in sample_ids if challenger_by_id[sample_id].error),
                    "state": "succeeded",
                }
            )

    def _emit_scoring_batches(self, run: RemoteRun, request: EvalRequest, scoring_records: list[dict[str, object]]) -> None:
        for batch_idx, batch in enumerate(_chunks(scoring_records, request.dataset.scoring_batch_size), start=1):
            run.append_event(
                {
                    "type": "scoring_batch_done",
                    "eval_run_id": str(request.eval_run_id),
                    "batch_id": f"score-{batch_idx:04d}",
                    "sample_ids": [str(record["sample_id"]) for record in batch],
                    "judge_config_hash": request.scoring.judge_config_hash,
                    "judge_count": request.scoring.judge_count,
                    "allowed_scores": request.scoring.allowed_scores,
                    "state": "succeeded",
                }
            )

    def _mock_score_pairs(
        self,
        *,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        judge_count: int,
    ) -> list[dict[str, object]]:
        king_by_id = {result.sample_id: result for result in king_results}
        challenger_by_id = {result.sample_id: result for result in challenger_results}
        records: list[dict[str, object]] = []
        for sample in samples:
            king = king_by_id[sample.sample_id]
            challenger = challenger_by_id[sample.sample_id]
            if king.error or challenger.error:
                continue
            score = 1.0 if len(challenger.text) > len(king.text) else 0.5 if len(challenger.text) == len(king.text) else 0.0
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "judge_scores": [score] * judge_count,
                    "sample_score": score,
                }
            )
        return records

    def _mock_verdict(
        self,
        *,
        request: EvalRequest,
        topology: GpuTopology,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        scoring_records: list[dict[str, object]],
    ) -> dict[str, object]:
        king_errors = sum(1 for result in king_results if result.error)
        chal_errors = sum(1 for result in challenger_results if result.error)
        sample_scores = [float(record["sample_score"]) for record in scoring_records]
        score_challenger = mean(sample_scores) if sample_scores else 0.0
        score_king = 1 - score_challenger
        return {
            "type": "verdict",
            "eval_run_id": str(request.eval_run_id),
            "state": "succeeded" if sample_scores else "failed",
            "challenger_won": score_challenger > score_king if sample_scores else None,
            "score_challenger": score_challenger if sample_scores else None,
            "score_king": score_king if sample_scores else None,
            "judge_count": request.scoring.judge_count,
            "allowed_scores": request.scoring.allowed_scores,
            "valid_turns": len(scoring_records),
            "total_turns": len(samples),
            "king_vllm_errors": king_errors,
            "chal_vllm_errors": chal_errors,
            "judge_errors": 0,
            "gpu_topology": topology.as_dict(),
            "artifacts": {},
            "fault_class": "REMOTE_EVAL_FAULT" if not sample_scores else None,
            "fault_code": "no_valid_generated_pairs" if not sample_scores else None,
            "fault_message": "No sample pair had both king and challenger output" if not sample_scores else None,
            "retryable": True if not sample_scores else None,
        }


def _parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU group contains duplicate IDs")
    return gpu_ids


def _chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]
