from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class Challenger(BaseModel):
    model_uri: str
    model_hash: str


class PreviousKing(BaseModel):
    model_uri: str
    model_hash: str
    king_version: int


class DatasetConfig(BaseModel):
    version: str
    manifest_uri: str
    manifest_hash: str
    sample_count: int
    max_turns_per_sample: int = 10
    sample_seed: str
    sampling_algo: str
    generation_batch_size: int = 8
    scoring_batch_size: int = 8
    sample_ids: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    judge_config_hash: str
    judge_count: int = 3
    allowed_scores: list[float] = Field(default_factory=lambda: [0, 0.5, 1])


class GpuRequest(BaseModel):
    accelerator: str = "B200"
    min_gpus: int = 8
    preferred_gpus: int = 8
    previous_king_gpu_count: int = 4
    challenger_gpu_count: int = 4
    tensor_parallel_size_per_model: int = 4


class EvalRequest(BaseModel):
    eval_run_id: UUID
    submission_id: UUID
    challenger: Challenger
    previous_king: PreviousKing
    dataset: DatasetConfig
    scoring: ScoringConfig
    gpu_request: GpuRequest = Field(default_factory=GpuRequest)
    artifact_prefix: str


class RemoteHost(BaseModel):
    id: str
    base_url: str
    role: str
    state: str
    gpu_count: int
    free_gpu_count: int
    accelerator_type: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    last_heartbeat_at: datetime | None = None


class EvalVerdict(BaseModel):
    type: str = "verdict"
    eval_run_id: UUID
    state: str
    challenger_won: bool | None = None
    score_challenger: float | None = None
    score_king: float | None = None
    judge_count: int | None = None
    allowed_scores: list[float] | None = None
    valid_turns: int | None = None
    total_turns: int | None = None
    king_vllm_errors: int = 0
    chal_vllm_errors: int = 0
    judge_errors: int = 0
    gpu_topology: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    fault_class: str | None = None
    fault_code: str | None = None
    fault_message: str | None = None
    retryable: bool | None = None


class SubmissionStatus(BaseModel):
    id: UUID
    state: str
    fault_class: str | None = None
    fault_code: str | None = None
    fault_message: str | None = None
    retry_count: int = 0
    updated_at: datetime | None = None
