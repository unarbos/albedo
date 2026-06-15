from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from .models import EvalRequest


@dataclass
class RemoteRun:
    remote_run_id: str
    request: EvalRequest
    state: str
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    worker_started: bool = False

    def set_state(self, state: str) -> None:
        self.state = state
        self.updated_at = datetime.now(UTC)

    def append_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.updated_at = datetime.now(UTC)

    def fail(self, *, fault_code: str, fault_message: str, retryable: bool = True) -> None:
        self.append_event(
            {
                "type": "verdict",
                "eval_run_id": str(self.request.eval_run_id),
                "state": "failed",
                "fault_class": "REMOTE_EVAL_FAULT",
                "fault_code": fault_code,
                "fault_message": fault_message,
                "retryable": retryable,
                "artifacts": {},
            }
        )
        self.set_state("failed")

    def as_status(self) -> dict[str, Any]:
        verdict = self.final_verdict()
        if verdict:
            return verdict
        return {
            "remote_run_id": self.remote_run_id,
            "eval_run_id": str(self.request.eval_run_id),
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def final_verdict(self) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("type") == "verdict":
                return event
        return None


class RemoteRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RemoteRun] = {}
        self._lock = RLock()

    def start(self, request: EvalRequest, *, challenger_won: bool = False, auto_verdict: bool = False) -> RemoteRun:
        remote_run_id = str(request.eval_run_id)
        with self._lock:
            existing = self._runs.get(remote_run_id)
            if existing:
                return existing

            run = RemoteRun(remote_run_id=remote_run_id, request=request, state="accepted")
            run.append_event(
                {
                    "type": "eval_started",
                    "remote_run_id": remote_run_id,
                    "eval_run_id": str(request.eval_run_id),
                    "message": "Remote eval run accepted",
                }
            )
            if auto_verdict:
                for event in _smoke_progress_and_verdict(request, challenger_won=challenger_won):
                    run.append_event(event)
                run.set_state("succeeded")
            self._runs[remote_run_id] = run
            return run

    def mark_worker_started(self, remote_run_id: str) -> RemoteRun | None:
        with self._lock:
            run = self._runs.get(remote_run_id)
            if not run or run.worker_started or run.state in {"succeeded", "failed"}:
                return None
            run.worker_started = True
            run.set_state("queued")
            return run

    def get(self, remote_run_id: str) -> RemoteRun | None:
        with self._lock:
            return self._runs.get(remote_run_id)

    def list_active(self) -> list[RemoteRun]:
        with self._lock:
            return [run for run in self._runs.values() if run.state not in {"succeeded", "failed"}]


def _smoke_progress_and_verdict(request: EvalRequest, *, challenger_won: bool) -> list[dict[str, Any]]:
    score_challenger = 0.58 if challenger_won else 0.42
    score_king = 1 - score_challenger
    return [
        {
            "type": "generation_started",
            "eval_run_id": str(request.eval_run_id),
            "message": "Smoke generation event emitted by control-plane test mode",
        },
        {
            "type": "scoring_started",
            "eval_run_id": str(request.eval_run_id),
            "message": "Smoke scoring event emitted by control-plane test mode",
        },
        {
            "type": "verdict",
            "eval_run_id": str(request.eval_run_id),
            "state": "succeeded",
            "challenger_won": challenger_won,
            "score_challenger": score_challenger,
            "score_king": score_king,
            "judge_count": request.scoring.judge_count,
            "allowed_scores": request.scoring.allowed_scores,
            "valid_turns": request.dataset.sample_count,
            "total_turns": request.dataset.sample_count,
            "king_vllm_errors": 0,
            "chal_vllm_errors": 0,
            "judge_errors": 0,
            "gpu_topology": {
                "accelerator": request.gpu_request.accelerator,
                "previous_king": ["0", "1", "2", "3"],
                "challenger": ["4", "5", "6", "7"],
                "tensor_parallel_size_per_model": request.gpu_request.tensor_parallel_size_per_model,
            },
            "artifacts": {},
        },
    ]
