from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from sanity_remote.models import SanityRunRequest


@dataclass
class SanityRun:
    run_id: str
    request: SanityRunRequest
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

    def succeed(self, *, responses: list[str], heuristics: list[dict[str, Any]]) -> None:
        self.append_event(
            {
                "type": "result",
                "run_id": self.run_id,
                "state": "succeeded",
                "responses": responses,
                "heuristics": heuristics,
            }
        )
        self.set_state("succeeded")

    def fail(self, *, fault_code: str, fault_message: str, retryable: bool = True) -> None:
        self.append_event(
            {
                "type": "result",
                "run_id": self.run_id,
                "state": "failed",
                "fault_code": fault_code,
                "fault_message": fault_message,
                "retryable": retryable,
            }
        )
        self.set_state("failed")

    def final_result(self) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("type") == "result":
                return event
        return None

    def as_status(self) -> dict[str, Any]:
        result = self.final_result()
        if result:
            return result
        return {
            "run_id": self.run_id,
            "digest": self.request.digest,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SanityRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, SanityRun] = {}
        self._lock = RLock()

    def start(self, request: SanityRunRequest) -> SanityRun:
        with self._lock:
            existing = self._runs.get(request.run_id)
            if existing:
                return existing
            run = SanityRun(run_id=request.run_id, request=request, state="accepted")
            run.append_event(
                {"type": "accepted", "run_id": request.run_id, "digest": request.digest}
            )
            self._runs[request.run_id] = run
            return run

    def mark_worker_started(self, run_id: str) -> SanityRun | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run.worker_started or run.state in {"succeeded", "failed"}:
                return None
            run.worker_started = True
            run.set_state("queued")
            return run

    def get(self, run_id: str) -> SanityRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_active(self) -> list[SanityRun]:
        with self._lock:
            return [run for run in self._runs.values() if run.state not in {"succeeded", "failed"}]
