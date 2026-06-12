"""In-memory run store for the stateless sanity worker - events polled by the dispatcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from sanity_remote.models import SanityRunRequest


@dataclass
class SanityRun:
    # One generation run held in memory; a dead worker simply loses it and the dispatcher requeues.
    run_id: str
    request: SanityRunRequest
    state: str
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    worker_started: bool = False

    def set_state(self, state: str) -> None:
        # Updates state and bumps the timestamp.
        self.state = state
        self.updated_at = datetime.now(UTC)

    def append_event(self, event: dict[str, Any]) -> None:
        # Records a progress/result event for the dispatcher to poll.
        self.events.append(event)
        self.updated_at = datetime.now(UTC)

    def succeed(self, *, responses: list[str], heuristics: list[dict[str, Any]]) -> None:
        # Emits the terminal success result carrying the generated responses + heuristic verdicts.
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
        # Emits a terminal failure result (retryable=infra, else a miner/model fault).
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
        # Returns the terminal result event if present.
        for event in reversed(self.events):
            if event.get("type") == "result":
                return event
        return None

    def as_status(self) -> dict[str, Any]:
        # Returns the final result if done, else a lightweight status snapshot.
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
    # Thread-safe map of run_id -> SanityRun; idempotent on run_id.
    def __init__(self) -> None:
        self._runs: dict[str, SanityRun] = {}
        self._lock = RLock()

    def start(self, request: SanityRunRequest) -> SanityRun:
        # Registers a run (returns the existing one on a duplicate run_id) and records acceptance.
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
        # Moves an accepted run to queued once; None if already started or terminal.
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run.worker_started or run.state in {"succeeded", "failed"}:
                return None
            run.worker_started = True
            run.set_state("queued")
            return run

    def get(self, run_id: str) -> SanityRun | None:
        # Returns the run for this id, or None.
        with self._lock:
            return self._runs.get(run_id)

    def list_active(self) -> list[SanityRun]:
        # Returns runs that have not reached a terminal state.
        with self._lock:
            return [run for run in self._runs.values() if run.state not in {"succeeded", "failed"}]
