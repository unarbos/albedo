"""Postgres access for the sanity pre-eval dispatcher (psycopg, mirrors the eval repository)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from albedo_eval_service.models import RemoteHost


@dataclass(frozen=True)
class ClaimedPreEval:
    # A claimed pre-eval job ready to dispatch to a worker.
    submission_id: UUID
    attempt_id: UUID
    remote_host: RemoteHost
    request: Any  # SanityRunRequest (kept untyped to avoid coupling to the worker package)


@dataclass(frozen=True)
class ActivePreEval:
    # An in-flight pre-eval recovered for reconciliation.
    submission_id: UUID
    attempt_id: UUID
    remote_host: RemoteHost
    repo: str
    digest: str
    prompts: list[str]

    @property
    def run_id(self) -> str:
        # The worker run_id equals the stage-attempt id (no separate storage needed).
        return str(self.attempt_id)


class PreEvalRepository:
    # Durable state transitions for pre-eval; transaction boundaries are explicit for crash recovery.

    def __init__(self, database_url: str, *, min_free_gpus: int = 1, max_retry_count: int = 5) -> None:
        self.database_url = database_url
        self._min_free_gpus = min_free_gpus
        self._max_retry_count = max_retry_count

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def claim_next_pre_eval(self, *, worker_id: str, lease_seconds: int, request_builder: Callable[..., Any]) -> ClaimedPreEval | None:
        # Claims the oldest claimable submission under an advisory lock.
        # Picks HIPPIUS_VALIDATED first (fresh), then PRE_EVAL_RETRYABLE (retries), both
        # capped at max_retry_count so a broken submission cannot loop forever.
        lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        with self._connect() as conn, conn.transaction():
            locked = conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('pre_eval')) AS locked").fetchone()
            if not locked or not locked["locked"]:
                return None

            submission = conn.execute(
                """
                SELECT ms.*, cc.block_hash
                FROM model_submissions ms
                JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                WHERE ms.state IN ('HIPPIUS_VALIDATED', 'PRE_EVAL_RETRYABLE')
                  AND ms.retry_count < %s
                  AND cc.block_hash IS NOT NULL
                ORDER BY
                  CASE WHEN ms.state = 'HIPPIUS_VALIDATED' THEN 0 ELSE 1 END ASC,
                  ms.priority ASC,
                  ms.retry_count ASC,
                  ms.created_at ASC
                FOR UPDATE OF ms SKIP LOCKED
                LIMIT 1
                """,
                (self._max_retry_count,),
            ).fetchone()
            if not submission:
                return None

            host = conn.execute(
                """
                SELECT id, base_url, role, state, gpu_count, free_gpu_count,
                       accelerator_type, capabilities, last_heartbeat_at
                FROM remote_gpu_hosts
                WHERE role = 'PRE_EVAL' AND state = 'READY' AND free_gpu_count >= %s
                ORDER BY free_gpu_count DESC, last_heartbeat_at DESC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (self._min_free_gpus,),
            ).fetchone()
            if not host:
                return None

            attempt_number = self._next_attempt_number(conn, submission["id"], "PRE_EVAL")
            attempt_id = uuid4()
            remote_host = RemoteHost(**host)
            request = request_builder(submission, remote_host, attempt_id)
            snapshot = {"host_id": remote_host.id, **request.model_dump(mode="json")}

            conn.execute(
                """
                INSERT INTO stage_attempts (
                    id, submission_id, stage, attempt_number, state, worker_id,
                    lease_expires_at, started_at, input_snapshot
                )
                VALUES (%s, %s, 'PRE_EVAL', %s, 'RUNNING', %s, %s, now(), %s)
                """,
                (
                    attempt_id,
                    submission["id"],
                    attempt_number,
                    worker_id,
                    lease_expires_at,
                    Jsonb(snapshot),
                ),
            )
            conn.execute(
                """
                UPDATE model_submissions
                SET state = 'PRE_EVAL_RUNNING', updated_at = now(),
                    fault_class = NULL, fault_code = NULL, fault_message = NULL
                WHERE id = %s
                """,
                (submission["id"],),
            )
            self.record_event_inside_tx(
                conn,
                submission_id=submission["id"],
                stage_attempt_id=attempt_id,
                event_type="pre_eval_claimed",
                severity="INFO",
                message=f"Pre-eval claimed by {worker_id} on host {remote_host.id}",
                data={"host_id": remote_host.id},
            )
            return ClaimedPreEval(
                submission_id=submission["id"],
                attempt_id=attempt_id,
                remote_host=remote_host,
                request=request,
            )

    def heartbeat_attempt(self, *, attempt_id: UUID, lease_seconds: int) -> None:
        # Extends the lease while the dispatcher is actively polling the worker.
        lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute(
                "UPDATE stage_attempts SET lease_expires_at = %s WHERE id = %s AND state = 'RUNNING'",
                (lease_expires_at, attempt_id),
            )

    def record_remote_event(self, *, submission_id: UUID, attempt_id: UUID, event: dict[str, Any]) -> None:
        # Persists a worker event under the attempt.
        with self._connect() as conn, conn.transaction():
            self.record_event_inside_tx(
                conn,
                submission_id=submission_id,
                stage_attempt_id=attempt_id,
                event_type=f"remote_{event.get('type', 'event')}",
                severity="INFO",
                message=str(event.get("message") or event.get("type") or "remote event"),
                data=event,
            )

    def list_reconcilable_pre_eval(self, *, limit: int = 10) -> list[ActivePreEval]:
        # Finds RUNNING pre-eval attempts (dispatcher may have crashed mid-poll) to replay.
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sa.id AS attempt_id, sa.submission_id, sa.input_snapshot,
                       h.id, h.base_url, h.role, h.state, h.gpu_count, h.free_gpu_count,
                       h.accelerator_type, h.capabilities, h.last_heartbeat_at
                FROM stage_attempts sa
                JOIN model_submissions ms ON ms.id = sa.submission_id
                JOIN remote_gpu_hosts h ON h.id = (sa.input_snapshot->>'host_id')
                WHERE sa.stage = 'PRE_EVAL' AND sa.state = 'RUNNING' AND ms.state = 'PRE_EVAL_RUNNING'
                ORDER BY sa.started_at ASC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        active: list[ActivePreEval] = []
        for row in rows:
            snap = row["input_snapshot"] or {}
            host = RemoteHost(
                id=row["id"],
                base_url=row["base_url"],
                role=row["role"],
                state=row["state"],
                gpu_count=row["gpu_count"],
                free_gpu_count=row["free_gpu_count"],
                accelerator_type=row["accelerator_type"],
                capabilities=row["capabilities"] or {},
                last_heartbeat_at=row["last_heartbeat_at"],
            )
            active.append(
                ActivePreEval(
                    submission_id=row["submission_id"],
                    attempt_id=row["attempt_id"],
                    remote_host=host,
                    repo=snap.get("model_uri", ""),
                    digest=snap.get("digest", ""),
                    prompts=list(snap.get("prompts", [])),
                )
            )
        return active

    def mark_pre_eval_passed(self, *, submission_id: UUID, attempt_id: UUID, repo: str, digest: str, responses: list[str], reason: str, timing: dict[str, Any],) -> None:
        # Records the cached result, completes the attempt, and advances the submission to PRE_EVAL_PASSED.
        with self._connect() as conn, conn.transaction():
            self._write_sanity_result(conn, repo, digest, True, reason, responses, timing)
            conn.execute(
                "UPDATE stage_attempts SET state = 'SUCCEEDED', finished_at = now(), result_summary = %s WHERE id = %s",
                (Jsonb({"passed": True, "reason": reason}), attempt_id),
            )
            conn.execute(
                "UPDATE model_submissions SET state = 'PRE_EVAL_PASSED', updated_at = now() WHERE id = %s",
                (submission_id,),
            )
            self.record_event_inside_tx(
                conn,
                submission_id=submission_id,
                stage_attempt_id=attempt_id,
                event_type="pre_eval_passed",
                severity="INFO",
                message="Pre-eval passed",
                data={},
            )

    def mark_pre_eval_failed(self, *, submission_id: UUID, attempt_id: UUID, repo: str, digest: str, fault_class: str, fault_code: str, fault_message: str, retryable: bool, responses: list[str] | None = None, artifact_uri: str | None = None,) -> None:
        # Fails the attempt; retryable -> PRE_EVAL_RETRYABLE (unless retries exhausted, then
        # TERMINAL_INVALID), terminal -> TERMINAL_INVALID with a cached sanity_results row.
        attempt_state = "FAILED_RETRYABLE" if retryable else "FAILED_TERMINAL"
        with self._connect() as conn, conn.transaction():
            if not retryable:
                self._write_sanity_result(
                    conn, repo, digest, False, fault_message, responses or [], {}
                )
                if artifact_uri:
                    self._insert_sanity_artifact(conn, submission_id, attempt_id, artifact_uri)
            conn.execute(
                """
                UPDATE stage_attempts
                SET state = %s, finished_at = now(), fault_class = %s, fault_code = %s, fault_message = %s
                WHERE id = %s
                """,
                (attempt_state, fault_class, fault_code, fault_message, attempt_id),
            )
            # Cap retryable failures: once retry_count reaches max, move to TERMINAL_INVALID so the
            # submission does not sit in PRE_EVAL_RETRYABLE forever unclaimed by the claim query.
            conn.execute(
                """
                UPDATE model_submissions
                SET state = CASE
                        WHEN %s AND retry_count + 1 >= %s THEN 'TERMINAL_INVALID'
                        WHEN %s THEN 'PRE_EVAL_RETRYABLE'
                        ELSE 'TERMINAL_INVALID'
                    END,
                    fault_class = %s, fault_code = %s, fault_message = %s,
                    retry_count = retry_count + 1, updated_at = now()
                WHERE id = %s
                """,
                (retryable, self._max_retry_count, retryable, fault_class, fault_code, fault_message, submission_id),
            )
            self.record_event_inside_tx(
                conn,
                submission_id=submission_id,
                stage_attempt_id=attempt_id,
                event_type="pre_eval_failed",
                severity="ERROR",
                message=fault_message,
                data={"fault_class": fault_class, "fault_code": fault_code, "retryable": retryable},
            )

    def sweep_abandoned_pre_eval(self, *, worker_id: str) -> int:
        # Reclaims expired RUNNING pre-eval attempts (dead dispatcher/host) back to the queue.
        # When retry_count already reached the cap the submission moves to TERMINAL_INVALID instead
        # of RETRYABLE so it does not sit in a ghost state that the claim query never picks up.
        with self._connect() as conn, conn.transaction():
            rows = conn.execute(
                """
                SELECT sa.id AS attempt_id, sa.submission_id, ms.retry_count
                FROM stage_attempts sa
                JOIN model_submissions ms ON ms.id = sa.submission_id
                WHERE sa.stage = 'PRE_EVAL' AND sa.state = 'RUNNING'
                  AND sa.lease_expires_at < now() AND ms.state = 'PRE_EVAL_RUNNING'
                FOR UPDATE OF sa, ms SKIP LOCKED
                """
            ).fetchall()
            for row in rows:
                exhausted = row["retry_count"] + 1 >= self._max_retry_count
                next_state = "TERMINAL_INVALID" if exhausted else "PRE_EVAL_RETRYABLE"
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET state = 'ABANDONED', finished_at = now(), fault_class = 'INFRA_FAULT',
                        fault_code = 'pre_eval_lease_expired', fault_message = 'lease expired before completion'
                    WHERE id = %s
                    """,
                    (row["attempt_id"],),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = %s, fault_class = 'INFRA_FAULT',
                        fault_code = 'pre_eval_lease_expired', fault_message = 'lease expired before completion',
                        retry_count = retry_count + 1, updated_at = now()
                    WHERE id = %s
                    """,
                    (next_state, row["submission_id"]),
                )
                self.record_event_inside_tx(
                    conn,
                    submission_id=row["submission_id"],
                    stage_attempt_id=row["attempt_id"],
                    event_type="pre_eval_abandoned",
                    severity="WARN",
                    message=f"Pre-eval lease expired before completion (-> {next_state})",
                    data={"worker_id": worker_id, "exhausted": exhausted},
                )
            return len(rows)

    @staticmethod
    def _insert_sanity_artifact(conn: psycopg.Connection, submission_id: UUID, attempt_id: UUID, uri: str) -> None:
        # Records the uploaded fault report so the dashboard can link it (artifact_type SANITY_RESULT).
        bucket, object_key = (None, None)
        if uri.startswith("s3://"):
            bucket, _, object_key = uri[len("s3://") :].partition("/")
        conn.execute(
            """
            INSERT INTO artifacts (
                id, submission_id, stage_attempt_id, artifact_type,
                storage_backend, uri, bucket, object_key, content_type
            )
            VALUES (%s, %s, %s, 'SANITY_RESULT', 's3', %s, %s, %s, 'application/json')
            """,
            (uuid4(), submission_id, attempt_id, uri, bucket or None, object_key or None),
        )

    @staticmethod
    def _write_sanity_result(conn: psycopg.Connection, repo: str, digest: str, passed: bool, reason: str, responses: list[str], timing: dict[str, Any],) -> None:
        # Upserts the digest-keyed cache row (first verdict wins).
        conn.execute(
            """
            INSERT INTO sanity_results (repo, digest, passed, reason, responses, timing, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (digest) DO NOTHING
            """,
            (repo, digest, passed, reason, Jsonb(responses), Jsonb(timing)),
        )

    @staticmethod
    def record_event_inside_tx(conn: psycopg.Connection, *, submission_id: UUID, stage_attempt_id: UUID | None, event_type: str, severity: str, message: str, data: dict[str, Any],) -> None:
        # Inserts an audit event row.
        conn.execute(
            """
            INSERT INTO events (id, submission_id, stage_attempt_id, event_type, severity, message, data)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (uuid4(), submission_id, stage_attempt_id, event_type, severity, message, Jsonb(data)),
        )

    @staticmethod
    def _next_attempt_number(conn: psycopg.Connection, submission_id: UUID, stage: str) -> int:
        # Returns the next attempt number for this submission/stage.
        row = conn.execute(
            """
            SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
            FROM stage_attempts WHERE submission_id = %s AND stage = %s
            """,
            (submission_id, stage),
        ).fetchone()
        return int(row["next_attempt"])
