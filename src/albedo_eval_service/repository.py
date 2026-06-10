from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .artifacts import artifact_records_from_verdict
from .models import EvalRequest, RemoteHost, SubmissionStatus


@dataclass(frozen=True)
class ClaimedEval:
    submission_id: UUID
    attempt_id: UUID
    eval_run_id: UUID
    remote_host: RemoteHost
    request: EvalRequest


@dataclass(frozen=True)
class ActiveEval:
    submission_id: UUID
    attempt_id: UUID
    eval_run_id: UUID
    remote_run_id: str
    remote_host: RemoteHost


class EvalRepository:
    """Postgres access for eval dispatching.

    Methods keep transaction boundaries explicit because the service relies on
    durable state transitions for crash recovery.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def get_submission(self, submission_id: UUID) -> SubmissionStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, state, fault_class, fault_code, fault_message, retry_count, updated_at
                FROM model_submissions
                WHERE id = %s
                """,
                (submission_id,),
            ).fetchone()
        return SubmissionStatus(**row) if row else None

    def claim_next_eval(self, *, worker_id: str, lease_seconds: int, request_builder) -> ClaimedEval | None:
        lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)

        with self._connect() as conn:
            with conn.transaction():
                locked = conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('full_eval')) AS locked").fetchone()
                if not locked or not locked["locked"]:
                    return None

                running = conn.execute(
                    "SELECT id FROM model_submissions WHERE state = 'EVAL_RUNNING' LIMIT 1"
                ).fetchone()
                if running:
                    return None

                submission = conn.execute(
                    """
                    SELECT ms.*, cc.block_hash
                    FROM model_submissions ms
                    JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                    WHERE ms.state = 'EVAL_QUEUED'
                      AND cc.block_hash IS NOT NULL
                    ORDER BY ms.priority ASC, ms.created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """
                ).fetchone()
                if not submission:
                    return None

                host = conn.execute(
                    """
                    SELECT id, base_url, role, state, gpu_count, free_gpu_count,
                           accelerator_type, capabilities, last_heartbeat_at
                    FROM remote_gpu_hosts
                    WHERE role = 'EVAL'
                      AND state = 'READY'
                      AND free_gpu_count >= 8
                    ORDER BY free_gpu_count DESC, last_heartbeat_at DESC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """
                ).fetchone()
                if not host:
                    return None

                king = conn.execute(
                    """
                    SELECT kv.version AS king_version, kv.model_hash, a.uri AS model_uri
                    FROM reigns r
                    JOIN reign_members rm ON rm.reign_id = r.id AND rm.slot = 1
                    JOIN king_versions kv ON kv.id = rm.king_version_id
                    JOIN artifacts a ON a.id = kv.artifact_id
                    WHERE r.state = 'ACTIVE'
                    ORDER BY r.version DESC
                    LIMIT 1
                    """
                ).fetchone()
                if not king:
                    self._mark_retryable_inside_tx(
                        conn,
                        submission["id"],
                        "INFRA_FAULT",
                        "missing_active_lead_king",
                        "No active lead king is available for eval",
                    )
                    return None

                attempt_number = self._next_attempt_number(conn, submission["id"], "EVAL")
                attempt_id = uuid4()
                eval_run_id = uuid4()
                remote_host = RemoteHost(**host)
                request = request_builder(submission, king, remote_host, eval_run_id)

                conn.execute(
                    """
                    INSERT INTO stage_attempts (
                        id, submission_id, stage, attempt_number, state, worker_id,
                        lease_expires_at, started_at, input_snapshot
                    )
                    VALUES (%s, %s, 'EVAL', %s, 'RUNNING', %s, %s, now(), %s)
                    """,
                    (
                        attempt_id,
                        submission["id"],
                        attempt_number,
                        worker_id,
                        lease_expires_at,
                        Jsonb(request.model_dump(mode="json")),
                    ),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = 'EVAL_RUNNING', updated_at = now(), fault_class = NULL,
                        fault_code = NULL, fault_message = NULL
                    WHERE id = %s
                    """,
                    (submission["id"],),
                )
                conn.execute(
                    """
                    INSERT INTO eval_runs (
                        id, submission_id, stage_attempt_id, king_model_hash,
                        challenger_model_hash, remote_host_id, state, gpu_count,
                        dataset_version, dataset_manifest_hash, dataset_sample_seed, dataset_sample_ids,
                        dataset_max_turns_per_sample, dataset_sampling_algo, judge_config_hash, judge_count,
                        sample_count, started_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, 'DISPATCHED', 8,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
                    )
                    """,
                    (
                        eval_run_id,
                        submission["id"],
                        attempt_id,
                        king["model_hash"],
                        submission["model_hash"],
                        remote_host.id,
                        request.dataset.version,
                        request.dataset.manifest_hash,
                        request.dataset.sample_seed,
                        request.dataset.sample_ids,
                        request.dataset.max_turns_per_sample,
                        request.dataset.sampling_algo,
                        request.scoring.judge_config_hash,
                        request.scoring.judge_count,
                        request.dataset.sample_count,
                    ),
                )
                self.record_event_inside_tx(
                    conn,
                    submission_id=submission["id"],
                    stage_attempt_id=attempt_id,
                    event_type="eval_claimed",
                    severity="INFO",
                    message=f"Eval claimed by {worker_id} on host {remote_host.id}",
                    data={"eval_run_id": str(eval_run_id), "remote_host_id": remote_host.id},
                )

            return ClaimedEval(
                submission_id=submission["id"],
                attempt_id=attempt_id,
                eval_run_id=eval_run_id,
                remote_host=remote_host,
                request=request,
            )

    def record_remote_event(self, *, submission_id: UUID, attempt_id: UUID, event: dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.transaction():
                self.record_event_inside_tx(
                    conn,
                    submission_id=submission_id,
                    stage_attempt_id=attempt_id,
                    event_type=f"remote_{event.get('type', 'event')}",
                    severity="INFO",
                    message=str(event.get("message") or event.get("type") or "Remote eval event"),
                    data=event,
                )


    def list_reconcilable_eval_runs(self, *, limit: int = 10) -> list[ActiveEval]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT er.id AS eval_run_id, er.remote_run_id, er.submission_id,
                       er.stage_attempt_id AS attempt_id,
                       h.id, h.base_url, h.role, h.state, h.gpu_count,
                       h.free_gpu_count, h.accelerator_type, h.capabilities,
                       h.last_heartbeat_at
                FROM eval_runs er
                JOIN stage_attempts sa ON sa.id = er.stage_attempt_id
                JOIN model_submissions ms ON ms.id = er.submission_id
                JOIN remote_gpu_hosts h ON h.id = er.remote_host_id
                WHERE er.remote_run_id IS NOT NULL
                  AND er.state IN ('DISPATCHED', 'GENERATING', 'SCORING', 'VERDICT_READY')
                  AND sa.state = 'RUNNING'
                  AND ms.state = 'EVAL_RUNNING'
                ORDER BY er.started_at ASC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        active: list[ActiveEval] = []
        for row in rows:
            active.append(
                ActiveEval(
                    submission_id=row["submission_id"],
                    attempt_id=row["attempt_id"],
                    eval_run_id=row["eval_run_id"],
                    remote_run_id=row["remote_run_id"],
                    remote_host=RemoteHost(
                        id=row["id"],
                        base_url=row["base_url"],
                        role=row["role"],
                        state=row["state"],
                        gpu_count=row["gpu_count"],
                        free_gpu_count=row["free_gpu_count"],
                        accelerator_type=row["accelerator_type"],
                        capabilities=row["capabilities"] or {},
                        last_heartbeat_at=row["last_heartbeat_at"],
                    ),
                )
            )
        return active

    def heartbeat_attempt(self, *, attempt_id: UUID, lease_seconds: int) -> None:
        lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stage_attempts
                SET lease_expires_at = %s
                WHERE id = %s AND state = 'RUNNING'
                """,
                (lease_expires_at, attempt_id),
            )

    def set_remote_run_id(self, *, eval_run_id: UUID, remote_run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE eval_runs
                SET remote_run_id = %s
                WHERE id = %s
                """,
                (remote_run_id, eval_run_id),
            )


    def sweep_abandoned_eval_attempts(self, *, worker_id: str) -> int:
        with self._connect() as conn:
            with conn.transaction():
                rows = conn.execute(
                    """
                    SELECT sa.id AS attempt_id, sa.submission_id, er.id AS eval_run_id
                    FROM stage_attempts sa
                    JOIN model_submissions ms ON ms.id = sa.submission_id
                    LEFT JOIN eval_runs er ON er.stage_attempt_id = sa.id
                    WHERE sa.stage = 'EVAL'
                      AND sa.state = 'RUNNING'
                      AND sa.lease_expires_at < now()
                      AND ms.state = 'EVAL_RUNNING'
                    FOR UPDATE SKIP LOCKED
                    """
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """
                        UPDATE stage_attempts
                        SET state = 'ABANDONED', finished_at = now(),
                            fault_class = 'REMOTE_EVAL_FAULT',
                            fault_code = 'eval_attempt_lease_expired',
                            fault_message = 'Eval attempt lease expired before completion'
                        WHERE id = %s
                        """,
                        (row["attempt_id"],),
                    )
                    if row.get("eval_run_id"):
                        conn.execute(
                            """
                            UPDATE eval_runs
                            SET state = 'FAILED_RETRYABLE', finished_at = now(),
                                fault_class = 'REMOTE_EVAL_FAULT',
                                fault_code = 'eval_attempt_lease_expired',
                                fault_message = 'Eval attempt lease expired before completion'
                            WHERE id = %s
                            """,
                            (row["eval_run_id"],),
                        )
                    conn.execute(
                        """
                        UPDATE model_submissions
                        SET state = 'EVAL_RETRYABLE',
                            fault_class = 'REMOTE_EVAL_FAULT',
                            fault_code = 'eval_attempt_lease_expired',
                            fault_message = 'Eval attempt lease expired before completion',
                            retry_count = retry_count + 1,
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (row["submission_id"],),
                    )
                    self.record_event_inside_tx(
                        conn,
                        submission_id=row["submission_id"],
                        stage_attempt_id=row["attempt_id"],
                        event_type="eval_attempt_abandoned",
                        severity="WARN",
                        message="Eval attempt lease expired before completion",
                        data={"worker_id": worker_id, "eval_run_id": str(row.get("eval_run_id") or "")},
                    )
                return len(rows)

    def mark_eval_succeeded(
        self,
        *,
        submission_id: UUID,
        attempt_id: UUID,
        eval_run_id: UUID,
        verdict: dict[str, Any],
    ) -> None:
        next_state = "EVAL_WIN" if verdict.get("challenger_won") else "COMPLETE_LOSS"
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE eval_runs
                    SET state = 'SUCCEEDED',
                        generated_sample_count = COALESCE(%s, generated_sample_count),
                        scored_sample_count = COALESCE(%s, scored_sample_count),
                        score_challenger = %s,
                        score_king = %s,
                        win_margin = %s,
                        challenger_won = %s,
                        valid_turns = %s,
                        total_turns = %s,
                        king_vllm_errors = %s,
                        chal_vllm_errors = %s,
                        judge_errors = %s,
                        gpu_topology = %s,
                        finished_at = now()
                    WHERE id = %s
                    """,
                    (
                        verdict.get("generated_sample_count"),
                        verdict.get("scored_sample_count"),
                        verdict.get("score_challenger"),
                        verdict.get("score_king"),
                        _win_margin(verdict),
                        verdict.get("challenger_won"),
                        verdict.get("valid_turns"),
                        verdict.get("total_turns"),
                        verdict.get("king_vllm_errors", 0),
                        verdict.get("chal_vllm_errors", 0),
                        verdict.get("judge_errors", 0),
                        Jsonb(verdict.get("gpu_topology", {})),
                        eval_run_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET state = 'SUCCEEDED', finished_at = now(), result_summary = %s
                    WHERE id = %s
                    """,
                    (Jsonb(verdict), attempt_id),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = %s, updated_at = now(),
                        finished_at = CASE WHEN %s = 'COMPLETE_LOSS' THEN now() ELSE finished_at END
                    WHERE id = %s
                    """,
                    (next_state, next_state, submission_id),
                )
                self._insert_artifacts_from_verdict_inside_tx(conn, submission_id, attempt_id, verdict)
                self.record_event_inside_tx(
                    conn,
                    submission_id=submission_id,
                    stage_attempt_id=attempt_id,
                    event_type="eval_succeeded",
                    severity="INFO",
                    message=f"Eval completed with state {next_state}",
                    data=verdict,
                )

    def mark_eval_failed(
        self,
        *,
        submission_id: UUID,
        attempt_id: UUID,
        eval_run_id: UUID,
        fault_class: str,
        fault_code: str,
        fault_message: str,
        retryable: bool,
    ) -> None:
        attempt_state = "FAILED_RETRYABLE" if retryable else "FAILED_TERMINAL"
        submission_state = "EVAL_RETRYABLE" if retryable else "TERMINAL_INVALID"
        eval_state = "FAILED_RETRYABLE" if retryable else "FAILED_TERMINAL"
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE eval_runs
                    SET state = %s, fault_class = %s, fault_code = %s,
                        fault_message = %s, finished_at = now()
                    WHERE id = %s
                    """,
                    (eval_state, fault_class, fault_code, fault_message, eval_run_id),
                )
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET state = %s, finished_at = now(), fault_class = %s,
                        fault_code = %s, fault_message = %s
                    WHERE id = %s
                    """,
                    (attempt_state, fault_class, fault_code, fault_message, attempt_id),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = %s, fault_class = %s, fault_code = %s,
                        fault_message = %s, retry_count = retry_count + 1,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (submission_state, fault_class, fault_code, fault_message, submission_id),
                )
                self.record_event_inside_tx(
                    conn,
                    submission_id=submission_id,
                    stage_attempt_id=attempt_id,
                    event_type="eval_failed",
                    severity="ERROR",
                    message=fault_message,
                    data={"fault_class": fault_class, "fault_code": fault_code, "retryable": retryable},
                )

    @staticmethod
    def record_event_inside_tx(
        conn: psycopg.Connection,
        *,
        submission_id: UUID,
        stage_attempt_id: UUID | None,
        event_type: str,
        severity: str,
        message: str,
        data: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO events (
                id, submission_id, stage_attempt_id, event_type, severity, message, data
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (uuid4(), submission_id, stage_attempt_id, event_type, severity, message, Jsonb(data)),
        )


    @staticmethod
    def _insert_artifacts_from_verdict_inside_tx(
        conn: psycopg.Connection,
        submission_id: UUID,
        attempt_id: UUID,
        verdict: dict[str, Any],
    ) -> None:
        artifacts = verdict.get("artifacts")
        if not isinstance(artifacts, dict):
            return
        for artifact in artifact_records_from_verdict(
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            artifacts=artifacts,
        ):
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, submission_id, stage_attempt_id, artifact_type,
                    storage_backend, uri, bucket, object_key
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    uuid4(),
                    artifact.submission_id,
                    artifact.stage_attempt_id,
                    artifact.artifact_type,
                    artifact.storage_backend,
                    artifact.uri,
                    artifact.bucket,
                    artifact.object_key,
                ),
            )

    @staticmethod
    def _next_attempt_number(conn: psycopg.Connection, submission_id: UUID, stage: str) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
            FROM stage_attempts
            WHERE submission_id = %s AND stage = %s
            """,
            (submission_id, stage),
        ).fetchone()
        return int(row["next_attempt"])

    def _mark_retryable_inside_tx(
        self,
        conn: psycopg.Connection,
        submission_id: UUID,
        fault_class: str,
        fault_code: str,
        fault_message: str,
    ) -> None:
        conn.execute(
            """
            UPDATE model_submissions
            SET state = 'EVAL_RETRYABLE', fault_class = %s, fault_code = %s,
                fault_message = %s, retry_count = retry_count + 1, updated_at = now()
            WHERE id = %s
            """,
            (fault_class, fault_code, fault_message, submission_id),
        )


def _win_margin(verdict: dict[str, Any]) -> float | None:
    challenger = verdict.get("score_challenger")
    king = verdict.get("score_king")
    if challenger is None or king is None:
        return None
    return float(challenger) - float(king)
