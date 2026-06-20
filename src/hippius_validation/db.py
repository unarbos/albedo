"""Async Postgres layer for the Hippius validation stage."""

from __future__ import annotations

import json

import asyncpg

_VALIDATED_OR_BEYOND = (
    "HIPPIUS_VALIDATED",
    "PRE_EVAL_QUEUED",
    "PRE_EVAL_RUNNING",
    "PRE_EVAL_PASSED",
    "EVAL_QUEUED",
    "EVAL_RUNNING",
    "EVAL_WIN",
    "SET_REIGN_RUNNING",
    "REIGN_SET",
    "WEIGHT_SET_RUNNING",
    "COMPLETE_LOSS",
    "COMPLETE_CORONATED",
)


async def connect(db_url: str) -> asyncpg.Pool:
    if not db_url:
        raise RuntimeError("no DB url - set ALBEDO_POSTGRES_* in .env")
    return await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=4)


async def enqueue_from_commits(pool: asyncpg.Pool, netuid: int) -> int:
    """Repair legacy chain_commits that do not yet have model_submissions."""
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            """
            WITH upserted_miners AS (
                INSERT INTO miners (hotkey, uid, netuid, updated_at)
                SELECT DISTINCT cc.hotkey, cc.uid, cc.netuid, now()
                FROM chain_commits cc
                WHERE cc.netuid = $1
                  AND cc.submission_id IS NULL
                ON CONFLICT (hotkey) DO UPDATE SET
                    uid = EXCLUDED.uid,
                    netuid = EXCLUDED.netuid,
                    updated_at = now()
                RETURNING id, hotkey
            ),
            inserted AS (
                INSERT INTO model_submissions (
                    miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,
                    commit_hash, state, idempotency_key
                )
                SELECT m.id, cc.id, cc.netuid, cc.uid, cc.hotkey, cc.model_uri,
                       cc.commit_payload->>'digest', 'SUBMITTED',
                       'chain:' || cc.netuid || ':' || cc.hotkey || ':' || cc.payload_hash
                FROM chain_commits cc
                JOIN miners m ON m.hotkey = cc.hotkey
                WHERE cc.netuid = $1
                  AND cc.submission_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM model_submissions ms WHERE ms.chain_commit_id = cc.id
                  )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    miner_id = EXCLUDED.miner_id,
                    chain_commit_id = EXCLUDED.chain_commit_id,
                    uid = EXCLUDED.uid,
                    model_uri = EXCLUDED.model_uri,
                    updated_at = now()
                RETURNING id, chain_commit_id
            ),
            linked AS (
                UPDATE chain_commits cc
                SET submission_id = i.id
                FROM inserted i
                WHERE cc.id = i.chain_commit_id
                  AND cc.submission_id IS NULL
                RETURNING 1
            )
            SELECT count(*) FROM linked
            """,
            netuid,
        )
    return int(row or 0)


async def claim_next(
    pool: asyncpg.Pool, worker_id: str, lease_seconds: int
) -> asyncpg.Record | None:
    """Claim the oldest submitted or retryable model for Hippius validation."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            submission = await conn.fetchrow(
                """
                SELECT ms.id AS submission_id, ms.chain_commit_id, ms.hotkey,
                       ms.model_uri, ms.commit_hash, ms.retry_count, ms.priority,
                       cc.block_number, cc.payload_hash
                FROM model_submissions ms
                JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                WHERE ms.state IN ('SUBMITTED', 'HIPPIUS_RETRYABLE')
                ORDER BY cc.block_number ASC, ms.priority ASC, ms.created_at ASC
                FOR UPDATE OF ms SKIP LOCKED
                LIMIT 1
                """
            )
            if submission is None:
                return None

            attempt_number = await conn.fetchval(
                """
                SELECT COALESCE(MAX(attempt_number), 0) + 1
                FROM stage_attempts
                WHERE submission_id = $1 AND stage = 'HIPPIUS'
                """,
                submission["submission_id"],
            )
            attempt = await conn.fetchrow(
                """
                INSERT INTO stage_attempts (
                    submission_id, stage, attempt_number, state, worker_id,
                    lease_expires_at, started_at, input_snapshot
                )
                VALUES (
                    $1, 'HIPPIUS', $2, 'RUNNING', $3,
                    now() + ($4 * interval '1 second'), now(), $5::jsonb
                )
                RETURNING id, attempt_number
                """,
                submission["submission_id"],
                attempt_number,
                worker_id,
                lease_seconds,
                json.dumps(
                    {
                        "chain_commit_id": str(submission["chain_commit_id"]),
                        "model_uri": submission["model_uri"],
                        "hotkey": submission["hotkey"],
                        "block_number": submission["block_number"],
                    }
                ),
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'HIPPIUS_RUNNING', fault_class = NULL,
                    fault_code = NULL, fault_message = NULL, updated_at = now()
                WHERE id = $1
                """,
                submission["submission_id"],
            )
            await _record_event(
                conn,
                submission["submission_id"],
                attempt["id"],
                "hippius_claimed",
                "INFO",
                f"Hippius validation claimed by {worker_id}",
                {"worker_id": worker_id},
            )
            return {
                **dict(submission),
                "id": attempt["id"],
                "attempt_number": attempt["attempt_number"],
            }


async def heartbeat(pool: asyncpg.Pool, attempt_id, lease_seconds: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE stage_attempts
            SET lease_expires_at = now() + ($2 * interval '1 second')
            WHERE id = $1 AND stage = 'HIPPIUS' AND state = 'RUNNING'
            """,
            attempt_id,
            lease_seconds,
        )


async def mark_done(pool: asyncpg.Pool, attempt_id, result_summary: dict) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            model_hash = result_summary.get("model_hash") or row["model_hash"] or row["commit_hash"]
            manifest_uri = _model_manifest_uri(row["model_uri"])
            await _record_model_manifest_artifact(conn, row, attempt_id, model_hash, manifest_uri)
            result_summary = {**result_summary, "model_manifest_uri": manifest_uri}
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'SUCCEEDED', finished_at = now(), lease_expires_at = NULL,
                    result_summary = $2::jsonb
                WHERE id = $1
                """,
                attempt_id,
                json.dumps(result_summary),
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'HIPPIUS_VALIDATED',
                    model_hash = COALESCE(model_hash, $2),
                    updated_at = now(),
                    fault_class = NULL,
                    fault_code = NULL,
                    fault_message = NULL
                WHERE id = $1
                """,
                row["submission_id"],
                model_hash,
            )
            await _record_event(
                conn,
                row["submission_id"],
                attempt_id,
                "hippius_succeeded",
                "INFO",
                "Hippius validation succeeded",
                result_summary,
            )


async def mark_failed(
    pool: asyncpg.Pool,
    attempt_id,
    *,
    fault_class: str,
    fault_code: str,
    fault_message: str,
    result_summary: dict,
) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'FAILED_TERMINAL', finished_at = now(), lease_expires_at = NULL,
                    fault_class = $2, fault_code = $3, fault_message = $4,
                    result_summary = $5::jsonb
                WHERE id = $1
                """,
                attempt_id,
                fault_class,
                fault_code,
                fault_message,
                json.dumps(result_summary),
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'TERMINAL_INVALID', fault_class = $2,
                    fault_code = $3, fault_message = $4, updated_at = now(), finished_at = now()
                WHERE id = $1
                """,
                row["submission_id"],
                fault_class,
                fault_code,
                fault_message,
            )
            await _record_event(
                conn,
                row["submission_id"],
                attempt_id,
                "hippius_failed_terminal",
                "WARN",
                fault_message,
                {"fault_class": fault_class, "fault_code": fault_code, **result_summary},
            )


async def mark_retry(
    pool: asyncpg.Pool,
    attempt_id,
    *,
    attempt_number: int,
    max_attempts: int,
    fault_class: str,
    fault_code: str,
    fault_message: str,
) -> str:
    """Infra fault: persist HIPPIUS_RETRYABLE if under cap, else terminal infra failure."""
    terminal = attempt_number >= max_attempts
    attempt_state = "FAILED_TERMINAL" if terminal else "FAILED_RETRYABLE"
    visible_state = "TERMINAL_INFRA_FAILED" if terminal else "HIPPIUS_RETRYABLE"
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = $2, worker_id = NULL, lease_expires_at = NULL,
                    finished_at = now(), fault_class = $3, fault_code = $4,
                    fault_message = $5
                WHERE id = $1
                """,
                attempt_id,
                attempt_state,
                fault_class,
                fault_code,
                fault_message,
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = $2, fault_class = $3, fault_code = $4,
                    fault_message = $5, retry_count = retry_count + 1,
                    updated_at = now(),
                    finished_at = CASE WHEN $2 = 'TERMINAL_INFRA_FAILED' THEN now() ELSE finished_at END
                WHERE id = $1
                """,
                row["submission_id"],
                visible_state,
                fault_class,
                fault_code,
                fault_message,
            )
            await _record_event(
                conn,
                row["submission_id"],
                attempt_id,
                "hippius_retryable_failed",
                "ERROR" if terminal else "WARN",
                fault_message,
                {"fault_class": fault_class, "fault_code": fault_code, "terminal": terminal},
            )
    return "failed" if terminal else "queued"


async def sweep_expired(pool: asyncpg.Pool) -> int:
    """Return expired Hippius attempts to HIPPIUS_RETRYABLE for crash recovery."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                UPDATE stage_attempts sa
                SET state = 'ABANDONED', worker_id = NULL, lease_expires_at = NULL,
                    finished_at = now(), fault_class = 'INFRA_FAULT',
                    fault_code = 'hippius_attempt_lease_expired',
                    fault_message = 'Hippius validation lease expired before completion'
                FROM model_submissions ms
                WHERE ms.id = sa.submission_id
                  AND sa.stage = 'HIPPIUS'
                  AND sa.state = 'RUNNING'
                  AND sa.lease_expires_at < now()
                  AND ms.state = 'HIPPIUS_RUNNING'
                RETURNING sa.id, sa.submission_id
                """
            )
            for row in rows:
                await conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = 'HIPPIUS_RETRYABLE', fault_class = 'INFRA_FAULT',
                        fault_code = 'hippius_attempt_lease_expired',
                        fault_message = 'Hippius validation lease expired before completion',
                        retry_count = retry_count + 1, updated_at = now()
                    WHERE id = $1
                    """,
                    row["submission_id"],
                )
                await _record_event(
                    conn,
                    row["submission_id"],
                    row["id"],
                    "hippius_attempt_abandoned",
                    "WARN",
                    "Hippius validation lease expired before completion",
                    {},
                )
    return len(rows)


async def hotkey_validated(pool: asyncpg.Pool, hotkey: str) -> bool:
    """Has this hotkey already had a model pass Hippius validation or beyond?"""
    async with pool.acquire() as conn:
        return bool(
            await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM model_submissions
                    WHERE hotkey = $1
                      AND state = ANY($2::text[])
                )
                """,
                hotkey,
                list(_VALIDATED_OR_BEYOND),
            )
        )


async def _attempt_submission(conn: asyncpg.Connection, attempt_id) -> asyncpg.Record:
    row = await conn.fetchrow(
        """
        SELECT sa.submission_id, ms.commit_hash, ms.model_hash, ms.model_uri
        FROM stage_attempts sa
        JOIN model_submissions ms ON ms.id = sa.submission_id
        WHERE sa.id = $1 AND sa.stage = 'HIPPIUS'
        FOR UPDATE OF sa, ms
        """,
        attempt_id,
    )
    if row is None:
        raise RuntimeError(f"HIPPIUS attempt not found: {attempt_id}")
    return row


async def _record_model_manifest_artifact(
    conn: asyncpg.Connection,
    submission: asyncpg.Record,
    attempt_id,
    model_hash: str | None,
    manifest_uri: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO artifacts (
            submission_id, stage_attempt_id, artifact_type, storage_backend,
            uri, sha256, content_type
        )
        SELECT $1, $2, 'MODEL_MANIFEST', $3, $4, $5,
               'application/vnd.oci.image.manifest.v1+json'
        WHERE NOT EXISTS (
            SELECT 1
            FROM artifacts
            WHERE submission_id = $1
              AND artifact_type = 'MODEL_MANIFEST'
              AND uri = $4
        )
        """,
        submission["submission_id"],
        attempt_id,
        _storage_backend_for_model_uri(manifest_uri),
        manifest_uri,
        _manifest_sha256(model_hash),
    )


def _model_manifest_uri(model_uri: str) -> str:
    if model_uri.startswith(("s3://", "file://", "local-cache://")):
        return model_uri
    if model_uri.startswith("registry.hippius.com/"):
        return model_uri
    if "@sha256:" in model_uri:
        return f"registry.hippius.com/{model_uri}"
    return model_uri


def _storage_backend_for_model_uri(model_uri: str) -> str:
    if model_uri.startswith("s3://"):
        return "s3"
    if model_uri.startswith(("local-cache://", "file://")):
        return "local-cache"
    return "hippius"


def _manifest_sha256(model_hash: str | None) -> str | None:
    if not model_hash:
        return None
    return model_hash.removeprefix("sha256:")


async def _record_event(
    conn: asyncpg.Connection,
    submission_id,
    stage_attempt_id,
    event_type: str,
    severity: str,
    message: str,
    data: dict,
) -> None:
    await conn.execute(
        """
        INSERT INTO events (submission_id, stage_attempt_id, event_type, severity, message, data)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        submission_id,
        stage_attempt_id,
        event_type,
        severity,
        message,
        json.dumps(data),
    )
