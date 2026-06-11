"""Async Postgres layer (asyncpg) for the Hippius validation stage.

The hippius_stage_attempts and success_validated tables are managed externally (like
chain_commits) — this service does not create them.
"""
from __future__ import annotations

import json

import asyncpg


async def connect(db_url: str) -> asyncpg.Pool:
    if not db_url:
        raise RuntimeError("no DB url — set ALBEDO_POSTGRES_* in .env")
    return await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=4)


async def enqueue_from_commits(pool: asyncpg.Pool, netuid: int) -> int:
    """Insert queued attempts for commits not yet queued, skipping already-evaluated hotkeys."""
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            """
            WITH inserted AS (
                INSERT INTO hippius_stage_attempts
                    (chain_commit_id, hotkey, model_uri, block_number, state)
                SELECT cc.id, cc.hotkey, cc.model_uri, cc.block_number, 'queued'
                FROM chain_commits cc
                WHERE cc.netuid = $1
                  AND cc.hotkey NOT IN (SELECT hotkey FROM success_validated)
                ON CONFLICT (chain_commit_id) DO NOTHING
                RETURNING 1
            )
            SELECT count(*) FROM inserted
            """,
            netuid,
        )
    return int(row or 0)


async def claim_next(pool: asyncpg.Pool, worker_id: str, lease_seconds: int) -> asyncpg.Record | None:
    """Claim the oldest queued attempt (oldest block first). Returns the row or None."""
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            UPDATE hippius_stage_attempts SET
                state = 'running',
                worker_id = $1,
                attempt_number = attempt_number + 1,
                started_at = now(),
                lease_expires_at = now() + ($2 * interval '1 second')
            WHERE id = (
                SELECT id FROM hippius_stage_attempts
                WHERE state = 'queued'
                ORDER BY block_number ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            worker_id, lease_seconds,
        )


async def heartbeat(pool: asyncpg.Pool, attempt_id, lease_seconds: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE hippius_stage_attempts SET lease_expires_at = now() + ($2 * interval '1 second') "
            "WHERE id = $1 AND state = 'running'",
            attempt_id, lease_seconds,
        )


async def mark_done(pool: asyncpg.Pool, attempt_id, result_summary: dict) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE hippius_stage_attempts SET state='done', finished_at=now(), "
            "lease_expires_at=NULL, result_summary=$2::jsonb WHERE id=$1",
            attempt_id, json.dumps(result_summary),
        )


async def mark_failed(pool: asyncpg.Pool, attempt_id, *, fault_class: str, fault_code: str,
                      fault_message: str, result_summary: dict) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE hippius_stage_attempts SET state='failed', finished_at=now(), "
            "lease_expires_at=NULL, fault_class=$2, fault_code=$3, fault_message=$4, "
            "result_summary=$5::jsonb WHERE id=$1",
            attempt_id, fault_class, fault_code, fault_message, json.dumps(result_summary),
        )


async def mark_retry(pool: asyncpg.Pool, attempt_id, *, attempt_number: int, max_attempts: int,
                     fault_class: str, fault_code: str, fault_message: str) -> str:
    """Infra fault: re-queue if under the attempt cap, else fail terminally. Returns new state."""
    terminal = attempt_number >= max_attempts
    new_state = "failed" if terminal else "queued"
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE hippius_stage_attempts SET state=$2, worker_id=NULL, lease_expires_at=NULL, "
            "finished_at = CASE WHEN $2='failed' THEN now() ELSE NULL END, "
            "fault_class=$3, fault_code=$4, fault_message=$5 WHERE id=$1",
            attempt_id, new_state, fault_class, fault_code, fault_message,
        )
    return new_state


async def sweep_expired(pool: asyncpg.Pool) -> int:
    """Return expired 'running' attempts to 'queued' (crash recovery)."""
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            """
            WITH swept AS (
                UPDATE hippius_stage_attempts SET state='queued', worker_id=NULL, lease_expires_at=NULL
                WHERE state='running' AND lease_expires_at < now()
                RETURNING 1
            )
            SELECT count(*) FROM swept
            """
        )
    return int(n or 0)


async def hotkey_validated(pool: asyncpg.Pool, hotkey: str) -> bool:
    """READ-ONLY: has this hotkey already had its one evaluation (success_validated)?"""
    async with pool.acquire() as conn:
        return bool(await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM success_validated WHERE hotkey=$1)", hotkey,
        ))
