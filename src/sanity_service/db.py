"""Postgres result cache and audit log for sanity_service."""
from __future__ import annotations

import json
from datetime import datetime

import asyncpg
from loguru import logger

_pool: asyncpg.Pool | None = None

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS sanity_results (
        id          BIGSERIAL    PRIMARY KEY,
        repo        TEXT         NOT NULL,
        digest      TEXT         NOT NULL UNIQUE,
        passed      BOOLEAN      NOT NULL,
        reason      TEXT         NOT NULL DEFAULT '',
        responses   JSONB        NOT NULL DEFAULT '[]'::jsonb,
        timing      JSONB        NOT NULL DEFAULT '{}'::jsonb,
        checked_at  TIMESTAMPTZ  NOT NULL
    );
    CREATE INDEX IF NOT EXISTS sanity_results_passed_checked_idx
        ON sanity_results (passed, checked_at DESC);
"""


async def init(db_url: str) -> None:
    # Connect and ensure the sanity_results table exists; no-op if db_url is empty.
    global _pool
    if not db_url:
        logger.info("[sanity/db] no DB URL configured - result cache disabled")
        return
    _pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=4)
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)
    logger.info("[sanity/db] connected and table ready")


def is_connected() -> bool:
    # True if the pool is open and ready.
    return _pool is not None


async def close() -> None:
    # Close the pool on service shutdown.
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_cached(digest: str) -> dict | None:
    # Return the stored result for this digest, or None if not seen before.
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT repo, digest, passed, reason, responses, timing, checked_at "
            "FROM sanity_results WHERE digest = $1",
            digest,
        )
    if not row:
        return None
    return {
        "passed":       row["passed"],
        "reason":       row["reason"],
        "responses":    json.loads(row["responses"]),
        "timing":       json.loads(row["timing"]),
        "model_repo":   row["repo"],
        "model_digest": row["digest"],
        "checked_at":   row["checked_at"].isoformat(),
        "infra_fault":  False,
        "llm_gate":     "cached",
        "cached":       True,
    }


async def insert_result(result) -> None:
    # Persist a SanityResult; silently skips if the digest is already stored.
    if not _pool:
        return
    timing = {
        "total_s":      result.timing.total_s,
        "download_s":   result.timing.download_s,
        "vllm_s":       result.timing.vllm_s,
        "prompts_s":    result.timing.prompts_s,
        "model_cached": result.timing.model_cached,
        "vllm_reused":  result.timing.vllm_reused,
    }
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sanity_results
                (repo, digest, passed, reason, responses, timing, checked_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
            ON CONFLICT (digest) DO NOTHING
            """,
            result.model_repo,
            result.model_digest,
            result.passed,
            result.reason,
            json.dumps(result.responses),
            json.dumps(timing),
            datetime.fromisoformat(result.checked_at),
        )
