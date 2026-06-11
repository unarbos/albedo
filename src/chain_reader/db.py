"""Async Postgres layer for chain commits and canonical submission creation."""
from __future__ import annotations

import json

import asyncpg

from chain_reader.chain import Commit


async def connect(db_url: str) -> asyncpg.Pool:
    if not db_url:
        raise RuntimeError("no DB url - set ALBEDO_POSTGRES_* in .env")
    return await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=4)


async def insert_new_commits(pool: asyncpg.Pool, commits: list[Commit]) -> int:
    """Insert new commits and create their model_submissions.

    Returns the count of newly discovered chain commits. Existing commits are
    repaired into submissions if needed, so a partial older ingest can resume.
    """
    if not commits:
        return 0
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for c in commits:
                miner_id = await conn.fetchval(
                    """
                    INSERT INTO miners (hotkey, uid, netuid, updated_at)
                    VALUES ($1, $2, $3, now())
                    ON CONFLICT (hotkey) DO UPDATE SET
                        uid = EXCLUDED.uid,
                        netuid = EXCLUDED.netuid,
                        updated_at = now()
                    RETURNING id
                    """,
                    c.hotkey,
                    c.uid,
                    c.netuid,
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO chain_commits
                        (netuid, block_number, block_hash, extrinsic_hash, uid, hotkey,
                         commit_payload, model_uri, payload_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
                    ON CONFLICT (netuid, hotkey, payload_hash) DO UPDATE SET
                        block_number = EXCLUDED.block_number,
                        block_hash = EXCLUDED.block_hash,
                        extrinsic_hash = COALESCE(chain_commits.extrinsic_hash, EXCLUDED.extrinsic_hash),
                        uid = EXCLUDED.uid,
                        commit_payload = EXCLUDED.commit_payload,
                        model_uri = EXCLUDED.model_uri
                    RETURNING id, submission_id, (xmax = 0) AS inserted
                    """,
                    c.netuid,
                    c.block_number,
                    c.block_hash,
                    c.extrinsic_hash,
                    c.uid,
                    c.hotkey,
                    json.dumps(c.commit_payload),
                    c.model_uri,
                    c.payload_hash,
                )
                if row is not None and row["inserted"]:
                    inserted += 1
                if row is None or row["submission_id"] is not None:
                    continue

                idempotency_key = f"chain:{c.netuid}:{c.hotkey}:{c.payload_hash}"
                submission_id = await conn.fetchval(
                    """
                    INSERT INTO model_submissions (
                        miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,
                        commit_hash, state, idempotency_key
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 'SUBMITTED', $8)
                    ON CONFLICT (idempotency_key) DO UPDATE SET
                        miner_id = EXCLUDED.miner_id,
                        chain_commit_id = EXCLUDED.chain_commit_id,
                        uid = EXCLUDED.uid,
                        model_uri = EXCLUDED.model_uri,
                        updated_at = now()
                    RETURNING id
                    """,
                    miner_id,
                    row["id"],
                    c.netuid,
                    c.uid,
                    c.hotkey,
                    c.model_uri,
                    c.commit_payload.get("digest"),
                    idempotency_key,
                )
                await conn.execute(
                    "UPDATE chain_commits SET submission_id = $1 WHERE id = $2 AND submission_id IS NULL",
                    submission_id,
                    row["id"],
                )
    return inserted
