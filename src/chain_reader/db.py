"""Async Postgres layer (asyncpg) — diff-only commit inserts.

The chain_commits table is managed externally (not created by this service).
"""
from __future__ import annotations

import json

import asyncpg

from chain_reader.chain import Commit

_INSERT = """
INSERT INTO chain_commits
    (netuid, block_number, block_hash, extrinsic_hash, uid, hotkey,
     commit_payload, model_uri, payload_hash)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
ON CONFLICT (netuid, hotkey, payload_hash) DO NOTHING
RETURNING id
"""


async def connect(db_url: str) -> asyncpg.Pool:
    if not db_url:
        raise RuntimeError("no DB url — set ALBEDO_POSTGRES_* in .env")
    return await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=4)


async def insert_new_commits(pool: asyncpg.Pool, commits: list[Commit]) -> int:
    """Insert only commits not already stored (the diff). Returns the count inserted."""
    if not commits:
        return 0
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for c in commits:
                row = await conn.fetchrow(
                    _INSERT,
                    c.netuid, c.block_number, c.block_hash, c.extrinsic_hash,
                    c.uid, c.hotkey, json.dumps(c.commit_payload),
                    c.model_uri, c.payload_hash,
                )
                if row is not None:
                    inserted += 1
    return inserted
