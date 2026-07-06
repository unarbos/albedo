"""Async Postgres layer for the used_hotkeys guard ledger."""
from __future__ import annotations

import asyncpg
from loguru import logger as log


async def is_used(conn: asyncpg.Connection, hotkey: str) -> bool:
    """True if this hotkey is in the ledger (legacy or already burned by eval)."""
    return bool(await conn.fetchval("SELECT 1 FROM used_hotkeys WHERE hotkey = $1", hotkey))


async def record_legacy(pool: asyncpg.Pool, rows: list[tuple[str, int, str]], ignore_to_block: int) -> int:
    """Seed the ledger with every (hotkey, block, raw_payload) committed at/before ``ignore_to_block``.

    Idempotent: a hotkey is recorded at most once. Returns the number of rows newly inserted.
    """
    legacy = [(hk, block, raw) for hk, block, raw in rows if block <= ignore_to_block]
    if not legacy:
        return 0
    log.debug(f"[chain-guard] record_legacy starting: candidates={len(legacy)} ignore_to_block={ignore_to_block}")
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for hotkey, block, raw in legacy:
                status = await conn.execute(
                    """
                    INSERT INTO used_hotkeys (hotkey, block_number, raw_payload, source)
                    VALUES ($1, $2, $3, 'backfill')
                    ON CONFLICT (hotkey) DO NOTHING
                    """,
                    hotkey,
                    block,
                    raw,
                )
                if status.endswith("1"):
                    inserted += 1
    log.debug(f"[chain-guard] record_legacy done: inserted={inserted} of candidates={len(legacy)}")
    return inserted
