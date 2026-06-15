"""Block-paced loop: scan the chain each new block and persist the diff to Postgres."""
from __future__ import annotations

import asyncio

from loguru import logger as log

from chain_reader import chain, config, db
from chain_guard import db as guard_db, scan as guard_scan


async def run() -> None:
    pool = await db.connect(config.DB_URL)
    subtensor = await asyncio.to_thread(chain.connect, config.NETWORK)

    log.info("chain_reader started — netuid={} network={} start_block={} ignore_commits_to_block={} (startup: full scan, diff-only insert)",
             config.NETUID, config.NETWORK, config.START_BLOCK, config.IGNORE_COMMITS_TO_BLOCK)

    # chain_guard startup backfill: seed used_hotkeys with every hotkey that committed at/before
    # IGNORE_COMMITS_TO_BLOCK, so those hotkeys are blocked from eval. Idempotent across restarts.
    if config.IGNORE_COMMITS_TO_BLOCK > 0:
        log.info("chain_guard backfill — starting (ignore_commits_to_block={})", config.IGNORE_COMMITS_TO_BLOCK)
        raw = await asyncio.to_thread(guard_scan.scan_all_raw, subtensor, config.NETUID)
        seeded = await guard_db.record_legacy(pool, raw, config.IGNORE_COMMITS_TO_BLOCK)
        log.info("chain_guard backfill — finished: scanned={} seeded_blocked_hotkeys={}", len(raw), seeded)
    else:
        log.info("chain_guard backfill — skipped (IGNORE_COMMITS_TO_BLOCK unset/0; no hotkeys blocked)")

    log.info("chain_guard backfill — done for this run; entering poll loop (per-commit guard check stays active)")

    last_block: int | None = None
    try:
        while True:
            try:
                cur = await asyncio.to_thread(subtensor.get_current_block)
                if cur != last_block:
                    commits = await asyncio.to_thread(chain.scan_commitments, subtensor, config.NETUID, config.START_BLOCK)
                    n_new = await db.insert_new_commits(pool, commits)
                    log.info("block={} scanned={} new={}", cur, len(commits), n_new)
                    last_block = cur
            except Exception as exc:  # noqa: BLE001 — keep the loop alive across RPC/DB blips
                log.opt(exception=True).warning("tick failed ({}) — retrying", exc)
            await asyncio.sleep(config.POLL_INTERVAL_S)
    finally:
        await pool.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("chain_reader stopped")


if __name__ == "__main__":
    main()
