"""Block-paced loop: scan the chain each new block and persist the diff to Postgres."""
from __future__ import annotations

import asyncio

from loguru import logger as log

from chain_reader import chain, config, db


async def run() -> None:
    pool = await db.connect(config.DB_URL)
    subtensor = await asyncio.to_thread(chain.connect, config.NETWORK)

    log.info("chain_reader started — netuid={} network={} (startup: full scan, diff-only insert)",
             config.NETUID, config.NETWORK)

    last_block: int | None = None
    try:
        while True:
            try:
                cur = await asyncio.to_thread(subtensor.get_current_block)
                if cur != last_block:
                    commits = await asyncio.to_thread(chain.scan_commitments, subtensor, config.NETUID)
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
