from __future__ import annotations

import asyncio

from loguru import logger as log

from albedo_config import get_chain_reader_settings
from chain_guard import db as guard_db
from chain_guard import scan as guard_scan
from chain_guard import swap as guard_swap
from chain_reader import chain, db

config = get_chain_reader_settings()


async def run() -> None:
    pool = await db.connect(config.DB_URL)
    subtensor = await asyncio.to_thread(chain.connect, config.NETWORK)

    log.info(
        "chain_reader started — netuid={} network={} start_block={} ignore_commits_to_block={} (startup: full scan, diff-only insert)",  # noqa: E501
        config.NETUID,
        config.NETWORK,
        config.START_BLOCK,
        config.IGNORE_COMMITS_TO_BLOCK,
    )

    if config.IGNORE_COMMITS_TO_BLOCK > 0:
        log.info(
            "chain_guard backfill — starting (ignore_commits_to_block={})",
            config.IGNORE_COMMITS_TO_BLOCK,
        )
        try:
            raw = await asyncio.to_thread(guard_scan.scan_all_raw, subtensor, config.NETUID)
            seeded = await guard_db.record_legacy(pool, raw, config.IGNORE_COMMITS_TO_BLOCK)
            log.info(
                "chain_guard backfill — finished: scanned={} seeded_blocked_hotkeys={}",
                len(raw),
                seeded,
            )
        except Exception as exc:
            log.exception(
                f"[chain-reader] startup backfill failed, hotkeys may be unblocked: {exc}"
            )
    else:
        log.info(
            "chain_guard backfill — skipped (IGNORE_COMMITS_TO_BLOCK unset/0; no hotkeys blocked)"
        )

    log.info(
        "chain_guard backfill — done for this run; entering poll loop (per-commit guard check stays active)"  # noqa: E501
    )

    last_block: int | None = None
    try:
        while True:
            try:
                cur = await asyncio.to_thread(subtensor.get_current_block)
                if cur != last_block:
                    at_block = max(cur - config.SCAN_LAG_BLOCKS, 0)
                    snapshot = await asyncio.to_thread(
                        chain.metagraph_snapshot, subtensor, config.NETUID, at_block
                    )

                    swaps = guard_swap.find_swaps(await guard_db.load_uid_state(pool), snapshot)
                    if swaps:
                        swaps = await asyncio.to_thread(
                            chain.confirm_swaps, subtensor, swaps, at_block
                        )
                    if swaps:
                        n_ledgered = await guard_db.record_swaps(pool, swaps, config.NETUID, cur)
                        log.warning(
                            "hotkey swaps detected={} newly_ledgered={}", len(swaps), n_ledgered
                        )
                    await guard_db.refresh_registration_blocks(pool, snapshot)

                    uid_map = {hotkey: uid for uid, hotkey, _reg in snapshot}
                    commits = await asyncio.to_thread(
                        chain.scan_commitments,
                        subtensor,
                        config.NETUID,
                        config.START_BLOCK,
                        uid_map,
                        at_block,
                    )
                    n_new = await db.insert_new_commits(pool, commits)
                    log.info("block={} scanned={} new={}", cur, len(commits), n_new)
                    last_block = cur
            except Exception as exc:
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
