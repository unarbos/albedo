"""Read on-chain commitments (miner side) — reuses the validator's chain scanner."""
from __future__ import annotations

from loguru import logger

from chain_reader.chain import connect, scan_commitments


def fetch(netuid: int, network: str, hotkey: str | None = None) -> list:
    """Return v7 commits on ``netuid`` (optionally filtered to one hotkey), oldest block first."""
    logger.info(f"connecting to {network}…")
    sub = connect(network)
    logger.info(f"scanning commitments on netuid {netuid}…")
    commits = scan_commitments(sub, netuid)
    if hotkey:
        commits = [c for c in commits if c.hotkey == hotkey]
    logger.info(f"found {len(commits)} v7 commit(s)")
    return sorted(commits, key=lambda c: c.block_number)


def print_commits(netuid: int, network: str, hotkey: str | None = None) -> int:
    commits = fetch(netuid, network, hotkey)
    if not commits:
        print("no v7 commits found" + (f" for hotkey {hotkey}" if hotkey else ""))
        return 0
    for c in commits:
        print(f"block={c.block_number}  hotkey={c.hotkey[:10]}…  {c.model_uri}")
    print(f"\n{len(commits)} commit(s)")
    return len(commits)
