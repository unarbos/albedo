"""Raw, all-versions chain scan — every revealed commitment, unparsed.

Where chain_reader.chain.scan_commitments narrows to well-formed v5 reveals, chain_guard needs
*everything* (v5, v4, v3, …) to seed the legacy ledger. It reuses chain_reader's revealed-commit
iterator and just keeps the raw payload string per (hotkey, block).
"""
from __future__ import annotations

from typing import Any

from chain_reader.chain import _iter_revealed


def scan_all_raw(subtensor: Any, netuid: int) -> list[tuple[str, int, str]]:
    """Return every revealed commitment on ``netuid`` as (hotkey, block, raw_payload)."""
    return list(_iter_revealed(subtensor, netuid))
