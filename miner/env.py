"""Load miner configuration from a ``.env`` file (no external dependency).

Miners copy ``.env.example_miners`` → ``.env`` and fill in their wallet / Hippius / chain
values. Real process environment variables always win (we only ``setdefault``), so exported
vars or one-off ``CHAIN_NETWORK=test albedo …`` overrides still take precedence.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # the albedo repo root


def _load_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def load() -> None:
    """Load ``.env`` from the repo root and the current working directory (root first)."""
    _load_file(_REPO_ROOT / ".env")
    cwd_env = Path.cwd() / ".env"
    if cwd_env != _REPO_ROOT / ".env":
        _load_file(cwd_env)
