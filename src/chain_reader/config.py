"""chain_reader configuration — loaded from albedo/.env + process environment."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments, no export."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)  # process env wins over .env


_load_dotenv(_ENV_PATH)


def _db_url() -> str:
    """Postgres DSN built strictly from the ALBEDO_POSTGRES_* env vars."""
    user = os.environ.get("ALBEDO_POSTGRES_USER", "")
    password = os.environ.get("ALBEDO_POSTGRES_PASSWORD", "")
    db = os.environ.get("ALBEDO_POSTGRES_DB", "")
    host = os.environ.get("ALBEDO_POSTGRES_HOST", "")
    port = os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "")
    if not all((user, password, db, host, port)):
        return ""
    auth = f"{quote_plus(user)}:{quote_plus(password)}"
    return f"postgresql://{auth}@{host}:{port}/{db}"


DB_URL: str = _db_url()
NETUID: int = int(os.environ.get("CHAIN_NETUID", "97"))
NETWORK: str = os.environ.get("CHAIN_NETWORK", "finney")
# How often to poll for a new block (seconds). Bittensor block time is ~12s.
POLL_INTERVAL_S: float = float(os.environ.get("CHAIN_POLL_INTERVAL_S", "2"))
# chain_reader eval filter: only commits at/after this block are eval candidates.
# Empty/unset -> 0 (no commits skipped).
START_BLOCK: int = int(os.environ.get("CHAIN_START_BLOCK", "0") or "0")
# chain_guard backfill boundary: at startup, every hotkey that committed at/before this block is
# seeded into the used_hotkeys ledger and blocked from eval. Empty/unset -> 0 (backfill disabled).
IGNORE_COMMITS_TO_BLOCK: int = int(os.environ.get("IGNORE_COMMITS_TO_BLOCK", "0") or "0")
