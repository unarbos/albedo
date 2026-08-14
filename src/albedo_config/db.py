from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def postgres_dsn() -> str:
    user = os.environ.get("ALBEDO_POSTGRES_USER", "")
    password = os.environ.get("ALBEDO_POSTGRES_PASSWORD", "")
    db = os.environ.get("ALBEDO_POSTGRES_DB", "")
    host = os.environ.get("ALBEDO_POSTGRES_HOST", "")
    port = os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "")
    if not all((user, password, db, host, port)):
        return ""
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"
