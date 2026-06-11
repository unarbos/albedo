"""sanity_service configuration - loaded from albedo/.env + process environment."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    # Minimal .env loader - KEY=VALUE lines, # comments, no external dependency.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(_ENV_PATH)


def _db_url() -> str:
    # Postgres DSN from ALBEDO_POSTGRES_* vars; empty string disables DB features.
    user     = os.environ.get("ALBEDO_POSTGRES_USER", "")
    password = os.environ.get("ALBEDO_POSTGRES_PASSWORD", "")
    db       = os.environ.get("ALBEDO_POSTGRES_DB", "")
    host     = os.environ.get("ALBEDO_POSTGRES_HOST", "")
    port     = os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "")
    if not all((user, password, db, host, port)):
        return ""
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


# vLLM / runner
VLLM_PORT:        int   = int(os.environ.get("SANITY_VLLM_PORT", "9101"))
GPUS:             str   = os.environ.get("SANITY_GPUS", "0")
GPU_UTIL:         float = float(os.environ.get("SANITY_GPU_UTIL", "0.5"))
VLLM_DTYPE:       str   = os.environ.get("SANITY_VLLM_DTYPE", "bfloat16")
DOWNLOAD_TIMEOUT: float = float(os.environ.get("SANITY_DOWNLOAD_TIMEOUT", "300"))
VLLM_STARTUP_S:   float = float(os.environ.get("SANITY_VLLM_STARTUP_S", "180"))

# OpenRouter LLM coherence gate (skipped when OR_API_KEY is empty)
OR_API_KEY: str = os.environ.get("SANITY_OR_API_KEY", "")
OR_MODEL:   str = os.environ.get("SANITY_OR_MODEL", "deepseek/deepseek-v3.2")

# FastAPI service port
PORT: int = int(os.environ.get("SANITY_PORT", "9100"))

# Postgres result cache + audit log (optional - disabled when unset)
DB_URL: str = _db_url()
