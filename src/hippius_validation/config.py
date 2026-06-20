"""hippius_validation configuration — loaded from albedo/.env + process environment."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

_ROOT = Path(__file__).resolve().parents[2]          # albedo repo root
_PKG = Path(__file__).resolve().parent                  # hippius_validation/
_ENV_PATH = _ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments, no export."""
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
    """Postgres DSN built strictly from the ALBEDO_POSTGRES_* env vars."""
    user = os.environ.get("ALBEDO_POSTGRES_USER", "")
    password = os.environ.get("ALBEDO_POSTGRES_PASSWORD", "")
    db = os.environ.get("ALBEDO_POSTGRES_DB", "")
    host = os.environ.get("ALBEDO_POSTGRES_HOST", "")
    port = os.environ.get("ALBEDO_POSTGRES_HOST_PORT", "")
    if not all((user, password, db, host, port)):
        return ""
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


# --- Postgres / chain scope ---
DB_URL: str = _db_url()
NETUID: int = int(os.environ.get("CHAIN_NETUID", "97"))

# --- Hippius model download cache (where downloaded models land) ---
# Empty/unset ALBEDO_MODEL_CACHE_DIR falls back to ~/.cache/albedo_models (resolved at runtime).
MODEL_CACHE_DIR: str = os.environ.get("ALBEDO_MODEL_CACHE_DIR") or str(Path.home() / ".cache" / "albedo_models")
# The shared downloader (config_validation) reads CV_MODEL_CACHE_DIR — point it at ours.
os.environ["CV_MODEL_CACHE_DIR"] = MODEL_CACHE_DIR

# --- OpenSearch (dedup corpus) ---
OPENSEARCH_URL: str = os.environ.get("ALBEDO_OPENSEARCH_URL", "http://127.0.0.1:9200")
OPENSEARCH_USER: str = os.environ.get("ALBEDO_OPENSEARCH_USER", "")
OPENSEARCH_PASSWORD: str = os.environ.get("ALBEDO_OPENSEARCH_PASSWORD", "")
OPENSEARCH_INDEX: str = os.environ.get("ALBEDO_OPENSEARCH_INDEX", "albedo_fingerprints")

# --- Hippius S3 (artifact publishing) ---
S3_BUCKET: str = os.environ.get("ALBEDO_S3_BUCKET", "")
S3_ENDPOINT: str = os.environ.get("ALBEDO_S3_ENDPOINT", "https://s3.hippius.com")
S3_ACCESS_KEY: str = os.environ.get("ALBEDO_S3_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.environ.get("ALBEDO_S3_SECRET_KEY", "")
# The two aggregate fingerprint files in the bucket, updated for every model.
FP_FILE: str = os.environ.get("ALBEDO_FP_FILE", "fingerprint.json")
TENSORS_FILE: str = os.environ.get("ALBEDO_TENSORS_FILE", "tensors.json")

# --- Architecture spec (universal, file-driven check) ---
ARCH_SPEC_PATH: str = os.environ.get("ALBEDO_ARCH_SPEC", str(_PKG / "validate" / "architecture_spec.json"))

# --- Dedup threshold + kNN fan-out ---
SIM_THRESHOLD: float = float(os.environ.get("ALBEDO_SIM_THRESHOLD", "0.95"))
KNN_CANDIDATES: int = int(os.environ.get("ALBEDO_KNN_CANDIDATES", "20"))

# --- Strict file allowlist (file-manifest check) ---
REQUIRED_FILES: tuple[str, ...] = ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json")
REQUIRE_SAFETENSORS: bool = True
ALLOWED_FILES: tuple[str, ...] = (
    "generation_config.json", "special_tokens_map.json", "added_tokens.json",
    # chat_template.jinja may be present, but the system uses its OWN canonical template
    "chat_template.jinja", "merges.txt", "vocab.json",
    "model.safetensors.index.json",
    ".gitattributes", "LICENSE", "README.md", "configuration.json",
)
ALLOWED_GLOBS: tuple[str, ...] = ("model-*-of-*.safetensors", "model.safetensors")
FORBIDDEN_GLOBS: tuple[str, ...] = ("*.py",)

# --- Worker timing ---
POLL_INTERVAL_S: float = float(os.environ.get("ALBEDO_HV_POLL_S", "5"))
LEASE_SECONDS: int = int(os.environ.get("ALBEDO_HV_LEASE_S", "600"))
HEARTBEAT_S: float = float(os.environ.get("ALBEDO_HV_HEARTBEAT_S", "30"))
MAX_ATTEMPTS: int = int(os.environ.get("ALBEDO_HV_MAX_ATTEMPTS", "5"))
