"""Single source of truth for validator constants, read from chain.toml at import time."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # albedo repo root (where chain.toml lives)
_OVERRIDE = os.environ.get("CV_CHAIN_OVERRIDE", "").strip()
_TOML_PATH = Path(_OVERRIDE) if _OVERRIDE else _ROOT / "chain.toml"
if _OVERRIDE and not _TOML_PATH.is_absolute():
    _TOML_PATH = _ROOT / _TOML_PATH

with open(_TOML_PATH, "rb") as _fh:
    _T = tomllib.load(_fh)

_c = _T.get("chain", {})
_a = _T.get("arch", {})
_s = _T.get("seed", {})
_f = _T.get("files", {})
_p = _T.get("preeval", {})

# Chain identity
NAME: str = _c.get("name", "Albedo")
SEED_REPO: str = _c.get("seed_repo", "")
REPO_PATTERN: str = _c.get("repo_pattern", "")

# Network / netuid (env-overridable; defaults target SN97 mainnet)
NETWORK: str = os.environ.get("CV_NETWORK", "finney")
NETUID: int = int(os.environ.get("CV_NETUID", "97"))

# Arch lock — challenger config.json must match the seed on these keys.
COMPAT_KEYS: tuple[str, ...] = ("vocab_size", "model_type")
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_a.get("extra_lock_keys", []))
ALL_LOCK_KEYS: tuple[str, ...] = COMPAT_KEYS + EXTRA_LOCK_KEYS

# Genesis seed reference for the architecture check.
SEED_DIGEST: str = _s.get("seed_digest", "")
if SEED_DIGEST and not SEED_DIGEST.startswith("sha256:"):
    raise RuntimeError(
        f"chain.toml [seed].seed_digest must be a 'sha256:' digest; got {SEED_DIGEST[:12]!r}"
    )

# Strict file allowlist (check #2)
REQUIRED_FILES: tuple[str, ...] = tuple(_f.get("required", []))
REQUIRE_SAFETENSORS: bool = bool(_f.get("require_safetensors", True))
ALLOWED_FILES: tuple[str, ...] = tuple(_f.get("allowed", []))
ALLOWED_GLOBS: tuple[str, ...] = tuple(_f.get("allowed_globs", []))
FORBIDDEN_GLOBS: tuple[str, ...] = tuple(_f.get("forbidden_globs", []))

# Dedup (check #4)
SIM_THRESHOLD: float = float(_p.get("similarity_threshold", 0.95))

# Local cache root for model downloads.
MODEL_CACHE_DIR: str = os.environ.get(
    "CV_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "cv_models")
)

# Primary model storage backend (HF unless overridden); per-model detection still applies.
MODEL_BACKEND: str = os.environ.get("ALBEDO_MODEL_BACKEND", "hf").strip().lower()
# Xet is the live HF fast-transfer path (huggingface_hub>=1.0); set before any hub import.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
