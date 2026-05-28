"""Single source of truth for the active king chain, judge, dataset, duel.

Reads `chain.toml` at the repo root and exposes constants used by
`validator.py`, `miner.py`, `eval.py`, and the website (indirectly via
`dashboard.json`).

To swap the king to a new generation, edit `chain.toml` (and add
`archs/<new>/` if the architecture changes); no code edits should be
necessary.

Override knob: `ALBEDO_CHAIN_OVERRIDE` env var, when set, points at an
alternate TOML (relative to repo root or absolute path). Used by soak / smoke
runs so the live `chain.toml` stays untouched.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import re
import tomllib
from types import ModuleType

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_OVERRIDE = os.environ.get("ALBEDO_CHAIN_OVERRIDE", "").strip()
if _OVERRIDE:
    _candidate = pathlib.Path(_OVERRIDE)
    _TOML_PATH = _candidate if _candidate.is_absolute() else (_REPO_ROOT / _candidate)
else:
    _TOML_PATH = _REPO_ROOT / "chain.toml"

with open(_TOML_PATH, "rb") as _f:
    _doc = tomllib.load(_f)

_chain = _doc.get("chain", {})
_arch = _doc.get("arch", {})
_seed = _doc.get("seed", {})
_judge = _doc.get("judge", {})
_dataset = _doc.get("dataset", {})
_duel = _doc.get("duel", {})

_VALID_SEED_REPO_BACKENDS = {"hf", "hippius"}


def _default_seed_repo_backend(seed_digest: str) -> str:
    digest = (seed_digest or "").strip()
    if digest.startswith("sha256:"):
        return "hippius"
    if digest.startswith("hf:"):
        return "hf"
    return "hf"


NAME: str = _chain["name"]
SEED_REPO: str = _chain["seed_repo"]
REPO_PATTERN: str = _chain.get("repo_pattern") or rf"^[^/]+/{re.escape(NAME)}-.+$"

ARCH_MODULE: str = _arch.get("module", "")
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_arch.get("extra_lock_keys", []))

SEED_TOKENIZER_REPO: str = _seed.get("tokenizer_repo", "")
SEED_DIGEST: str = _seed.get("seed_digest", "")
SEED_REPO_BACKEND: str = (_seed.get("repo_backend") or _default_seed_repo_backend(SEED_DIGEST)).strip().lower()
if SEED_REPO_BACKEND not in _VALID_SEED_REPO_BACKENDS:
    raise RuntimeError(
        f"chain.toml [seed].repo_backend must be one of "
        f"{sorted(_VALID_SEED_REPO_BACKENDS)}, got {SEED_REPO_BACKEND!r}"
    )

SEED_NAMESPACE: str = SEED_REPO.split("/", 1)[0] if "/" in SEED_REPO else ""

# [judge]
# Accept either `models = [...]` (preferred, multi-judge) or legacy `model = "..."`
# (single judge). At least one judge model must be configured.
_judge_models_raw = _judge.get("models")
if _judge_models_raw is None:
    _legacy = _judge.get("model", "")
    JUDGE_MODELS: tuple[str, ...] = (_legacy,) if _legacy else ()
else:
    JUDGE_MODELS = tuple(str(m).strip() for m in _judge_models_raw if str(m).strip())
if not JUDGE_MODELS:
    raise RuntimeError(
        "chain.toml [judge] must define `models = [...]` (preferred) or `model = '...'`"
    )
# `JUDGE_MODEL` (singular) kept for short labels (e.g. brand sub-line on the
# dashboard) — first entry of the list, as a sensible "primary judge" tag.
JUDGE_MODEL: str = JUDGE_MODELS[0]
JUDGE_BASE_URL_ENV: str = _judge.get("base_url_env", "CHUTES_BASE_URL")
JUDGE_API_KEY_ENV: str = _judge.get("api_key_env", "CHUTES_API_KEY")
JUDGE_TEMPERATURE: float = float(_judge.get("temperature", 0.0))
JUDGE_MAX_TOKENS: int = int(_judge.get("max_tokens", 256))
_thinking_models_raw = _judge.get("thinking_models")
JUDGE_THINKING_MODELS: frozenset[str] = frozenset(
    str(m).strip() for m in (_thinking_models_raw or []) if str(m).strip()
)
JUDGE_THINKING_MAX_TOKENS: int = int(
    _judge.get("thinking_max_tokens", max(JUDGE_MAX_TOKENS, 4096))
)
JUDGE_RETRY_MAX: int = int(_judge.get("retry_max", 3))
JUDGE_RETRY_INITIAL_BACKOFF_S: float = float(_judge.get("retry_initial_backoff_s", 1.5))
JUDGE_TIE_BAND: float = float(_judge.get("tie_band", 0.01))
JUDGE_DEFAULT_BASE_URL = "https://llm.chutes.ai/v1"

# [dataset]
DATASET_REPO: str = _dataset.get("repo", "")
DATASET_SHARD_GLOB: str = _dataset.get("shard_glob", "data/train-*.parquet")
DATASET_MANIFEST_SHA256: str = _dataset.get("manifest_sha256", "")

# [duel]
DUEL_N_SAMPLES: int = int(_duel.get("n_samples", 32))
DUEL_MAX_TURNS_PER_SAMPLE: int = int(_duel.get("max_turns_per_sample", 10))
DUEL_ALPHA: float = float(_duel.get("alpha", 0.001))
DUEL_GATE_ALPHA: float = float(_duel.get("gate_alpha", 0.05))
DUEL_BOOTSTRAP_RESAMPLES: int = int(_duel.get("bootstrap_resamples", 10000))
DUEL_GEN_TEMPERATURE: float = float(_duel.get("gen_temperature", 1.0))
DUEL_GEN_MAX_TOKENS: int = int(_duel.get("gen_max_tokens", 1024))
DUEL_GEN_MAX_MODEL_LEN: int = int(_duel.get("gen_max_model_len", 32768))
DUEL_KING_CHAIN_DEPTH: int = int(_duel.get("king_chain_depth", 5))


def load_arch() -> ModuleType:
    """Import the configured architecture module.

    The arch package's import side effect is to register its config + model
    classes with HuggingFace `AutoConfig` / `AutoModelForCausalLM` so any
    downstream `from_pretrained` resolves the king without trust_remote_code.
    """
    if not ARCH_MODULE:
        raise RuntimeError("chain.toml is missing [arch].module")
    return importlib.import_module(ARCH_MODULE)


__all__ = [
    "NAME",
    "SEED_REPO",
    "REPO_PATTERN",
    "ARCH_MODULE",
    "EXTRA_LOCK_KEYS",
    "SEED_TOKENIZER_REPO",
    "SEED_DIGEST",
    "SEED_REPO_BACKEND",
    "SEED_NAMESPACE",
    "JUDGE_MODEL",
    "JUDGE_MODELS",
    "JUDGE_BASE_URL_ENV",
    "JUDGE_API_KEY_ENV",
    "JUDGE_TEMPERATURE",
    "JUDGE_MAX_TOKENS",
    "JUDGE_THINKING_MODELS",
    "JUDGE_THINKING_MAX_TOKENS",
    "JUDGE_RETRY_MAX",
    "JUDGE_RETRY_INITIAL_BACKOFF_S",
    "JUDGE_TIE_BAND",
    "JUDGE_DEFAULT_BASE_URL",
    "DATASET_REPO",
    "DATASET_SHARD_GLOB",
    "DATASET_MANIFEST_SHA256",
    "DUEL_N_SAMPLES",
    "DUEL_MAX_TURNS_PER_SAMPLE",
    "DUEL_ALPHA",
    "DUEL_GATE_ALPHA",
    "DUEL_BOOTSTRAP_RESAMPLES",
    "DUEL_GEN_TEMPERATURE",
    "DUEL_GEN_MAX_TOKENS",
    "DUEL_GEN_MAX_MODEL_LEN",
    "DUEL_KING_CHAIN_DEPTH",
    "load_arch",
]
