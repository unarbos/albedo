"""Single source of truth for all subnet constants, read from chain.toml at import time."""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent.parent          # repo root
# ALBEDO_CHAIN_OVERRIDE points at an alternate TOML (absolute or repo-relative)
# for soak/smoke runs without touching the live chain.toml.
_OVERRIDE = os.environ.get("ALBEDO_CHAIN_OVERRIDE", "").strip()
_TOML_PATH = Path(_OVERRIDE) if _OVERRIDE else _ROOT / "chain.toml"
if _OVERRIDE and not _TOML_PATH.is_absolute():
    _TOML_PATH = _ROOT / _TOML_PATH
_T    = tomllib.load(open(_TOML_PATH, "rb"))

_c = _T.get("chain",   {})
_a = _T.get("arch",    {})
_s = _T.get("seed",    {})
_j = _T.get("judge",   {})
_d = _T.get("dataset", {})
_u = _T.get("duel",    {})
_p = _T.get("preeval", {})

# Chain identity
NAME:           str = _c.get("name", "Albedo")
SEED_REPO:      str = _c.get("seed_repo", "")
SEED_NAMESPACE: str = SEED_REPO.split("/")[0] if "/" in SEED_REPO else ""
REPO_PATTERN:   str = _c.get("repo_pattern", rf"^[^/]+/{re.escape(NAME)}-.+$")
DISPLAY_START_BLOCK: int = int(
    os.environ.get("ALBEDO_DISPLAY_START_BLOCK") or _c.get("display_start_block", 0)
)
# Arch lock
ARCH_MODULE:      str            = _a.get("module", "")
EXTRA_LOCK_KEYS:  tuple[str,...] = tuple(_a.get("extra_lock_keys", []))

COMPAT_KEYS:      tuple[str,...] = ("vocab_size", "model_type")
ALL_LOCK_KEYS:    tuple[str,...] = COMPAT_KEYS + EXTRA_LOCK_KEYS

# Seed / genesis. The subnet is Hippius-only: the genesis must be seeded to
# Hippius (sha256: digest) via scripts/seed_genesis.py before mining can start.
SEED_DIGEST:         str = _s.get("seed_digest", "")
SEED_TOKENIZER_REPO: str = _s.get("tokenizer_repo", SEED_REPO)
if SEED_DIGEST and not SEED_DIGEST.startswith("sha256:"):
    raise RuntimeError(
        f"chain.toml [seed].seed_digest must be a Hippius 'sha256:' digest; "
        f"got {SEED_DIGEST[:12]!r}"
    )

# Judge
JUDGE_MODELS:          list[str] = _j.get("models", [])
JUDGE_MODEL:           str       = JUDGE_MODELS[0] if JUDGE_MODELS else ""
JUDGE_BASE_URL_ENV:    str       = _j.get("base_url_env", "CHUTES_BASE_URL")
JUDGE_API_KEY_ENV:     str       = _j.get("api_key_env",  "CHUTES_API_KEY")
JUDGE_TEMPERATURE:     float     = float(_j.get("temperature", 0.0))
JUDGE_MAX_TOKENS:      int       = int(_j.get("max_tokens", 256))
# Score path (5-key JSON after a possible thinking preamble) needs more room than probe.
JUDGE_SCORE_MAX_TOKENS: int      = int(_j.get("score_max_tokens", 768))
JUDGE_THINKING_TOKENS: int       = int(_j.get("thinking_max_tokens", 4096))
JUDGE_THINKING_MODELS: list[str] = _j.get("thinking_models", [])
# The five dimensions judged head-to-head per turn (S4 pairwise scoring), canonical order.
JUDGE_METRIC_KEYS: tuple[str, ...] = tuple(_j.get(
    "metric_keys", ("correctness", "grounding", "progress", "protocol", "efficiency")))
JUDGE_RETRY_MAX:        int   = int(_j.get("retry_max", 3))
JUDGE_RETRY_BACKOFF:    float = float(_j.get("retry_initial_backoff_s", 1.5))
JUDGE_TIE_BAND:         float = float(_j.get("tie_band", 0.01))
# Hard ceiling on total time spent retrying one judge call (all attempts combined).
# After this deadline, the call returns a parse_failure verdict (score=0.0) so
# the duel can continue rather than blocking indefinitely on rate limits.
# `retry_timeout_s` is the live-prod chain.toml name for the same knob; accept both.
JUDGE_CALL_TIMEOUT_S:   float = float(_j.get("call_timeout_s", _j.get("retry_timeout_s", 300.0)))
# Max 429 wait per retry — prevents the 4× backoff from growing to 384 s/attempt.
# `retry_max_backoff_s` is the live-prod name for the same knob; accept both.
JUDGE_429_MAX_WAIT_S:   float = float(_j.get("max_429_wait_s", _j.get("retry_max_backoff_s", 60.0)))

# Per-model request shaping (reduce Chutes 429s). Each judge model gets its own
# in-flight concurrency cap + minimum spacing between calls; env overrides win.
JUDGE_MAX_CONCURRENCY_PER_MODEL: int = int(
    os.environ.get("ALBEDO_JUDGE_MODEL_MAX_CONCURRENCY", _j.get("max_concurrency_per_model", 3)))
JUDGE_MIN_INTERVAL_S_PER_MODEL: float = float(
    os.environ.get("ALBEDO_JUDGE_MODEL_MIN_INTERVAL_S", _j.get("min_interval_s_per_model", 0.0)))
# Per-model overrides: {model: {max_concurrency, min_interval_s}}.
JUDGE_RATE_LIMITS: dict = _j.get("rate_limits", {})
# Bounded 429 retries on a single call before giving up (was effectively infinite).
JUDGE_429_MAX_RETRIES: int = int(_j.get("max_429_retries", 8))

# Chutes -> OpenRouter fallback. On a Chutes 429 (beyond a short grace), re-issue the
# same call to OpenRouter instead of waiting out the rate limit.
JUDGE_FALLBACK_ENABLED: bool = (
    os.environ.get("ALBEDO_JUDGE_FALLBACK", str(int(bool(_j.get("fallback", {}).get("enabled", True)))))
    .lower() not in ("0", "false", "no", "")
)
_jf = _j.get("fallback", {})
JUDGE_FALLBACK_BASE_URL:    str   = _jf.get("base_url", "https://openrouter.ai/api")
JUDGE_FALLBACK_API_KEY_ENV: str   = _jf.get("api_key_env", "OPENROUTER_API_KEY")
JUDGE_CHUTES_429_GRACE_S:   float = float(_jf.get("chutes_429_grace_s", 4.0))
# Chutes judge id -> OpenRouter model id.
JUDGE_FALLBACK_MODEL_MAP: dict = _jf.get("model_map", {})
# Models that should request reasoning on OpenRouter (reasoning:{enabled:true}).
JUDGE_FALLBACK_REASONING_MODELS: list[str] = _jf.get("reasoning_models", list(JUDGE_THINKING_MODELS))

# --- Unified transport (v2): stream-gated Chutes -> batched OpenRouter ---
# Accept/reject gate: max wait for the FIRST streamed Chutes chunk before deciding
# Chutes isn't taking the request (then -> OpenRouter). 2s catches a fast 429/reject.
JUDGE_CHUTES_TRY_S:  float = float(_j.get("chutes_try_s", 2.0))
# Max wait for an ACCEPTED Chutes stream to finish generating (httpx read cap).
JUDGE_CHUTES_MAX_S:  float = float(_j.get("chutes_max_s", 150.0))
# OpenRouter per-attempt timeout and bounded retries (network/429/parse-fail).
JUDGE_OR_TIMEOUT_S:  float = float(_j.get("or_timeout_s", 150.0))
JUDGE_OR_RETRIES:    int   = int(_j.get("or_retries", 1))
# Per-judge ceiling for the OpenRouter phase (Chutes is capped separately above);
# once exceeded the judge is left unscored (caller treats as parse_failure/untested).
JUDGE_TOTAL_S:       float = float(_j.get("judge_total_s", 330.0))
# Circuit-breaker: after this many consecutive tasks with NO Chutes success, an eval's
# ChutesJudge instance stops trying Chutes (OpenRouter-only) for the rest of that eval.
JUDGE_CHUTES_GIVEUP_TASKS: int = int(_j.get("chutes_giveup_tasks", 5))

# Dataset
DATASET_REPO:            str = _d.get("repo", "")
DATASET_SHARD_GLOB:      str = _d.get("shard_glob", "data/train-*.parquet")
DATASET_MANIFEST_SHA256: str = _d.get("manifest_sha256", "")

# Duel
DUEL_N_SAMPLES:       int   = int(_u.get("n_samples", 64))
DUEL_MAX_TURNS:       int   = int(_u.get("max_turns_per_sample", 10))
DUEL_ALPHA:           float = float(_u.get("alpha", 0.001))
DUEL_GATE_ALPHA:      float = float(_u.get("gate_alpha", 0.05))
DUEL_RESAMPLES:       int   = int(_u.get("bootstrap_resamples", 10_000))
DUEL_GEN_TEMP:        float = float(_u.get("gen_temperature", 1.0))
DUEL_GEN_MAX_TOKENS:  int   = int(_u.get("gen_max_tokens", 1024))
DUEL_GEN_MAX_LEN:     int   = int(_u.get("gen_max_model_len", 32768))
# Challenger must beat king by at least this many points on the 0–100 scale.
DUEL_WIN_MARGIN:      float = float(_u.get("win_margin", 1.0))
DUEL_KING_CHAIN_DEPTH: int  = int(_u.get("king_chain_depth", 5))
# Whole-duel SOFT budget: once exceeded, stop launching NEW turns, await in-flight,
# and finalize the verdict from completed turns (graceful partial; min_valid_turn_frac
# still applies). Sits just below the validator hard-timeout. env override wins.
DUEL_BUDGET_S: float = float(os.environ.get("ALBEDO_DUEL_BUDGET_S", _u.get("duel_budget_s", 11000.0)))
# Fraction of completed turns that must parse for a duel to count — guards against
# crowning on a lucky handful of turns when most failed (unfair comparison).
DUEL_MIN_VALID_TURN_FRAC: float = float(_u.get("min_valid_turn_frac", 0.8))
if not 0.0 < DUEL_MIN_VALID_TURN_FRAC <= 1.0:
    raise RuntimeError(
        f"chain.toml [duel].min_valid_turn_frac must be in (0.0, 1.0]; got {DUEL_MIN_VALID_TURN_FRAC}"
    )
if not 0.0 < DUEL_ALPHA < 1.0:
    raise RuntimeError(
        f"chain.toml [duel].alpha must be in (0.0, 1.0); got {DUEL_ALPHA}"
    )
if not 0.0 < DUEL_GATE_ALPHA < 1.0:
    raise RuntimeError(
        f"chain.toml [duel].gate_alpha must be in (0.0, 1.0); got {DUEL_GATE_ALPHA}"
    )
if not 1 <= DUEL_RESAMPLES <= 1_000_000:
    raise RuntimeError(
        f"chain.toml [duel].bootstrap_resamples must be in [1, 1_000_000]; got {DUEL_RESAMPLES}"
    )

# Pre-eval
PREEVAL_SIM_THRESHOLD: float = float(_p.get("similarity_threshold", 0.95))

