"""Runtime configuration for the stateless sanity GPU worker (SANITY_REMOTE_* env / .env)."""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SanityRemoteSettings(BaseSettings):
    # All knobs the GPU worker needs; it holds no DB creds, no dataset, no OpenRouter key.
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANITY_REMOTE_", extra="ignore")

    auth_token: str = ""
    host_id: str = "sanity-remote-local"
    host_role: str = "PRE_EVAL"
    ready: bool = True
    api_port: int = 9100  # the worker's own HTTP API (the dispatcher reaches this via the tunnel)

    # vLLM / generation
    gpu_ids: str = "0"
    gpu_util: float = 0.5
    vllm_port: int = 9101
    vllm_dtype: str = "bfloat16"
    vllm_startup_s: float = 180.0
    vllm_python: str = sys.executable  # override if vLLM lives in a separate venv
    download_timeout_s: float = 300.0
    model_cache_dir: str = "/tmp/albedo-sanity-models"
    gen_max_tokens: int = 1024
    max_model_len: int = 8192

    # Control-plane smoke mode (no GPU) for API/idempotency tests; echoes prompts as responses.
    mock_auto_result: bool = False
    # Skip text heuristics entirely - responses go straight to the LLM judge gate.
    skip_heuristics: bool = False


@lru_cache
def get_remote_settings() -> SanityRemoteSettings:
    # Cached singleton so FastAPI dependencies share one settings instance.
    return SanityRemoteSettings()
