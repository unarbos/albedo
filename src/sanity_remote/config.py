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
    gpu_ids: str = "0,1"
    gpu_util: float = 0.95
    vllm_port: int = 9101
    vllm_dtype: str = "bfloat16"
    vllm_startup_s: float = 600.0
    vllm_python: str = sys.executable  # override if vLLM lives in a separate venv
    vllm_quantization: str = ""
    vllm_enforce_eager: bool = False
    # Passed as --moe-backend <value>; "triton" avoids FlashInfer JIT on CUDA 13 / sm_120f hosts.
    vllm_moe_backend: str = ""
    tensor_parallel_size: int = 2  # GPU_IDS must list exactly this many indices
    cpu_offload_gb: int = 0  # GB to spill to CPU RAM; 2x5090 BF16 needs ~6 for the 67 GB model
    download_timeout_s: float = 1800.0  # 67 GB model can take 20+ min from Hippius
    model_cache_dir: str = "/root/miners_models"
    max_model_len: int = 8192
    kv_cache_dtype: str = "auto"
    gen_temperature: float = 0.7
    gen_top_p: float = 0.8
    gen_top_k: int = 20
    gen_min_p: float = 0.0
    gen_read_timeout_s: float = 300.0

    # Control-plane smoke mode (no GPU) for API/idempotency tests; echoes prompts as responses.
    mock_auto_result: bool = False
    # Skip text heuristics entirely - responses go straight to the LLM judge gate.
    skip_heuristics: bool = False


@lru_cache
def get_remote_settings() -> SanityRemoteSettings:
    # Cached singleton so FastAPI dependencies share one settings instance.
    return SanityRemoteSettings()
