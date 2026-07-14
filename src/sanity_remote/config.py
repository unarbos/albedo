"""Runtime configuration for the stateless sanity GPU worker (SANITY_REMOTE_* env / .env)."""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from albedo_eval_service.canonical_model_config import canonical_max_model_len


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
    # 1200s so the first-ever flashinfer GDN kernel JIT compile (~8 min on a fresh box) finishes
    # within the health-wait instead of timing out and killing vLLM mid-compile.
    vllm_startup_s: float = 1200.0
    vllm_python: str = sys.executable  # override if vLLM lives in a separate venv
    vllm_quantization: str = ""
    # Default on: the Qwen3.x MoE hybrids use gated-delta-net (Mamba-style) cache blocks, and
    # CUDA-graph capture fails when max_num_seqs exceeds available Mamba blocks. Eager avoids it.
    vllm_enforce_eager: bool = True
    # Passed as --moe-backend <value>; "triton" avoids FlashInfer JIT on CUDA 13 / sm_120f hosts.
    vllm_moe_backend: str = ""
    vllm_compile_cache_dir: str = ""
    tensor_parallel_size: int = 2  # GPU_IDS must list exactly this many indices
    cpu_offload_gb: int = 0  # GB to spill to CPU RAM; 2x5090 BF16 needs ~6 for the 67 GB model
    # Outer ceiling on a cold model fetch. Must stay above the supervised-download worst case
    # (config_validation.storage._supervise: Hippius 1200s x 2 = 2400s) so the stall watchdog —
    # which kills + resumes cleanly — is what fires, not this blunt cancel (a cancel here can't
    # reclaim the download thread and would race a retry onto the same dir).
    download_timeout_s: float = 3000.0  # 67 GB from Hippius, plus supervised-download retries
    model_cache_dir: str = "/root/miners_models"
    max_model_len: int = canonical_max_model_len()
    kv_cache_dtype: str = "auto"
    vllm_limit_mm: str = '{"image": 0, "video": 0}'
    gen_temperature: float = 0.7
    gen_top_p: float = 0.8
    gen_top_k: int = 20
    gen_min_p: float = 0.0
    # 900s so a generation request that triggers the first-time GDN kernel compile doesn't time out.
    gen_read_timeout_s: float = 900.0

    # Control-plane smoke mode (no GPU) for API/idempotency tests; echoes prompts as responses.
    mock_auto_result: bool = False
    # Skip text heuristics entirely - responses go straight to the LLM judge gate.
    skip_heuristics: bool = False


@lru_cache
def get_remote_settings() -> SanityRemoteSettings:
    # Cached singleton so FastAPI dependencies share one settings instance.
    return SanityRemoteSettings()
