"""Runtime configuration for the king-chat service (KING_CHAT_* env / .env).

Standalone script package (run as `.venv/bin/python chat_to_king/supervisor.py`), so this is a flat
module imported as a sibling — not part of the installed wheel.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KingChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KING_CHAT_", extra="ignore")

    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("KING_CHAT_DATABASE_URL", "ALBEDO_EVAL_DATABASE_URL"),
    )
    poll_interval_s: float = 30.0
    king_override_uri: str = ""
    king_override_hash: str = ""

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 9202
    reload_notice: str = ""

    llms_path: str = "website/llms.txt"
    llms_url: str = ""
    llms_keywords: str = "albedo,sn97,subnet 97,netuid 97"
    llms_max_chars: int = 0

    served_model_name: str = "albedo-king"
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 9201
    vllm_python: str = sys.executable
    gpu_ids: str = "4,5,6,7"
    tensor_parallel_size: int = 4
    gpu_util: float = 0.90
    vllm_dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    max_model_len: int = 32768
    max_num_seqs: int = 0
    vllm_limit_mm: str = '{"image": 0, "video": 0}'
    vllm_moe_backend: str = "triton"
    vllm_quantization: str = ""
    vllm_enforce_eager: bool = False
    cpu_offload_gb: int = 0
    vllm_startup_s: float = 900.0
    download_timeout_s: float = 3600.0  
    models_dir: str = "/root/albedo-current-king"


@lru_cache
def get_king_chat_settings() -> KingChatSettings:
    return KingChatSettings()
