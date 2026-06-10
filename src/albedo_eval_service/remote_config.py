from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RemoteSettings(BaseSettings):
    """Runtime configuration for the remote eval control-plane API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_REMOTE_",
        extra="ignore",
    )

    auth_token: str = ""
    host_id: str = "remote-eval-local"
    host_role: str = "EVAL"
    gpu_count: int = 8
    free_gpu_count: int = 8
    accelerator_type: str = "B200"
    ready: bool = True

    # Control-plane smoke mode exists only for API health/idempotency tests.
    mock_auto_verdict: bool = False
    mock_challenger_won: bool = False

    generation_backend: str = "vllm"
    dataset_root: str | None = None
    previous_king_model: str | None = None
    challenger_model: str | None = None
    previous_king_gpu_ids: str = "0,1,2,3"
    challenger_gpu_ids: str = "4,5,6,7"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int | None = None
    enforce_eager: bool = False


@lru_cache
def get_remote_settings() -> RemoteSettings:
    return RemoteSettings()
