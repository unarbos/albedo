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
    accelerator_type: str = "H200"
    ready: bool = True
    mock_auto_verdict: bool = True
    mock_challenger_won: bool = False


@lru_cache
def get_remote_settings() -> RemoteSettings:
    return RemoteSettings()
