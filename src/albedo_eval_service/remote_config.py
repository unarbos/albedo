from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from .canonical_model_config import GENESIS_MODEL_CONFIG_REF


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
    max_new_tokens: int = 32768
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int | None = None
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "auto"
    use_canonical_model_config: bool = True
    canonical_model_config_ref: str = GENESIS_MODEL_CONFIG_REF

    resolve_model_artifacts: bool = True
    model_cache_dir: str = "/tmp/albedo-remote-models"
    artifact_spool_dir: str = "/tmp/albedo-remote-artifacts"
    remote_state_dir: str = "/tmp/albedo-remote-state"

    scoring_backend: str = "http"
    scoring_base_url: str | None = None
    scoring_auth_token: str = ""
    scoring_timeout_seconds: float = 1800.0
    scoring_min_valid_fraction: float = 0.5

    upload_artifacts: bool = True
    cleanup_local_artifacts: bool = False
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_session_token: str | None = None


@lru_cache
def get_remote_settings() -> RemoteSettings:
    return RemoteSettings()
