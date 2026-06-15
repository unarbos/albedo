from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend-side eval service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_EVAL_",
        extra="ignore",
    )

    database_url: str = Field(..., description="Postgres DSN")
    worker_id: str = "eval-dispatcher"
    remote_auth_token: str = ""

    dataset_version: str = "AlienKevin/SWE-ZERO-12M-trajectories"
    dataset_manifest_uri: str
    dataset_manifest_hash: str = "982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f"
    dataset_manifest_path: str | None = None
    sample_count: int = 128
    max_turns_per_sample: int = 10
    sampling_algo: str = "swe-zero-manifest-sample-v1"
    judge_config_hash: str
    judge_count: int = 3

    artifact_bucket: str = "albedo-artifacts"
    artifact_prefix: str = "s3://albedo-artifacts"

    lease_seconds: int = 1800
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
