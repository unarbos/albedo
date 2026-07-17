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

    dataset_version: str = "swe-zero+mini-coder-v1"
    dataset_manifest_uri: str
    # TODO: still the single-source SWE-ZERO hash. Repin to the combined swe-zero+mini-coder
    # manifest sha256 from `scripts/build_manifest.py` output before deploying 2 datasets
    # (also update src/sanity_service/settings.py).
    dataset_manifest_hash: str = "980d50ad40e0b5863a4e624b9e313441bda38626fbba089efb95cbec8aa1a9f4"
    dataset_manifest_path: str | None = None
    sample_count: int = 64
    max_turns_per_sample: int = 10
    sampling_algo: str = "swe-zero-multi-source-sample-v1"
    judge_config_hash: str
    judge_count: int = 3

    artifact_bucket: str = "albedo-artifacts"
    artifact_prefix: str = "s3://albedo-artifacts"

    lease_seconds: int = 1800
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    max_retry_count: int = 3
    prefetch_next_challenger: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
