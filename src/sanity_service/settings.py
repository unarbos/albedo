"""Dispatcher settings for the sanity pre-eval service (SANITY_DISPATCH_* env / .env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from sanity_service.config import DB_URL as _DEFAULT_DB_URL


class SanitySettings(BaseSettings):
    # Stable-side dispatcher config; the judge key comes from ALBEDO_JUDGE_* (get_judge_settings).
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANITY_DISPATCH_", extra="ignore")

    database_url: str = _DEFAULT_DB_URL
    worker_id: str = "sanity-dispatcher"
    remote_auth_token: str = ""
    consensus: bool = False

    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = "982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f"
    dataset_root: str = ""
    sample_count: int = 3
    max_turns_per_sample: int = 10
    gen_max_tokens: int = 32768

    skip_viability: bool = False

    lease_seconds: int = 600
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    min_free_gpus: int = 1
    max_retry_count: int = 5


@lru_cache
def get_settings() -> SanitySettings:
    # Cached settings singleton.
    return SanitySettings()
