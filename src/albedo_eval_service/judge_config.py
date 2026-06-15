from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_JUDGE_",
        extra="ignore",
    )

    api_auth_token: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8091
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api"
    request_timeout_seconds: float = 90.0
    retry_count: int = 5
    retry_backoff_seconds: float = 5.0
    temperature: float = 0.0
    max_tokens: int = 768
    max_concurrency_per_model: int = 2
    min_valid_fraction: float = 0.5


@lru_cache
def get_judge_settings() -> JudgeSettings:
    return JudgeSettings()
