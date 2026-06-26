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
    retry_backoff_seconds: float = 1.5
    temperature: float = 0.0
    max_tokens: int = 768
    max_concurrency_per_model: int = 8
    min_valid_fraction: float = 0.5

    chutes_api_key: str = ""
    chutes_base_url: str = "https://llm.chutes.ai"
    glm52_model: str = "zai-org/GLM-5.2-TEE"
    glm52_chute_id: str = "08901219-159f-55a7-87cf-9d0d02744668"
    openrouter_glm52_model: str = "z-ai/glm-5.2"
    openrouter_glm52_quantizations: str = "fp8"
    category_count: int = 5
    category_prompt_version: str = "glm52-categories-v1"
    category_prep_ttl_seconds: float = 1800.0
    glm_request_timeout_seconds: float = 90.0
    glm_retry_count: int = 2
    glm_temperature: float = 0.0
    glm_max_tokens: int = 2048
    glm_candidate_max_tokens: int = 4096
    glm_max_concurrency: int = 8
    slack_error_webhook_url: str = ""


@lru_cache
def get_judge_settings() -> JudgeSettings:
    return JudgeSettings()
