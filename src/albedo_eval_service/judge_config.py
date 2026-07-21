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
    parse_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 768
    max_concurrency_per_model: int = 128
    min_valid_fraction: float = 0.8
    evaluator_model: str = "z-ai/glm-5.2"
    evaluator_providers: str = "z-ai,novita,siliconflow,streamlake"
    sota_models: str = "z-ai/glm-5.2"
    sota_max_tokens: int = 4096
    sota_trajectory_turns: int = 4
    num_questions: int = 50
    question_max_tokens: int = 16000
    simulation_max_tokens: int = 4096
    answer_max_tokens: int = 8000
    question_prep_ttl_seconds: float = 1800.0

    slack_error_webhook_url: str = ""


@lru_cache
def get_judge_settings() -> JudgeSettings:
    return JudgeSettings()
