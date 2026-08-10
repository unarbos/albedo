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
    # Starting per-model in-flight cap (adaptive gate initial). Ramp may climb
    # toward adaptive_concurrency_max until a 429 / sustained latency OOR, then
    # hold at ~80% of in-flight. Keep start in the EAGLE-friendly band — 72→96
    # flooded GLM and inflated latency to 100–280s (eval slower than 63m baseline).
    max_concurrency_per_model: int = 32
    # Cap how many samples /score-batch scores in parallel. Unbounded gather of
    # a full eval (e.g. 100) + per-model LiteLLM fan-out can stall the event
    # loop long enough for kubelet liveness to SIGKILL the pod (exit 137).
    max_score_sample_concurrency: int = 24
    # Adaptive 429/latency ramp (OpenRouterJudgeClient). Ceiling should stay ≤
    # LiteLLM max_parallel_requests and prefer EAGLE sweet-spot (~2×16–24/replica).
    adaptive_concurrency_enabled: bool = True
    adaptive_concurrency_max: int = 48
    adaptive_concurrency_min: int = 8
    adaptive_hold_ratio: float = 0.8
    adaptive_ramp_every_successes: int = 16
    adaptive_cooldown_successes: int = 32
    # Latency monitor: warn from the first request if absolute max is hit;
    # after a start-of-run baseline is set, also warn at baseline × ratio.
    # Sustained OOR triggers the same 80% hold as a 429.
    latency_max_seconds: float = 60.0
    latency_max_ratio: float = 3.0
    latency_baseline_samples: int = 8
    latency_oor_strikes_before_hold: int = 3
    min_valid_fraction: float = 0.8
    evaluator_model: str = "z-ai/glm-5.2"
    evaluator_providers: str = "z-ai,novita,siliconflow,streamlake"
    sota_models: str = "z-ai/glm-5.2"
    sota_max_tokens: int = 8192
    sota_trajectory_turns: int = 8
    num_questions: int = 50
    question_max_tokens: int = 20000
    simulation_max_tokens: int = 4096
    simulation_loop_reruns: int = 0
    simulation_model: str = "deepseek/deepseek-v4-flash-0731"
    simulation_providers: str = "deepseek,cloudflare"
    answer_max_tokens: int = 20000
    question_prep_ttl_seconds: float = 14400.0
    repo_context_url: str = ""
    repo_context_timeout_seconds: float = 20.0

    slack_error_webhook_url: str = ""


@lru_cache
def get_judge_settings() -> JudgeSettings:
    return JudgeSettings()
