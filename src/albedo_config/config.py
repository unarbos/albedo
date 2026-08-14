from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from albedo_eval_service.modelstore.canonical_model_config import (
    GENESIS_MODEL_CONFIG_REF,
    canonical_max_model_len,
)
from albedo_eval_service.shared.dataset_manifest import DEFAULT_DATASET_MANIFEST_HASH
from albedo_eval_service.shared.sampling import SAMPLING_ALGO

from .db import load_dotenv, postgres_dsn
from .models import (
    ENGY_MODELS,
    EVALUATOR_MODEL,
    EVALUATOR_PROVIDERS,
    SIMULATION_MODEL,
    SIMULATION_PROVIDERS,
    SOTA_MODELS,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_EVAL_",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(..., description="Postgres DSN")
    worker_id: str = "eval-dispatcher"
    remote_auth_token: str = Field(
        "",
        validation_alias=AliasChoices("ALBEDO_EVAL_REMOTE_AUTH_TOKEN", "ALBEDO_REMOTE_AUTH_TOKEN"),
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8080

    dataset_version: str = "mini-coder+open-swe+smith-rs+hero-v1"
    dataset_manifest_uri: str
    dataset_manifest_hash: str = DEFAULT_DATASET_MANIFEST_HASH
    dataset_manifest_path: str | None = None
    sample_count: int = 100
    sampling_algo: str = SAMPLING_ALGO
    judge_config_hash: str
    judge_count: int = 1

    artifact_bucket: str = "albedo-artifacts"
    artifact_prefix: str = "s3://albedo/albedo-eval-service"

    lease_seconds: int = 20
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    max_retry_count: int = 3
    prefetch_next_challenger: bool = True


@lru_cache
def get_eval_settings() -> Settings:
    return Settings()


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_JUDGE_",
        extra="ignore",
        populate_by_name=True,
    )

    api_auth_token: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8091
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api"
    engy_api_key: str = Field("", validation_alias=AliasChoices("ALBEDO_JUDGE_ENGY_API_KEY"))
    engy_base_url: str = "https://api.engy.ai"
    engy_models: str = ENGY_MODELS
    engy_max_errors: int = 25
    request_timeout_seconds: float = 1800.0
    retry_count: int = 5
    retry_backoff_seconds: float = 1.5
    parse_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 768
    max_concurrency_per_model: int = 128
    min_valid_fraction: float = 0.8
    evaluator_model: str = EVALUATOR_MODEL
    evaluator_providers: str = EVALUATOR_PROVIDERS
    sota_models: str = SOTA_MODELS
    sota_max_tokens: int = 8192
    sota_trajectory_turns: int = 8
    num_questions: int = 50
    question_max_tokens: int = 20000
    simulation_max_tokens: int = 4096
    simulation_loop_reruns: int = 0
    simulation_model: str = SIMULATION_MODEL
    simulation_providers: str = SIMULATION_PROVIDERS
    answer_max_tokens: int = 20000
    question_prep_ttl_seconds: float = 14400.0
    repo_context_url: str = ""
    repo_context_timeout_seconds: float = 20.0
    slack_error_webhook_url: str = ""


@lru_cache
def get_judge_settings() -> JudgeSettings:
    return JudgeSettings()


class ScoreBridgeClientSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_SCORE_BRIDGE_",
        extra="ignore",
        populate_by_name=True,
    )

    remote_ws_url: str = "ws://127.0.0.1:18090/score-bridge"
    remote_auth_token: str = Field(
        "",
        validation_alias=AliasChoices(
            "ALBEDO_SCORE_BRIDGE_REMOTE_AUTH_TOKEN", "ALBEDO_REMOTE_AUTH_TOKEN"
        ),
    )
    judge_base_url: str = "http://127.0.0.1:8091"
    judge_auth_token: str = Field(
        "",
        validation_alias=AliasChoices(
            "ALBEDO_SCORE_BRIDGE_JUDGE_AUTH_TOKEN", "ALBEDO_JUDGE_API_AUTH_TOKEN"
        ),
    )
    request_timeout_seconds: float = 1800.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    ping_interval_seconds: float = 20.0
    ping_timeout_seconds: float = 20.0
    websocket_max_size_bytes: int = 2048 * 1024 * 1024
    retry_count: int = 5
    retry_backoff_seconds: float = 1.5


class RemoteSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_REMOTE_",
        extra="ignore",
        populate_by_name=True,
    )

    auth_token: str = ""
    host_id: str = "remote-eval-local"
    host_role: str = "EVAL"
    gpu_count: int = 8
    free_gpu_count: int = 8
    accelerator_type: str = "RTX_PRO_6000_BLACKWELL"
    ready: bool = True

    eval_api_host: str = "0.0.0.0"
    eval_api_port: int = 8090

    mock_auto_verdict: bool = False
    mock_challenger_won: bool = False

    generation_backend: str = "vllm"
    dataset_root: str | None = None
    repo_context_url: str = ""
    previous_king_model: str | None = None
    challenger_model: str | None = None
    previous_king_gpu_ids: str = "0,1,2,3"
    challenger_gpu_ids: str = "4,5,6,7"
    max_new_tokens: int = 16384
    generation_result_timeout_seconds: float = 900.0
    trajectory_assistant_turns: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int | None = None
    enforce_eager: bool = False
    compile_cache_dir: str = ""
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "auto"
    use_canonical_model_config: bool = True
    canonical_model_config_ref: str = GENESIS_MODEL_CONFIG_REF

    resolve_model_artifacts: bool = True
    model_cache_dir: str = "/tmp/albedo-remote-models"
    model_download_concurrency: int = 8
    model_download_out_of_process: bool = True
    model_download_stall_seconds: float = 180.0
    model_download_stall_retries: int = 3
    hippius_download_stall_seconds: float = 1200.0
    model_download_read_timeout_seconds: float = 120.0
    artifact_spool_dir: str = "/tmp/albedo-remote-artifacts"
    remote_state_dir: str = "/tmp/albedo-remote-state"

    scoring_backend: str = "websocket"
    scoring_base_url: str | None = None
    scoring_auth_token: str = ""
    scoring_timeout_seconds: float = 1800.0
    scoring_batch_concurrency: int = 128
    scoring_retry_count: int = 5
    scoring_retry_backoff_seconds: float = 1.5
    scoring_min_valid_fraction: float = 0.8

    upload_artifacts: bool = True
    cleanup_local_artifacts: bool = False
    s3_endpoint_url: str | None = Field(
        None,
        validation_alias=AliasChoices("ALBEDO_REMOTE_S3_ENDPOINT_URL", "ALBEDO_S3_ENDPOINT"),
    )
    s3_region: str | None = None
    s3_access_key_id: str | None = Field(
        None,
        validation_alias=AliasChoices("ALBEDO_REMOTE_S3_ACCESS_KEY_ID", "ALBEDO_S3_ACCESS_KEY"),
    )
    s3_secret_access_key: str | None = Field(
        None,
        validation_alias=AliasChoices("ALBEDO_REMOTE_S3_SECRET_ACCESS_KEY", "ALBEDO_S3_SECRET_KEY"),
    )
    s3_session_token: str | None = None


@lru_cache
def get_remote_settings() -> RemoteSettings:
    return RemoteSettings()


def _control_db_url() -> str:
    load_dotenv()
    return postgres_dsn()


class SanitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SANITY_DISPATCH_",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(default_factory=_control_db_url)
    worker_id: str = "sanity-dispatcher"
    remote_auth_token: str = Field(
        "",
        validation_alias=AliasChoices(
            "SANITY_DISPATCH_REMOTE_AUTH_TOKEN", "SANITY_REMOTE_AUTH_TOKEN"
        ),
    )
    consensus: bool = False

    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = DEFAULT_DATASET_MANIFEST_HASH
    dataset_root: str = ""
    sample_count: int = 3
    trajectory_assistant_turns: int = 32
    gen_max_tokens: int = 16384

    skip_viability: bool = True

    lease_seconds: int = 3600
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    min_free_gpus: int = 1
    max_retry_count: int = 5


@lru_cache
def get_sanity_settings() -> SanitySettings:
    return SanitySettings()


class SanityRemoteSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANITY_REMOTE_", extra="ignore")

    auth_token: str = ""
    host_id: str = "sanity-remote-local"
    host_role: str = "PRE_EVAL"
    ready: bool = True
    api_port: int = 9100

    gpu_ids: str = "0,1"
    gpu_util: float = 0.95
    vllm_port: int = 9101
    vllm_dtype: str = "bfloat16"
    vllm_startup_s: float = 1200.0
    vllm_python: str = sys.executable
    vllm_quantization: str = ""
    vllm_enforce_eager: bool = True
    vllm_moe_backend: str = ""
    vllm_compile_cache_dir: str = ""
    tensor_parallel_size: int = 2
    data_parallel_size: int = 1
    data_parallel_size_local: int = 0
    cpu_offload_gb: int = 0
    download_timeout_s: float = 3000.0
    model_cache_dir: str = "/root/miners_models"
    max_model_len: int = canonical_max_model_len()
    kv_cache_dtype: str = "auto"
    vllm_limit_mm: str = '{"image": 0, "video": 0}'
    gen_temperature: float = 0.7
    gen_top_p: float = 0.8
    gen_top_k: int = 20
    gen_min_p: float = 0.0
    gen_read_timeout_s: float = 900.0

    mock_auto_result: bool = False
    skip_heuristics: bool = False


@lru_cache
def get_sanity_remote_settings() -> SanityRemoteSettings:
    return SanityRemoteSettings()


class RepoContextSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_REPO_CONTEXT_",
        extra="ignore",
    )

    api_host: str = "127.0.0.1"
    api_port: int = 8093
    cache_dir: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = ""
    dataset_root: str = ""
    github_token: str = ""
    max_paths: int = 2000
    max_files: int = 12
    max_file_chars: int = 30000
    max_context_chars: int = 120000
    max_snapshot_mb: int = 500
    max_cache_gb: float = 60.0
    max_trajectory_pairs: int = 8


@lru_cache
def get_repo_context_settings() -> RepoContextSettings:
    return RepoContextSettings()


class SetReignSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_SET_REIGN_",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(
        ...,
        description="Postgres DSN",
        validation_alias=AliasChoices("ALBEDO_SET_REIGN_DATABASE_URL", "ALBEDO_EVAL_DATABASE_URL"),
    )
    worker_id: str = "set-reign-worker"
    lease_seconds: int = 20
    dispatch_poll_seconds: float = 5.0


class ChainReaderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHAIN_",
        extra="ignore",
        populate_by_name=True,
    )

    DB_URL: str = Field(default_factory=_control_db_url)
    NETUID: int = 97
    NETWORK: str = "finney"
    POLL_INTERVAL_S: float = 2.0
    START_BLOCK: int = 0
    SCAN_LAG_BLOCKS: int = 3
    IGNORE_COMMITS_TO_BLOCK: int = Field(
        0, validation_alias=AliasChoices("IGNORE_COMMITS_TO_BLOCK")
    )

    @field_validator("START_BLOCK", "SCAN_LAG_BLOCKS", "IGNORE_COMMITS_TO_BLOCK", mode="before")
    @classmethod
    def _blank_is_zero(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return 0
        return value


@lru_cache
def get_chain_reader_settings() -> ChainReaderSettings:
    load_dotenv()
    return ChainReaderSettings()


_MV_DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "albedo_models")
_MV_ARCH_SPEC_PATH = str(
    Path(__file__).resolve().parents[1] / "model_validation" / "validate" / "architecture_spec.json"
)


class ModelValidationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_",
        extra="ignore",
        populate_by_name=True,
    )

    DB_URL: str = Field(default_factory=_control_db_url)
    NETUID: int = Field(97, validation_alias=AliasChoices("CHAIN_NETUID"))
    MODEL_CACHE_DIR: str = _MV_DEFAULT_CACHE_DIR
    OPENSEARCH_URL: str = "http://127.0.0.1:9200"
    OPENSEARCH_USER: str = ""
    OPENSEARCH_PASSWORD: str = ""
    OPENSEARCH_INDEX: str = "albedo_fingerprints"
    MAX_KNN_DIM: int = 16000
    S3_BUCKET: str = ""
    S3_ENDPOINT: str = "https://s3.hippius.com"
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    FP_FILE: str = "fingerprint.json"
    TENSORS_FILE: str = "tensors.json"
    ARCH_SPEC_PATH: str = Field(
        _MV_ARCH_SPEC_PATH, validation_alias=AliasChoices("ALBEDO_ARCH_SPEC")
    )
    SIM_THRESHOLD: float = 0.95
    KNN_CANDIDATES: int = 20
    POLL_INTERVAL_S: float = Field(5.0, validation_alias=AliasChoices("ALBEDO_HV_POLL_S"))
    LEASE_SECONDS: int = Field(600, validation_alias=AliasChoices("ALBEDO_HV_LEASE_S"))
    HEARTBEAT_S: float = Field(30.0, validation_alias=AliasChoices("ALBEDO_HV_HEARTBEAT_S"))
    MAX_ATTEMPTS: int = Field(5, validation_alias=AliasChoices("ALBEDO_HV_MAX_ATTEMPTS"))

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    REQUIRE_SAFETENSORS: ClassVar[bool] = True
    ALLOWED_FILES: ClassVar[tuple[str, ...]] = (
        "model.safetensors.index.json",
        ".gitattributes",
        "LICENSE",
        "README.md",
    )
    ALLOWED_GLOBS: ClassVar[tuple[str, ...]] = ("model-*-of-*.safetensors", "model.safetensors")
    FORBIDDEN_GLOBS: ClassVar[tuple[str, ...]] = ("*.py",)

    @field_validator("MODEL_CACHE_DIR", mode="before")
    @classmethod
    def _blank_cache_dir_uses_default(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return _MV_DEFAULT_CACHE_DIR
        return value


@lru_cache
def get_model_validation_settings() -> ModelValidationSettings:
    load_dotenv()
    return ModelValidationSettings()


class WeightSetterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    coldkey: str = ""
    hotkey: str = ""
    wallet_path: str = ""
    network: str = "finney"
    netuid: int = 97
    set_rate_blocks: int = 101
    poll_seconds: float = 12.0
    worker_id: str = "weight-setter"
    burn_uid: int = 0
    retry_backoff_seconds: int = 60

    @classmethod
    def from_env(cls) -> "WeightSetterSettings":
        return cls(
            database_url=os.environ.get("ALBEDO_EVAL_DATABASE_URL", ""),
            coldkey=os.environ.get("ALBEDO_WEIGHT_COLDKEY", ""),
            hotkey=os.environ.get("ALBEDO_WEIGHT_HOTKEY", ""),
            wallet_path=os.environ.get("ALBEDO_WEIGHT_WALLET_PATH", ""),
            network=os.environ.get("ALBEDO_WEIGHT_NETWORK")
            or os.environ.get("CHAIN_NETWORK", "finney"),
            netuid=int(
                os.environ.get("ALBEDO_WEIGHT_NETUID") or os.environ.get("CHAIN_NETUID") or "97"
            ),
            set_rate_blocks=int(os.environ.get("ALBEDO_WEIGHT_SET_RATE_BLOCKS", "101")),
            poll_seconds=float(os.environ.get("ALBEDO_WEIGHT_POLL_SECONDS", "12")),
            worker_id=os.environ.get("ALBEDO_WEIGHT_WORKER_ID", "weight-setter"),
            burn_uid=int(os.environ.get("ALBEDO_WEIGHT_BURN_UID", "0")),
            retry_backoff_seconds=int(os.environ.get("ALBEDO_WEIGHT_RETRY_BACKOFF_SECONDS", "60")),
        )
