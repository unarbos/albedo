from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)



def _env_csv(key: str, default: str) -> list[str]:
    raw = os.environ.get(key, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    dashboard_url: str = os.environ.get(
        "ALBEDO_SWEBENCH_DASHBOARD_URL",
        "https://us-east-1.hippius.com/albedo/dashboard.json",
    )
    state_dir: Path = Path(os.environ.get("ALBEDO_SWEBENCH_STATE_DIR", "/root/albedo-swebench-lite"))
    dataset_name: str = os.environ.get("ALBEDO_SWEBENCH_DATASET", "SWE-bench/SWE-bench_Lite")
    split: str = os.environ.get("ALBEDO_SWEBENCH_SPLIT", "test")
    limit_instances: int = _env_int("ALBEDO_SWEBENCH_LIMIT", 0)
    instance_filter: str = os.environ.get("ALBEDO_SWEBENCH_FILTER", "")
    generation_concurrency: int = _env_int("ALBEDO_SWEBENCH_GEN_CONCURRENCY", 16)
    generation_timeout_s: float = _env_float("ALBEDO_SWEBENCH_GEN_TIMEOUT_S", 240.0)
    generation_max_tokens: int = _env_int("ALBEDO_SWEBENCH_GEN_MAX_TOKENS", 2048)
    generation_temperature: float = _env_float("ALBEDO_SWEBENCH_GEN_TEMPERATURE", 0.0)
    agent_workers: int = _env_int("ALBEDO_SWEBENCH_AGENT_WORKERS", 32)
    harness_workers: int = _env_int("ALBEDO_SWEBENCH_MAX_WORKERS", 32)
    harness_timeout_s: int = _env_int("ALBEDO_SWEBENCH_TEST_TIMEOUT_S", 1800)
    vllm_host: str = os.environ.get("ALBEDO_SWEBENCH_VLLM_HOST", "127.0.0.1")
    vllm_port: int = _env_int("ALBEDO_SWEBENCH_VLLM_PORT", 18100)
    vllm_gpus: str = os.environ.get("ALBEDO_SWEBENCH_VLLM_GPUS", "0,1,2,3,4,5,6,7")
    vllm_data_parallel_size: int = _env_int("ALBEDO_SWEBENCH_VLLM_DATA_PARALLEL_SIZE", len(_env_csv("ALBEDO_SWEBENCH_VLLM_GPUS", "0,1,2,3,4,5,6,7")))
    vllm_dtype: str = os.environ.get("ALBEDO_SWEBENCH_VLLM_DTYPE", "bfloat16")
    vllm_max_model_len: int = _env_int("ALBEDO_SWEBENCH_VLLM_MAX_MODEL_LEN", 40960)
    vllm_gpu_memory_utilization: float = _env_float("ALBEDO_SWEBENCH_VLLM_GPU_MEM", 0.90)
    model_cache_dir: str = os.environ.get("ALBEDO_MODEL_CACHE_DIR", "/root/albedo/hippius_models")
    run_id_prefix: str = os.environ.get("ALBEDO_SWEBENCH_RUN_ID_PREFIX", "albedo-kings-lite")
    loop_sleep_s: int = _env_int("ALBEDO_SWEBENCH_LOOP_SLEEP_S", 60)
    s3_enabled: bool = _env_bool("ALBEDO_SWEBENCH_S3_ENABLED", True)
    s3_endpoint: str = os.environ.get(
        "ALBEDO_SWEBENCH_S3_ENDPOINT",
        os.environ.get("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com"),
    )
    s3_bucket: str = os.environ.get(
        "ALBEDO_SWEBENCH_S3_BUCKET",
        os.environ.get("ALBEDO_DS_BUCKET", "albedo"),
    )
    s3_access_key: str = os.environ.get(
        "ALBEDO_SWEBENCH_S3_ACCESS_KEY",
        os.environ.get("ALBEDO_DS_ACCESS_KEY", ""),
    )
    s3_secret_key: str = os.environ.get(
        "ALBEDO_SWEBENCH_S3_SECRET_KEY",
        os.environ.get("ALBEDO_DS_SECRET_KEY", ""),
    )
    s3_prefix: str = os.environ.get("ALBEDO_SWEBENCH_S3_PREFIX", "swebench-lite").strip("/")
    s3_public_base_url: str = os.environ.get("ALBEDO_SWEBENCH_S3_PUBLIC_BASE_URL", "")

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.state_dir / "reports"


SETTINGS = Settings()

