function env(key, fallback = "") {
  return process.env[key] ?? fallback;
}

const repo = __dirname + "/..";

module.exports = {
  apps: [
    {
      name: "albedo-swebench-lite-api",
      script: ".venv/bin/python",
      args: "-m uvicorn swebench_lite_service.api:app --host 127.0.0.1 --port 18080",
      cwd: repo,
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 1000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_SWEBENCH_STATE_DIR: env("ALBEDO_SWEBENCH_STATE_DIR", "/root/albedo-swebench-lite"),
        ALBEDO_SWEBENCH_LIMIT: env("ALBEDO_SWEBENCH_LIMIT", "0"),
        ALBEDO_MODEL_CACHE_DIR: env("ALBEDO_MODEL_CACHE_DIR", "/root/albedo/hippius_models"),
        ALBEDO_SWEBENCH_S3_ENABLED: env("ALBEDO_SWEBENCH_S3_ENABLED", "1"),
        ALBEDO_SWEBENCH_S3_ENDPOINT: env("ALBEDO_SWEBENCH_S3_ENDPOINT", env("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com")),
        ALBEDO_SWEBENCH_S3_BUCKET: env("ALBEDO_SWEBENCH_S3_BUCKET", env("ALBEDO_DS_BUCKET", "albedo")),
        ALBEDO_SWEBENCH_S3_PREFIX: env("ALBEDO_SWEBENCH_S3_PREFIX", "swebench-lite"),
        ALBEDO_SWEBENCH_S3_PUBLIC_BASE_URL: env("ALBEDO_SWEBENCH_S3_PUBLIC_BASE_URL", ""),
      },
    },
    {
      name: "albedo-swebench-lite-worker",
      script: ".venv/bin/python",
      args: "-m swebench_lite_service.worker",
      cwd: repo,
      autorestart: false,
      max_restarts: 1,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_SWEBENCH_STATE_DIR: env("ALBEDO_SWEBENCH_STATE_DIR", "/root/albedo-swebench-lite"),
        ALBEDO_SWEBENCH_LIMIT: env("ALBEDO_SWEBENCH_LIMIT", "0"),
        ALBEDO_SWEBENCH_FILTER: env("ALBEDO_SWEBENCH_FILTER", ""),
        ALBEDO_SWEBENCH_GEN_CONCURRENCY: env("ALBEDO_SWEBENCH_GEN_CONCURRENCY", "16"),
        ALBEDO_SWEBENCH_AGENT_WORKERS: env("ALBEDO_SWEBENCH_AGENT_WORKERS", "32"),
        ALBEDO_SWEBENCH_MAX_WORKERS: env("ALBEDO_SWEBENCH_MAX_WORKERS", "32"),
        ALBEDO_SWEBENCH_VLLM_GPUS: env("ALBEDO_SWEBENCH_VLLM_GPUS", "0,1,2,3,4,5,6,7"),
        ALBEDO_SWEBENCH_VLLM_DATA_PARALLEL_SIZE: env("ALBEDO_SWEBENCH_VLLM_DATA_PARALLEL_SIZE", "8"),
        ALBEDO_SWEBENCH_VLLM_PORT: env("ALBEDO_SWEBENCH_VLLM_PORT", "18100"),
        ALBEDO_SWEBENCH_VLLM_GPU_MEM: env("ALBEDO_SWEBENCH_VLLM_GPU_MEM", "0.90"),
        ALBEDO_SWEBENCH_VLLM_MAX_MODEL_LEN: env("ALBEDO_SWEBENCH_VLLM_MAX_MODEL_LEN", "40960"),
        ALBEDO_SWEBENCH_VLLM_ATTENTION_BACKEND: env("ALBEDO_SWEBENCH_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
        VLLM_USE_DEEP_GEMM: env("VLLM_USE_DEEP_GEMM", "0"),
        VLLM_MOE_USE_DEEP_GEMM: env("VLLM_MOE_USE_DEEP_GEMM", "0"),
        VLLM_DEEP_GEMM_WARMUP: env("VLLM_DEEP_GEMM_WARMUP", "skip"),
        CUDA_HOME: env("CUDA_HOME", repo + "/.venv/lib/python3.12/site-packages/nvidia/cu13"),
        CUDA_PATH: env("CUDA_PATH", repo + "/.venv/lib/python3.12/site-packages/nvidia/cu13"),
        PATH: repo + "/.venv/bin:" + env("PATH", "") + ":" + repo + "/.venv/lib/python3.12/site-packages/nvidia/cu13/bin",
        LD_LIBRARY_PATH: env("LD_LIBRARY_PATH", "") + ":" + repo + "/.venv/lib/python3.12/site-packages/nvidia/cu13/lib",
        ALBEDO_MODEL_CACHE_DIR: env("ALBEDO_MODEL_CACHE_DIR", "/root/albedo/hippius_models"),
        ALBEDO_SWEBENCH_S3_ENABLED: env("ALBEDO_SWEBENCH_S3_ENABLED", "1"),
        ALBEDO_SWEBENCH_S3_ENDPOINT: env("ALBEDO_SWEBENCH_S3_ENDPOINT", env("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com")),
        ALBEDO_SWEBENCH_S3_BUCKET: env("ALBEDO_SWEBENCH_S3_BUCKET", env("ALBEDO_DS_BUCKET", "albedo")),
        ALBEDO_SWEBENCH_S3_PREFIX: env("ALBEDO_SWEBENCH_S3_PREFIX", "swebench-lite"),
        ALBEDO_SWEBENCH_S3_PUBLIC_BASE_URL: env("ALBEDO_SWEBENCH_S3_PUBLIC_BASE_URL", ""),
        HIPPIUS_HUB_TOKEN: env("HIPPIUS_HUB_TOKEN"),
      },
    },
  ],
};

