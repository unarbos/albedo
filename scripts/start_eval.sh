#!/bin/bash
# Eval-box startup — run on the GPU host the validator's tunnel.sh forwards into.
# Source secrets from ALBEDO_EVAL_ENV_FILE (default ./eval.env) or export them
# directly. Recommended: populate eval.env via doppler or a secrets manager.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- Load env file if present -----------------------------------------------
ENV_FILE="${ALBEDO_EVAL_ENV_FILE:-./eval.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# --- Required: Chutes judge auth --------------------------------------------
: "${CHUTES_API_KEY:?must set CHUTES_API_KEY (cpk_... bearer token)}"
export CHUTES_BASE_URL="${CHUTES_BASE_URL:-https://llm.chutes.ai}"

# --- Required: Hippius Hub auth for materializing king/challenger weights ----
if [[ -z "${HIPPIUS_HUB_TOKEN:-}" && ( -z "${HIPPIUS_HUB_USERNAME:-}" || -z "${HIPPIUS_HUB_PASSWORD:-}" ) ]]; then
  echo "ERROR: set HIPPIUS_HUB_TOKEN, or HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD" >&2
  exit 1
fi

# --- Required: eval-trace + fingerprint-state S3 sink ----------------------
: "${ALBEDO_EVALS_S3_BUCKET:?must set ALBEDO_EVALS_S3_BUCKET (e.g. albedo)}"
: "${ALBEDO_EVALS_S3_ACCESS_KEY:?must set ALBEDO_EVALS_S3_ACCESS_KEY}"
: "${ALBEDO_EVALS_S3_SECRET_KEY:?must set ALBEDO_EVALS_S3_SECRET_KEY}"
export ALBEDO_EVALS_ENABLED="${ALBEDO_EVALS_ENABLED:-1}"
export ALBEDO_EVALS_S3_ENDPOINT="${ALBEDO_EVALS_S3_ENDPOINT:-https://s3.hippius.com}"
export ALBEDO_EVALS_S3_PREFIX="${ALBEDO_EVALS_S3_PREFIX:-evals}"
export ALBEDO_EVALS_LOCAL_DIR="${ALBEDO_EVALS_LOCAL_DIR:-/var/albedo/evals}"
export ALBEDO_EVALS_PUBLIC_BASE="${ALBEDO_EVALS_PUBLIC_BASE:-https://us-east-1.hippius.com}"

# --- Required: local SWE-ZERO corpus (run scripts/prefetch_dataset.py first) -
: "${ALBEDO_DATASET_DIR:?run scripts/prefetch_dataset.py first and export the printed path}"
test -d "$ALBEDO_DATASET_DIR"
test -f "$ALBEDO_DATASET_DIR/manifest.json"

# --- vLLM topology — both sides share GPU 0 (H100 80GB fits two 4B instances) -
# Each instance uses ~35% memory (~28GB); total ~56GB of 80GB.
# Override ALBEDO_KING_GPUS / ALBEDO_CHAL_GPUS for multi-GPU boxes.
export ALBEDO_KING_GPUS="${ALBEDO_KING_GPUS:-0}"
export ALBEDO_CHAL_GPUS="${ALBEDO_CHAL_GPUS:-0}"
export ALBEDO_GPU_MEMORY_UTILIZATION="${ALBEDO_GPU_MEMORY_UTILIZATION:-0.35}"
export ALBEDO_VLLM_DTYPE="${ALBEDO_VLLM_DTYPE:-bfloat16}"
# vLLM flags proven on the H200 stack (no vendored nvcc/deep_gemm).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

# --- Storage + tmp dirs -----------------------------------------------------
export ALBEDO_MODEL_CACHE_DIR="${ALBEDO_MODEL_CACHE_DIR:-/root/albedo/hippius_models}"
export ALBEDO_MIN_DISK_BYTES="${ALBEDO_MIN_DISK_BYTES:-53687091200}"  # 50 GB
export ALBEDO_TMP_DIR="${ALBEDO_TMP_DIR:-/root/albedo/tmp}"
mkdir -p "$ALBEDO_TMP_DIR/triton_cache" "$ALBEDO_TMP_DIR/torchinductor"
export TMPDIR="$ALBEDO_TMP_DIR"
export TRITON_CACHE_DIR="$ALBEDO_TMP_DIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$ALBEDO_TMP_DIR/torchinductor"

# --- Eval server bind + judge concurrency -----------------------------------
export ALBEDO_EVAL_HOST="${ALBEDO_EVAL_HOST:-0.0.0.0}"
export ALBEDO_EVAL_PORT="${ALBEDO_EVAL_PORT:-9001}"   # must match tunnel + validator
export ALBEDO_MAX_PARALLEL_TURNS="${ALBEDO_MAX_PARALLEL_TURNS:-8}"

# --- Launch -----------------------------------------------------------------
exec .venv/bin/uvicorn albedo.eval_server.endpoints:app \
    --host "$ALBEDO_EVAL_HOST" \
    --port "$ALBEDO_EVAL_PORT" \
    --log-level info
