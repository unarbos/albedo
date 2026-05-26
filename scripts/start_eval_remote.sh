#!/bin/bash
# Albedo eval launcher — run on the dedicated GPU eval box.
# Source secrets from ALBEDO_EVAL_ENV_FILE or export them in the shell.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${ALBEDO_EVAL_ENV_FILE:-./eval.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
: "${CHUTES_API_KEY:?must set CHUTES_API_KEY (via eval.env from doppler)}"
export CHUTES_BASE_URL="${CHUTES_BASE_URL:-https://llm.chutes.ai/v1}"

export ALBEDO_EVALS_S3_BUCKET="${ALBEDO_EVALS_S3_BUCKET:-albedo}"
export ALBEDO_EVALS_S3_ACCESS_KEY="${ALBEDO_EVALS_S3_ACCESS_KEY:-${HIPPIUS_ACCESS_KEY:-}}"
export ALBEDO_EVALS_S3_SECRET_KEY="${ALBEDO_EVALS_S3_SECRET_KEY:-${HIPPIUS_SECRET_KEY:-}}"
export ALBEDO_EVALS_ENABLED="${ALBEDO_EVALS_ENABLED:-1}"
export ALBEDO_EVALS_S3_ENDPOINT="${ALBEDO_EVALS_S3_ENDPOINT:-https://s3.hippius.com}"
export ALBEDO_EVALS_S3_PREFIX="${ALBEDO_EVALS_S3_PREFIX:-evals}"
export ALBEDO_EVALS_LOCAL_DIR="${ALBEDO_EVALS_LOCAL_DIR:-/var/albedo/evals}"
export ALBEDO_EVALS_PUBLIC_BASE="${ALBEDO_EVALS_PUBLIC_BASE:-https://us-east-1.hippius.com}"

if [[ -z "${HIPPIUS_HUB_TOKEN:-}" && ( -z "${HIPPIUS_HUB_USERNAME:-}" || -z "${HIPPIUS_HUB_PASSWORD:-}" ) ]]; then
  echo "ERROR: set HIPPIUS_HUB_TOKEN or HIPPIUS_HUB_USERNAME/PASSWORD" >&2
  exit 1
fi

export ALBEDO_DATASET_DIR="${ALBEDO_DATASET_DIR:-/var/albedo/dataset}"
test -d "$ALBEDO_DATASET_DIR"
test -f "$ALBEDO_DATASET_DIR/manifest.json" || test -n "$(find "$ALBEDO_DATASET_DIR" -name 'train-*.parquet' -print -quit)"

# Tune GPU assignments for your eval host. 1.7B fits on one GPU each.
export ALBEDO_KING_GPUS="${ALBEDO_KING_GPUS:-7}"
export ALBEDO_CHAL_GPUS="${ALBEDO_CHAL_GPUS:-6}"
export ALBEDO_GPU_MEMORY_UTILIZATION="${ALBEDO_GPU_MEMORY_UTILIZATION:-0.35}"
export ALBEDO_VLLM_MAX_MODEL_LEN="${ALBEDO_VLLM_MAX_MODEL_LEN:-32768}"
export ALBEDO_VLLM_DTYPE="${ALBEDO_VLLM_DTYPE:-bfloat16}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

export ALBEDO_EVAL_HOST="${ALBEDO_EVAL_HOST:-0.0.0.0}"
export ALBEDO_EVAL_PORT="${ALBEDO_EVAL_PORT:-9001}"
export ALBEDO_MAX_PARALLEL_TURNS="${ALBEDO_MAX_PARALLEL_TURNS:-8}"

exec .venv/bin/uvicorn eval:app \
  --host "$ALBEDO_EVAL_HOST" \
  --port "$ALBEDO_EVAL_PORT" \
  --log-level info
