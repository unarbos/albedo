#!/bin/bash
# Eval-box startup. Run on the GPU host that the validator's tunnel.sh
# forwards into. All eval-box config (GPUs, judge auth, dataset shard,
# eval-trace sink) lives in env vars and is centralised here so the
# operator can audit it in one place.
#
# Recommended: keep this file in a config-managed location (or sourced
# from a private env file) and run under tmux/pm2/systemd.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------
# Required: Chutes judge auth
# ---------------------------------------------------------------------
: "${CHUTES_API_KEY:?must set CHUTES_API_KEY (cpk_... bearer token)}"
export CHUTES_BASE_URL="${CHUTES_BASE_URL:-https://llm.chutes.ai/v1}"

# ---------------------------------------------------------------------
# Required: Hippius S3 for the eval-trace sink (training data publish)
# ---------------------------------------------------------------------
: "${ALBEDO_EVALS_S3_BUCKET:?must set ALBEDO_EVALS_S3_BUCKET (e.g. albedo)}"
: "${ALBEDO_EVALS_S3_ACCESS_KEY:?must set ALBEDO_EVALS_S3_ACCESS_KEY}"
: "${ALBEDO_EVALS_S3_SECRET_KEY:?must set ALBEDO_EVALS_S3_SECRET_KEY}"
export ALBEDO_EVALS_ENABLED="${ALBEDO_EVALS_ENABLED:-1}"
export ALBEDO_EVALS_S3_ENDPOINT="${ALBEDO_EVALS_S3_ENDPOINT:-https://s3.hippius.com}"
export ALBEDO_EVALS_S3_PREFIX="${ALBEDO_EVALS_S3_PREFIX:-evals}"
export ALBEDO_EVALS_LOCAL_DIR="${ALBEDO_EVALS_LOCAL_DIR:-/var/albedo/evals}"
# Public-readable HTTPS base for the bucket. Defaults to us-east-1.hippius.com
# path-style; change if you use a different region or a CDN.
export ALBEDO_EVALS_PUBLIC_BASE="${ALBEDO_EVALS_PUBLIC_BASE:-https://us-east-1.hippius.com}"

# ---------------------------------------------------------------------
# Required: Hippius Hub auth for materializing king/challenger weights
# ---------------------------------------------------------------------
# Either HIPPIUS_HUB_TOKEN, or both HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD.
# (S3-style HIPPIUS_ACCESS_KEY/SECRET_KEY do NOT authenticate the Hub.)
if [[ -z "${HIPPIUS_HUB_TOKEN:-}" && ( -z "${HIPPIUS_HUB_USERNAME:-}" || -z "${HIPPIUS_HUB_PASSWORD:-}" ) ]]; then
  echo "ERROR: set HIPPIUS_HUB_TOKEN, or HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# Required: local SWE-ZERO corpus (run scripts/prefetch_dataset.py first)
# ---------------------------------------------------------------------
: "${ALBEDO_DATASET_DIR:?run scripts/prefetch_dataset.py first and export the printed path}"
test -d "$ALBEDO_DATASET_DIR"
test -f "$ALBEDO_DATASET_DIR/manifest.json" || test -n "$(find "$ALBEDO_DATASET_DIR" -name 'train-*.parquet' -print -quit)"

# ---------------------------------------------------------------------
# vLLM topology — 8x H200 default split. Change if you have a different box.
# ---------------------------------------------------------------------
export ALBEDO_KING_GPUS="${ALBEDO_KING_GPUS:-0,1,2,3}"
export ALBEDO_CHAL_GPUS="${ALBEDO_CHAL_GPUS:-4,5,6,7}"
export ALBEDO_GPU_MEMORY_UTILIZATION="${ALBEDO_GPU_MEMORY_UTILIZATION:-0.85}"
export ALBEDO_VLLM_DTYPE="${ALBEDO_VLLM_DTYPE:-bfloat16}"
# vLLM 0.21 on this CUDA stack needs both disabled (no vendored nvcc for
# flashinfer JIT; no vendored deep_gemm package). Already proven on the
# H200 smoke run.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

# ---------------------------------------------------------------------
# Eval server bind
# ---------------------------------------------------------------------
export ALBEDO_EVAL_HOST="${ALBEDO_EVAL_HOST:-0.0.0.0}"
export ALBEDO_EVAL_PORT="${ALBEDO_EVAL_PORT:-9000}"
# Concurrency cap for judge calls (Chutes will rate-limit beyond ~16 in flight).
export ALBEDO_MAX_PARALLEL_TURNS="${ALBEDO_MAX_PARALLEL_TURNS:-8}"

# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------
exec .venv/bin/uvicorn eval:app \
    --host "$ALBEDO_EVAL_HOST" \
    --port "$ALBEDO_EVAL_PORT" \
    --log-level info
