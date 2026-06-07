#!/bin/bash
# One-shot eval box bootstrap for a fresh Lium GPU pod.
set -euo pipefail
cd /root/albedo

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl build-essential >/dev/null

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

python3 -m venv .venv
source .venv/bin/activate
uv pip install -e . vllm hf_transfer 2>&1 | tail -5

mkdir -p /var/albedo/dataset /root/albedo/hippius_models /root/albedo/tmp/triton_cache /root/albedo/tmp/torchinductor /var/albedo/logs /var/albedo/evals

if [[ ! -f /var/albedo/dataset/manifest.json ]]; then
  echo "prefetching dataset..."
  python scripts/prefetch_dataset.py --out /var/albedo/dataset 2>&1 | tail -10
fi

# Stop any prior eval
pkill -f "uvicorn eval:app" 2>/dev/null || true
sleep 2

nohup bash scripts/start_eval_remote.sh > /var/albedo/logs/eval_server.log 2>&1 &
echo "eval server pid=$!"
sleep 3
curl -sf http://127.0.0.1:9001/health | head -c 500 || tail -20 /var/albedo/logs/eval_server.log
