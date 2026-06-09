#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

python3 -m venv .venv
. .venv/bin/activate

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv pip install --upgrade pip wheel setuptools
uv pip install -e .
uv pip install -r swebench_lite_service/requirements.txt

if ! command -v pm2 >/dev/null 2>&1; then
  npm install -g pm2
fi

mkdir -p "${ALBEDO_SWEBENCH_STATE_DIR:-/root/albedo-swebench-lite}"

echo "Setup complete."
echo "Start on the pod with:"
echo "  pm2 start swebench_lite_service/ecosystem.config.js"

