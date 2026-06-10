#!/bin/bash
# One-shot bootstrap for a fresh GPU pod running the sanity service.
set -euo pipefail

: "${HIPPIUS_HUB_TOKEN:?ERROR: HIPPIUS_HUB_TOKEN must be set before running setup.sh}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl build-essential nodejs npm

# pm2
if ! command -v pm2 &>/dev/null; then
  npm install -g pm2
fi

# uv
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone repo and switch to dev branch
if [[ ! -d /root/albedo ]]; then
  git clone https://github.com/unarbos/albedo.git /root/albedo
fi
cd /root/albedo
git fetch origin
git checkout dev
git pull origin dev

# Install deps - albedo package + sanity service extras
uv venv
source .venv/bin/activate
uv pip install -e . vllm hf_transfer loguru

# Dirs
mkdir -p /root/sanity/models /var/log/sanity

# Start
pm2 start sanity_service/ecosystem.config.js
pm2 save
echo "Sanity service started. Health: curl http://localhost:9100/health"
