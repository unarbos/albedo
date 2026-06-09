#!/usr/bin/env bash
set -euo pipefail

: "${ALBEDO_SWEBENCH_HOST:?must set ALBEDO_SWEBENCH_HOST}"
SSH_PORT="${ALBEDO_SWEBENCH_SSH_PORT:-40296}"
SSH_USER="${ALBEDO_SWEBENCH_SSH_USER:-root}"
LOCAL_PORT="${ALBEDO_SWEBENCH_LOCAL_PORT:-18080}"
REMOTE_PORT="${ALBEDO_SWEBENCH_REMOTE_PORT:-18080}"

exec ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -p "${SSH_PORT}" \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${SSH_USER}@${ALBEDO_SWEBENCH_HOST}"

