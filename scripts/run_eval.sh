#!/bin/bash
# Auto-restart wrapper for the eval server.
# Runs start_eval.sh in a loop so a crash doesn't leave the eval box idle.
# Usage: nohup bash scripts/run_eval.sh > logs/eval.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

RESTART_DELAY=10   # seconds between crash and restart
MAX_RESTARTS=20    # give up after this many consecutive rapid restarts
RAPID_S=60         # a restart within this many seconds counts as "rapid"

restarts=0
last_start=0

while true; do
    now=$(date +%s)
    if (( now - last_start < RAPID_S )); then
        restarts=$(( restarts + 1 ))
    else
        restarts=0
    fi

    if (( restarts >= MAX_RESTARTS )); then
        echo "[run_eval] $(date): too many rapid restarts ($restarts) — giving up" >&2
        exit 1
    fi

    echo "[run_eval] $(date): starting eval server (restart #$restarts) ..."
    last_start=$(date +%s)

    bash scripts/start_eval.sh || true

    echo "[run_eval] $(date): eval server exited — restarting in ${RESTART_DELAY}s ..."
    sleep "$RESTART_DELAY"
done
