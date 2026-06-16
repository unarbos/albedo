#!/usr/bin/env bash
# Reclaim disk: delete cached model snapshots older than 4h (by mtime) from the
# Hippius model cache. Run periodically by pm2 (pm2/ecosystem.model-gc.config.js).
set -euo pipefail

# Mirror config.py: ALBEDO_MODEL_CACHE_DIR unset/empty → ~/.cache/albedo_models.
ROOT="${ALBEDO_MODEL_CACHE_DIR:-$HOME/.cache/albedo_models}"
MAX_AGE_MIN="${ALBEDO_MODEL_TTL_MIN:-240}"   # 4 hours

if [ ! -d "$ROOT" ]; then
  echo "model-gc: cache dir $ROOT does not exist — nothing to do"
  exit 0
fi

# Snapshot dirs are leaf dirs named sha256_<digest>. -prune so we don't descend into a dir
# we're about to delete.
while IFS= read -r -d '' dir; do
  echo "model-gc: removing $dir"
  rm -rf "$dir"
done < <(find "$ROOT" -type d -name 'sha256_*' -mmin "+$MAX_AGE_MIN" -prune -print0)

# Prune now-empty <namespace>/<name> parents left behind (keep the cache root itself).
find "$ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true

echo "model-gc: done (root=$ROOT, ttl=${MAX_AGE_MIN}m)"
