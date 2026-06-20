const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, "..", ".env");
  const env = {};
  if (!fs.existsSync(envPath)) return env;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    env[trimmed.slice(0, index)] = trimmed.slice(index + 1);
  }
  return env;
}

// King-aware cleanup of the eval OCI model cache (/root/albedo-models): keeps the active
// reign's models + the canonical seed + in-flight models, deletes evaluated losers and
// shifted-out former kings, and gives failed models a grace window (CLEANUP_FAIL_GRACE_HOURS,
// default 2h). Long-running daemon that rechecks every 60s (single-instance via a PID lock).
// DB + cache paths come from .env (ALBEDO_POSTGRES_*, ALBEDO_CACHE_DIR).
module.exports = {
  apps: [
    {
      name: "albedo-eval-cache-cleanup",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/python",
      args: "scripts/eval_cache_cleanup.py --execute --interval 60",
      autorestart: true,
      env: loadEnv(),
    },
  ],
};
