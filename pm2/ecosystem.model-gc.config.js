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

// Cron: delete cached model snapshots older than 4h to reclaim disk.
module.exports = {
  apps: [
    {
      name: "albedo-model-gc",
      cwd: path.resolve(__dirname, ".."),
      script: path.resolve(__dirname, "..", "scripts", "cleanup_models.sh"),
      interpreter: "bash",
      cron_restart: "0 * * * *", // hourly; each run deletes anything >4h old
      autorestart: false,
      env: loadEnv(),
    },
  ],
};
