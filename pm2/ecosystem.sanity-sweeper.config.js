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

// Cron: reclaim expired pre-eval leases (dead worker/dispatcher) back to the queue.
module.exports = {
  apps: [
    {
      name: "albedo-sanity-sweeper",
      cwd: path.resolve(__dirname, ".."),
      script: "/home/const/.local/bin/uv",
      args: "run sanity-dispatcher --sweep-abandoned",
      cron_restart: "*/1 * * * *",
      autorestart: false,
      env: loadEnv(),
    },
  ],
};
