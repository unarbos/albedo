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

// Cron: replay in-flight pre-evals whose dispatcher crashed mid-poll.
module.exports = {
  apps: [
    {
      name: "albedo-sanity-reconciler",
      cwd: path.resolve(__dirname, ".."),
      script: "/home/const/.local/bin/uv",
      args: "run sanity-dispatcher --reconcile-running",
      cron_restart: "*/1 * * * *",
      autorestart: false,
      env: loadEnv(),
    },
  ],
};
