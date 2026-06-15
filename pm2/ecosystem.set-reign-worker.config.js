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

module.exports = {
  apps: [
    {
      name: "albedo-set-reign-worker",
      cwd: path.resolve(__dirname, ".."),
      script: "uv",
      args: "run --no-sync python -m set_reign_worker",
      env: {
        ...loadEnv(),
        ALBEDO_SET_REIGN_WORKER_ID: "set-reign-worker",
      },
    },
  ],
};
