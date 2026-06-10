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

const env = loadEnv();

module.exports = {
  apps: [
    {
      name: "albedo-remote-eval-api",
      cwd: path.resolve(__dirname, ".."),
      script: env.ALBEDO_REMOTE_EVAL_PM2_SCRIPT || "uv",
      args: env.ALBEDO_REMOTE_EVAL_PM2_ARGS || "run albedo-remote-eval-api",
      env,
    },
  ],
};
