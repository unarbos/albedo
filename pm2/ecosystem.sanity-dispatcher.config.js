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

// Stable-side pre-eval dispatcher: claim -> sample -> push to worker -> judge -> persist.
module.exports = {
  apps: [
    {
      name: "albedo-sanity-dispatcher",
      cwd: path.resolve(__dirname, ".."),
      script: "/home/const/.local/bin/uv",
      args: "run sanity-dispatcher",
      env: loadEnv(),
    },
  ],
};
