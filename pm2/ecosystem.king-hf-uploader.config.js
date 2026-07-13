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
    // Drop a trailing inline comment ("VALUE   # note") before using the value.
    const value = trimmed.slice(index + 1).replace(/\s#.*$/, "").trim();
    env[trimmed.slice(0, index)] = value;
  }
  return env;
}

module.exports = {
  apps: [
    {
      name: "albedo-king-hf-uploader",
      cwd: path.resolve(__dirname, ".."),
      script: "uv",
      args: "run python scripts/king_hf_uploader.py",
      interpreter: "none",
      autorestart: true,
      env: loadEnv(),
    },
  ],
};
