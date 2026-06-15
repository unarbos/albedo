const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, "..", ".env");
  const env = { ...process.env };
  if (!fs.existsSync(envPath)) return env;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    const key = trimmed.slice(0, index);
    const value = trimmed.slice(index + 1);
    if (value !== "") env[key] = value;
  }
  return env;
}

const env = loadEnv();

// GPU-side stateless sanity worker. Deploy on any fresh GPU box; holds no DB/dataset/OR key.
module.exports = {
  apps: [
    {
      name: env.SANITY_REMOTE_PM2_NAME || "albedo-sanity-remote-api",
      cwd: path.resolve(__dirname, ".."),
      script: env.SANITY_REMOTE_UV_PATH || "uv",
      args: "run sanity-remote",
      autorestart: true,
      env,
    },
  ],
};
