const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, ".env");
  const env = { ...process.env, PYTHONPATH: path.resolve(__dirname, "src") };
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

module.exports = {
  apps: [
    {
      name: "chain_reader",
      cwd: __dirname,
      script: "uv",
      args: "run chain-reader",
      env,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
    {
      name: "hippius_validation",
      cwd: __dirname,
      script: "uv",
      args: "run hippius-validation",
      env,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
