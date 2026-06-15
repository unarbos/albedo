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

module.exports = {
  apps: [
    {
      name: "albedo-judge-api",
      cwd: path.resolve(__dirname, ".."),
      script: "uv",
      args: "run --no-sync albedo-judge-api",
      env: loadEnv(),
    },
  ],
};
