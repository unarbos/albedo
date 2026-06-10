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
const sshTarget = `${env.ALBEDO_GPU_HOST_USER}@${env.ALBEDO_GPU_HOST_SSH_HOST}`;

module.exports = {
  apps: [
    {
      name: "albedo-backend-to-gpu-api-tunnel",
      script: "ssh",
      args: [
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-L",
        `${env.ALBEDO_TUNNEL_BACKEND_LOCAL_GPU_PORT}:127.0.0.1:${env.ALBEDO_REMOTE_EVAL_API_PORT}`,
        sshTarget,
      ].join(" "),
      autorestart: true,
      env,
    },
    {
      name: "albedo-gpu-to-backend-api-tunnel",
      script: "ssh",
      args: [
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        `${env.ALBEDO_TUNNEL_GPU_REMOTE_BACKEND_PORT}:127.0.0.1:${env.ALBEDO_BACKEND_API_PORT}`,
        sshTarget,
      ].join(" "),
      autorestart: true,
      env,
    },
  ],
};
