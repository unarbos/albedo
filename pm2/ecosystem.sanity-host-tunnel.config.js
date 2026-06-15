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
const sshTarget = env.ALBEDO_SANITY_GPU_HOST_USER + "@" + env.ALBEDO_SANITY_GPU_HOST_SSH_HOST;
const sshArgs = [
  "-N",
  "-o",
  "ExitOnForwardFailure=yes",
  "-o",
  "ServerAliveInterval=30",
  "-o",
  "ServerAliveCountMax=3",
  "-L",
  env.ALBEDO_SANITY_TUNNEL_LOCAL_PORT + ":127.0.0.1:" + env.SANITY_REMOTE_API_PORT,
];

if (env.ALBEDO_SANITY_GPU_HOST_SSH_PORT) {
  sshArgs.push("-p", env.ALBEDO_SANITY_GPU_HOST_SSH_PORT);
}

sshArgs.push(sshTarget);

// Stable side opens an -L forward to the GPU worker, so the dispatcher reaches it at
// 127.0.0.1:ALBEDO_SANITY_TUNNEL_LOCAL_PORT. The stable side never needs the GPU box's address
// anywhere durable - re-point this one config when the GPU box changes.
module.exports = {
  apps: [
    {
      name: "albedo-sanity-host-tunnel",
      script: "ssh",
      args: sshArgs.join(" "),
      autorestart: true,
      env,
    },
  ],
};
