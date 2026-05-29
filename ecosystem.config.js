// PM2 process manifest for the Albedo validator.
//
// Two long-lived processes:
//   albedo-eval-tunnel   SSH forward from the validator host to the GPU eval box
//   albedo-validator     the main async king-of-the-hill loop
//
// Inject secrets via environment variables (e.g. `doppler run -- pm2 start ecosystem.config.js`).
//
// VALIDATOR WALLET / NETUID (read before changing):
//   - Albedo runs on subnet 97 (ALBEDO_NETUID=97).
//   - set_weights uses your registered validator hotkey on that subnet.
//   - Do NOT point at an unregistered hotkey or a different netuid.

function env(key, fallback = "") {
  return process.env[key] ?? fallback;
}

module.exports = {
  apps: [
    {
      name: "albedo-eval-tunnel",
      script: "./tunnel.sh",
      cwd: __dirname,
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 1000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_EVAL_HOST: env("ALBEDO_EVAL_HOST"),
        ALBEDO_EVAL_SSH_PORT: env("ALBEDO_EVAL_SSH_PORT", "22"),
        ALBEDO_EVAL_SSH_USER: env("ALBEDO_EVAL_SSH_USER", "root"),
        ALBEDO_EVAL_LOCAL_PORT: env("ALBEDO_EVAL_LOCAL_PORT", "9001"),
        ALBEDO_EVAL_REMOTE_PORT: env("ALBEDO_EVAL_REMOTE_PORT", "9001"),
      },
    },
    {
      name: "albedo-validator",
      script: "validator.py",
      interpreter: `${__dirname}/.venv/bin/python`,
      cwd: __dirname,
      max_restarts: 10,
      restart_delay: 5000,
      autorestart: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_EVAL_SERVER: env("ALBEDO_EVAL_SERVER", "http://localhost:9001"),
        ALBEDO_NETUID: env("ALBEDO_NETUID", "97"),
        ALBEDO_REQUIRE_COMMIT_REVEAL: env("ALBEDO_REQUIRE_COMMIT_REVEAL", "0"),
        ALBEDO_NETWORK: env("ALBEDO_NETWORK", "finney"),
        BT_WALLET_NAME: env("BT_WALLET_NAME", "default"),
        BT_WALLET_HOTKEY: env("BT_WALLET_HOTKEY", "default"),
        ALBEDO_BURN_UID: env("ALBEDO_BURN_UID", "0"),
        ALBEDO_R2_ENDPOINT: env("ALBEDO_R2_ENDPOINT", "https://s3.hippius.com"),
        ALBEDO_R2_BUCKET: env("ALBEDO_R2_BUCKET", "albedo-state"),
        ALBEDO_R2_ACCESS_KEY: env("ALBEDO_R2_ACCESS_KEY"),
        ALBEDO_R2_SECRET_KEY: env("ALBEDO_R2_SECRET_KEY"),
        ALBEDO_DS_ENDPOINT: env("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com"),
        ALBEDO_DS_BUCKET: env("ALBEDO_DS_BUCKET", "albedo"),
        ALBEDO_DS_ACCESS_KEY: env("ALBEDO_DS_ACCESS_KEY"),
        ALBEDO_DS_SECRET_KEY: env("ALBEDO_DS_SECRET_KEY"),
        ALBEDO_POLL_INTERVAL: env("ALBEDO_POLL_INTERVAL", "30"),
        ALBEDO_WEIGHT_INTERVAL: env("ALBEDO_WEIGHT_INTERVAL", "300"),
        ALBEDO_TICK_RESTART_AFTER: env("ALBEDO_TICK_RESTART_AFTER", "2400"),
        ALBEDO_STREAM_IDLE_WARN_S: env("ALBEDO_STREAM_IDLE_WARN_S", "600"),
        ALBEDO_STREAM_IDLE_KILL_S: env("ALBEDO_STREAM_IDLE_KILL_S", "1800"),
        ALBEDO_REEVAL_LOOKBACK_HOURS: env("ALBEDO_REEVAL_LOOKBACK_HOURS", "24"),
        ALBEDO_MAX_REEVAL_PER_HOTKEY: env("ALBEDO_MAX_REEVAL_PER_HOTKEY", "1"),
        ALBEDO_EVAL_BOX_BACKOFF_S: env("ALBEDO_EVAL_BOX_BACKOFF_S", "120"),
        ALBEDO_EVAL_BOX_BACKOFF_MAX_S: env("ALBEDO_EVAL_BOX_BACKOFF_MAX_S", "1800"),
        ALBEDO_PREEVAL_CLEAR_STATE: env("ALBEDO_PREEVAL_CLEAR_STATE", "0"),
        ALBEDO_DISPLAY_START_BLOCK: env("ALBEDO_DISPLAY_START_BLOCK", "8288861"),
      },
    },
  ],
};
