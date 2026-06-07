// PM2 process manifest for the Albedo validator (refactor / Hippius-only).
//
// Two long-lived processes:
//   albedo-eval-tunnel   SSH forward from the validator host to the GPU eval box
//   albedo-validator     the async king-of-the-hill loop
//
// Inject secrets via env (e.g. `doppler run -- pm2 start ecosystem.config.js`).
// The eval box runs scripts/start_eval.sh separately (it is not managed here).
//
// VALIDATOR WALLET / NETUID — read before changing:
//   - set ALBEDO_NETUID to your subnet; set_weights uses your registered hotkey there.
//   - do NOT point at an unregistered hotkey or a different netuid.

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
        // Eval server reached through the tunnel above.
        ALBEDO_EVAL_SERVER: env("ALBEDO_EVAL_SERVER", "http://localhost:9001"),
        ALBEDO_NETUID: env("ALBEDO_NETUID", "97"),
        ALBEDO_NETWORK: env("ALBEDO_NETWORK", "finney"),
        // Hippius-only subnet keeps reveals timelock-encrypted — leave CR required.
        ALBEDO_REQUIRE_COMMIT_REVEAL: env("ALBEDO_REQUIRE_COMMIT_REVEAL", "1"),
        BT_WALLET_NAME: env("BT_WALLET_NAME", "default"),
        BT_WALLET_HOTKEY: env("BT_WALLET_HOTKEY", "default"),
        ALBEDO_BURN_UID: env("ALBEDO_BURN_UID", "0"),
        // Validator state store (R2/Hippius S3).
        ALBEDO_R2_ENDPOINT: env("ALBEDO_R2_ENDPOINT", "https://s3.hippius.com"),
        ALBEDO_R2_BUCKET: env("ALBEDO_R2_BUCKET", "albedo-state"),
        ALBEDO_R2_ACCESS_KEY: env("ALBEDO_R2_ACCESS_KEY"),
        ALBEDO_R2_SECRET_KEY: env("ALBEDO_R2_SECRET_KEY"),
        // Eval-trace / dashboard store.
        ALBEDO_DS_ENDPOINT: env("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com"),
        ALBEDO_DS_BUCKET: env("ALBEDO_DS_BUCKET", "albedo"),
        ALBEDO_DS_ACCESS_KEY: env("ALBEDO_DS_ACCESS_KEY"),
        ALBEDO_DS_SECRET_KEY: env("ALBEDO_DS_SECRET_KEY"),
        // Loop cadence + robustness knobs.
        ALBEDO_POLL_INTERVAL: env("ALBEDO_POLL_INTERVAL", "30"),
        ALBEDO_WEIGHT_INTERVAL: env("ALBEDO_WEIGHT_INTERVAL", "300"),
        ALBEDO_EVAL_HARD_TIMEOUT_S: env("ALBEDO_EVAL_HARD_TIMEOUT_S", "2400"),
        ALBEDO_STREAM_IDLE_WARN_S: env("ALBEDO_STREAM_IDLE_WARN_S", "600"),
        ALBEDO_STREAM_IDLE_KILL_S: env("ALBEDO_STREAM_IDLE_KILL_S", "1800"),
        ALBEDO_REEVAL_LOOKBACK_HOURS: env("ALBEDO_REEVAL_LOOKBACK_HOURS", "24"),
        ALBEDO_MAX_REEVAL_PER_HOTKEY: env("ALBEDO_MAX_REEVAL_PER_HOTKEY", "1"),
        ALBEDO_EVAL_BOX_BACKOFF_S: env("ALBEDO_EVAL_BOX_BACKOFF_S", "120"),
        ALBEDO_EVAL_BOX_BACKOFF_MAX_S: env("ALBEDO_EVAL_BOX_BACKOFF_MAX_S", "1800"),
        // Dashboard display cutoff — kings crowned before this block are hidden.
        // Set to the first block of the current competition. 0 = show all.
        ALBEDO_DISPLAY_START_BLOCK: env("ALBEDO_DISPLAY_START_BLOCK", "8326394"),
      },
    },
  ],
};
