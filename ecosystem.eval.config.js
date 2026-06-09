// PM2 process manifest for the GPU eval box only.
//
// This runs scripts/start_eval.sh under PM2. The startup script loads
// ALBEDO_EVAL_ENV_FILE (default: ./eval.env), so put the Slack webhook and
// other eval-only secrets there.
//
// Example:
//   pm2 start ecosystem.eval.config.js
//   pm2 logs albedo-eval-server

function env(key, fallback = "") {
  return process.env[key] ?? fallback;
}

const logsDir = `${__dirname}/logs`;

module.exports = {
  apps: [
    {
      name: "albedo-eval-server",
      script: "./scripts/start_eval.sh",
      interpreter: "/bin/bash",
      cwd: __dirname,
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 1000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      out_file: `${logsDir}/eval.log`,
      error_file: `${logsDir}/eval.error.log`,
      merge_logs: false,
      env: {
        // Explicit path so PM2 always loads the intended eval env file.
        ALBEDO_EVAL_ENV_FILE: env("ALBEDO_EVAL_ENV_FILE", `${__dirname}/eval.env`),
        ALBEDO_EVAL_LOG_DIR: env("ALBEDO_EVAL_LOG_DIR", logsDir),
        // Common eval-server overrides can still be supplied via the shell or PM2.
        ALBEDO_EVAL_HOST: env("ALBEDO_EVAL_HOST", "0.0.0.0"),
        ALBEDO_EVAL_PORT: env("ALBEDO_EVAL_PORT", "9001"),
        ALBEDO_MAX_PARALLEL_TURNS: env("ALBEDO_MAX_PARALLEL_TURNS", "8"),
        ALBEDO_SLACK_WEBHOOK_URL: env("ALBEDO_SLACK_WEBHOOK_URL"),
        ALBEDO_SLACK_USERNAME: env("ALBEDO_SLACK_USERNAME", "Albedo Eval Server"),
        ALBEDO_SLACK_ICON_URL: env("ALBEDO_SLACK_ICON_URL"),
        ALBEDO_SLACK_COOLDOWN_S: env("ALBEDO_SLACK_COOLDOWN_S", "300"),
      },
    },
  ],
};
