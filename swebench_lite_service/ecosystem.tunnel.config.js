function env(key, fallback = "") {
  return process.env[key] ?? fallback;
}

module.exports = {
  apps: [
    {
      name: "albedo-swebench-lite-tunnel",
      script: "./tunnel.sh",
      cwd: __dirname,
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 1000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_SWEBENCH_HOST: env("ALBEDO_SWEBENCH_HOST", "216.243.220.131"),
        ALBEDO_SWEBENCH_SSH_PORT: env("ALBEDO_SWEBENCH_SSH_PORT", "40008"),
        ALBEDO_SWEBENCH_SSH_USER: env("ALBEDO_SWEBENCH_SSH_USER", "root"),
        ALBEDO_SWEBENCH_LOCAL_PORT: env("ALBEDO_SWEBENCH_LOCAL_PORT", "18080"),
        ALBEDO_SWEBENCH_REMOTE_PORT: env("ALBEDO_SWEBENCH_REMOTE_PORT", "18080"),
      },
    },
    {
      name: "albedo-swebench-lite-s3-uploader",
      script: "python3",
      args: "-m swebench_lite_service.host_s3_uploader --loop",
      cwd: __dirname + "/..",
      autorestart: true,
      restart_delay: 10000,
      max_restarts: 1000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        ALBEDO_SWEBENCH_REMOTE: env("ALBEDO_SWEBENCH_REMOTE", "root@216.243.220.131"),
        ALBEDO_SWEBENCH_SSH_PORT: env("ALBEDO_SWEBENCH_SSH_PORT", "40008"),
        ALBEDO_SWEBENCH_REMOTE_STATE_DIR: env("ALBEDO_SWEBENCH_REMOTE_STATE_DIR", "/root/albedo-swebench-lite"),
        ALBEDO_SWEBENCH_LOCAL_MIRROR: env("ALBEDO_SWEBENCH_LOCAL_MIRROR", "/tmp/albedo-swebench-lite-mirror"),
        ALBEDO_SWEBENCH_UPLOAD_INTERVAL: env("ALBEDO_SWEBENCH_UPLOAD_INTERVAL", "300"),
        ALBEDO_SWEBENCH_S3_ENDPOINT: env("ALBEDO_SWEBENCH_S3_ENDPOINT", env("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com")),
        ALBEDO_SWEBENCH_S3_BUCKET: env("ALBEDO_SWEBENCH_S3_BUCKET", env("ALBEDO_DS_BUCKET", "albedo")),
        ALBEDO_SWEBENCH_S3_ACCESS_KEY: env("ALBEDO_SWEBENCH_S3_ACCESS_KEY", env("ALBEDO_DS_ACCESS_KEY")),
        ALBEDO_SWEBENCH_S3_SECRET_KEY: env("ALBEDO_SWEBENCH_S3_SECRET_KEY", env("ALBEDO_DS_SECRET_KEY")),
        ALBEDO_SWEBENCH_S3_PREFIX: env("ALBEDO_SWEBENCH_S3_PREFIX", "swebench-lite"),
      },
    },
  ],
};

