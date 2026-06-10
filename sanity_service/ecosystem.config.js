module.exports = {
  apps: [
    {
      name: "albedo-sanity",
      script: ".venv/bin/uvicorn",
      args: "sanity_service.api:app --host 0.0.0.0 --port 9100 --log-level info",
      cwd: "/root/albedo",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 50,
      env: {
        SANITY_VLLM_PORT:        "9101",
        SANITY_GPUS:             "0",
        SANITY_GPU_UTIL:         "0.15",
        SANITY_VLLM_DTYPE:       "bfloat16",
        SANITY_DOWNLOAD_TIMEOUT: "300",
        SANITY_VLLM_STARTUP_S:   "120",
        // Model weights cached here - shared between download and vLLM
        ALBEDO_MODEL_CACHE_DIR:  "/root/sanity/models",
        // Hippius credentials - fill via doppler or export
        HIPPIUS_HUB_TOKEN:       "",
      },
    },
  ],
};
