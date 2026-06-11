module.exports = {
  apps: [
    {
      name: "albedo-sanity",
      script: ".venv/bin/python",
      args: "-m sanity_service",
      cwd: "/root/albedo",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 50,
      env: {
        SANITY_PORT:             "9100",
        SANITY_VLLM_PORT:        "9101",
        SANITY_GPUS:             "0",
        SANITY_GPU_UTIL:         "0.5",
        SANITY_VLLM_DTYPE:       "bfloat16",
        SANITY_DOWNLOAD_TIMEOUT: "300",
        SANITY_VLLM_STARTUP_S:   "180",
        // Model weights cached here - shared between download and vLLM
        ALBEDO_MODEL_CACHE_DIR:  "/root/sanity/models",
        // Optional: set to enable LLM coherence gate
        SANITY_OR_API_KEY:       "",
        SANITY_OR_MODEL:         "deepseek/deepseek-v3.2",
        // Hippius credentials - fill via doppler or export
        HIPPIUS_HUB_TOKEN:       "",
      },
    },
  ],
};
