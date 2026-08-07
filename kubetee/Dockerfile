# SN97 Denrite (albedo) — KubeTEE PoC evaluator image.
#
# Built INSIDE the albedo repo on the kubetee-poc branch so the image, the code
# it calls, and the manifest all stay in lockstep with the albedo tree.

# Base: albedo pins vllm==0.23.0 + torch==2.11.0 in pyproject.toml. The Docker Hub
# vllm/vllm-openai:v0.23.0 image ships exactly these versions, pre-validated for
# H200 (Hopper SM90). The NGC image (nvcr.io/nvidia/vllm:26.06-py3) ships
# v0.22.1-dev which crashes on the Qwen3.6-35B-A3B hybrid linear/full attention
# MoE architecture (UnicodeDecodeError in torch ops cache clearing).
# CUDA 12.9 build (cu129) is required for H200 driver compatibility.
FROM vllm/vllm-openai:v0.23.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---------------------------
# 1) Copy albedo sources + pyproject so pip-installed deps resolve.
# ---------------------------

COPY src ./src
COPY pyproject.toml ./pyproject.toml
# chain.toml is read at import time by config_validation/config.py:9 (imported
# by the HF download child process via _fastdl). Without it, the king/challenger
# model download fails with FileNotFoundError: /app/chain.toml.
COPY chain.toml ./chain.toml
# scripts/ is needed by the init container (prepare_datasets.py builds the
# dataset manifest). It imports sibling modules (prepare_datasets, build_manifest_meta)
# + albedo_eval_service under src/, so both must be on PYTHONPATH.
COPY scripts ./scripts
# assets/ contains the canonical Qwen3.6-35B-A3B tokenizer files used by
# RemoteEvalWorker._load_samples (canonical_model_config.py:15 → remote_worker.py:42).
# Without these, transformers fails with "Repo id must be in the form..." when
# trying to load the tokenizer from the local path.
COPY assets ./assets

# ---------------------------
# 2) Install albedo's non-vLLM deps. We do NOT pin transitive torch / triton
#    versions from upstream here — those are already on the Hopper NGC image
#    and albedo's pins (torch==2.11.0) aren't yet Hopper-clean; letting the
#    base image provide the GPU stack is the entire reason this base works.
# ---------------------------

RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "httpx>=0.27" \
    "boto3>=1.35" \
    "asyncpg>=0.29" \
    "psycopg[binary]>=3.2" \
    "loguru>=0.7" \
    "datasets" \
    "transformers" \
    "huggingface-hub==1.19.0" \
    "hf_transfer>=0.1" \
    "hippius-hub==0.6.0" \
    "opensearch-py>=2,<3" \
    "prompt_toolkit>=3" \
    "pydantic>=2.8" \
    "pydantic-settings>=2.4" \
    "pyarrow>=16" \
    "safetensors" \
    "numpy"

# Install the albedo package in editable mode so the [project.scripts] console
# entry points (albedo-judge-api, albedo-eval-dispatcher, etc.) are created on
# $PATH. The --no-deps flag avoids re-resolving deps we already installed above
# (and skips the torch/vllm pins that would fight the NGC base image's versions).
# The editable install uses the src/ copy below (no duplicate source).
RUN pip install --no-cache-dir --no-deps -e .

# ---------------------------
# 3) KubeTEE PoC entrypoint (this directory). Kept in a separate package so the
#    albedo tree remains untouched (upstream commits cleanly on the dev branch).
# ---------------------------

RUN mkdir -p /app/kubetee
COPY kubetee/app ./kubetee/app
COPY kubetee/deploy ./kubetee/deploy

# PYTHONPATH must find the albedo sources (src/), the PoC wrapper (kubetee/app/),
# and the scripts/ dir (for the init container's prepare_datasets.py sibling imports).
# `albedo_eval_service` is the importable top-level package (from
# [tool.hatch.build.targets.wheel] in pyproject.toml) because src/ is on the path.
ENV PYTHONPATH=/app/src:/app:/app/scripts

# ---------------------------
# Run the PoC entrypoint. The full env surface is documented in
# kubetee/docs/env.md and consumed at runtime from the Armada job spec +
# ConfigMap/Secret refs, not baked into the image.
# ---------------------------

ENTRYPOINT ["python", "/app/kubetee/app/run_eval.py"]
