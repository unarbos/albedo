#!/usr/bin/env bash
# Install hippius_validation's extra runtime deps into the target venv:
#   - opensearch-py (the dedup index client)
#   - config_validation (editable; shared fingerprint/architecture/hippius logic)
# The venv already provides bittensor, asyncpg, loguru, numpy, boto3, hippius_hub.
set -euo pipefail

VENV="${ALBEDO_VENV:-}"
CV_PATH="${CONFIG_VALIDATION_PATH:- }"
PY="${VENV}/bin/python"

[ -x "${PY}" ] || { echo "!! venv python not found at ${PY}" >&2; exit 1; }

echo "==> installing opensearch-py"
"${PY}" -m pip install -q "opensearch-py>=2,<3"

echo "==> installing config_validation (editable, no-deps) from ${CV_PATH}"
"${PY}" -m pip install -q -e "${CV_PATH}" --no-deps

echo "==> verifying imports"
"${PY}" -c "import opensearchpy, config_validation; print('opensearch-py + config_validation OK')"
