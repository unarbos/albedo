#!/usr/bin/env bash
# Stand up a local single-node OpenSearch for the hippius_validation dedup index.
# Pulls the image, runs the container (security disabled, bound to localhost), waits for health.
# Idempotent: re-running recreates the container.
set -euo pipefail

NAME="${ALBEDO_OS_CONTAINER:-albedo_opensearch}"
IMAGE="${ALBEDO_OS_IMAGE:-opensearchproject/opensearch:2}"
PORT="${ALBEDO_OS_PORT:-9200}"

echo "==> pulling ${IMAGE}"
docker pull "${IMAGE}"

echo "==> (re)creating container ${NAME} on 127.0.0.1:${PORT}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d --name "${NAME}" \
  -p "127.0.0.1:${PORT}:9200" \
  -e discovery.type=single-node \
  -e DISABLE_SECURITY_PLUGIN=true \
  -e DISABLE_INSTALL_DEMO_CONFIG=true \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  "${IMAGE}" >/dev/null

echo "==> waiting for cluster health (this can take ~30-60s on first boot)"
for _ in $(seq 1 90); do
  if curl -fs "http://localhost:${PORT}/_cluster/health" >/dev/null 2>&1; then
    curl -s "http://localhost:${PORT}/_cluster/health?pretty"
    echo "==> OpenSearch ready at http://localhost:${PORT}"
    exit 0
  fi
  sleep 2
done

echo "!! OpenSearch did not become healthy in time; check: docker logs ${NAME}" >&2
exit 1
