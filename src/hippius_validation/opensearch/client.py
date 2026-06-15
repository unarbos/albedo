"""OpenSearch connection, health check, and lazy index setup for the dedup corpus."""
from __future__ import annotations

import functools

from loguru import logger as log

from hippius_validation import config


@functools.lru_cache(maxsize=1)
def get_client():
    from opensearchpy import OpenSearch

    use_ssl = config.OPENSEARCH_URL.lower().startswith("https")
    auth = (config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD) if config.OPENSEARCH_USER else None
    return OpenSearch(
        hosts=[config.OPENSEARCH_URL],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


def health() -> bool:
    """True if the cluster is reachable and green/yellow."""
    try:
        status = get_client().cluster.health().get("status")
        log.info("opensearch health: {}", status)
        return status in ("green", "yellow")
    except Exception as exc:  # noqa: BLE001
        log.warning("opensearch health check failed: {}", exc)
        return False


def _mapping(dim: int) -> dict:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "key": {"type": "keyword"},
                "hotkey": {"type": "keyword"},
                "repo": {"type": "keyword"},
                "digest": {"type": "keyword"},
                "model_uri": {"type": "keyword"},
                "created_at": {"type": "date"},
                "norm_vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                },
                # Full fingerprint (incl. tensor_samples) — stored for exact rerank, not indexed.
                "fingerprint": {"type": "object", "enabled": False},
            }
        },
    }


def index_name(dim: int) -> str:
    """Per-architecture index name. knn_vector has a fixed dimension, and models with a
    different tensor count are a different architecture that must never be cross-compared,
    so each norm-vector length gets its own index."""
    return f"{config.OPENSEARCH_INDEX}_{dim}"


def ensure_index(dim: int) -> str:
    """Create the per-dimension dedup index if it does not exist; return its name."""
    name = index_name(dim)
    c = get_client()
    if not c.indices.exists(index=name):
        c.indices.create(index=name, body=_mapping(dim))
        log.info("created opensearch index {} (knn dim={})", name, dim)
    return name
