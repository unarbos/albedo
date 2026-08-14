from __future__ import annotations

import functools

from loguru import logger as log

from albedo_config import get_model_validation_settings

config = get_model_validation_settings()


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
    try:
        status = get_client().cluster.health().get("status")
        log.info("opensearch health: {}", status)
        return status in ("green", "yellow")
    except Exception as exc:
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
                "weights_hash": {"type": "keyword"},
                "norm_vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                },
                "fingerprint": {"type": "object", "enabled": False},
            }
        },
    }


def index_name(dim: int) -> str:
    return f"{config.OPENSEARCH_INDEX}_{dim}"


def ensure_index(dim: int) -> str:
    name = index_name(dim)
    c = get_client()
    if not c.indices.exists(index=name):
        c.indices.create(index=name, body=_mapping(dim))
        log.info("created opensearch index {} (knn dim={})", name, dim)
    else:
        c.indices.put_mapping(
            index=name, body={"properties": {"weights_hash": {"type": "keyword"}}}
        )
    return name
