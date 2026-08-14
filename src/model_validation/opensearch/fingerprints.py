from __future__ import annotations

from loguru import logger as log

from albedo_config import get_model_validation_settings
from config_validation.fingerprint import similarity
from model_validation.opensearch.client import ensure_index, get_client

config = get_model_validation_settings()


def find_duplicate(
    fp: dict,
    hotkey: str,
    threshold: float | None = None,
    k: int | None = None,
    weights_hash: str | None = None,
) -> dict:
    threshold = config.SIM_THRESHOLD if threshold is None else threshold
    k = config.KNN_CANDIDATES if k is None else k
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))

    if weights_hash:
        exact_body = {
            "size": 1,
            "_source": ["key", "hotkey", "model_uri"],
            "query": {
                "bool": {
                    "filter": [{"term": {"weights_hash": weights_hash}}],
                    "must_not": [{"term": {"hotkey": hotkey}}] if hotkey else [],
                }
            },
        }
        exact_hits = get_client().search(index=index, body=exact_body)["hits"]["hits"]
        if exact_hits:
            src = exact_hits[0].get("_source", {})
            log.warning("duplicate: exact weights_hash match vs {}", src.get("key", ""))
            return {
                "is_duplicate": True,
                "similarity": 1.0,
                "threshold": threshold,
                "matched_key": src.get("key", ""),
                "matched_hotkey": src.get("hotkey", ""),
                "matched_model_uri": src.get("model_uri", ""),
                "candidates_checked": len(exact_hits),
                "exact_weights_match": True,
            }

    body = {
        "size": k,
        "_source": ["key", "hotkey", "model_uri", "fingerprint"],
        "query": {"knn": {"norm_vector": {"vector": vec, "k": k}}},
    }
    hits = get_client().search(index=index, body=body)["hits"]["hits"]

    best_sim, matched = 0.0, {"key": "", "hotkey": "", "model_uri": ""}
    for hit in hits:
        src = hit.get("_source", {})
        if hotkey and src.get("hotkey") == hotkey:
            continue
        sim = similarity(fp, src.get("fingerprint", {}))
        if sim > best_sim:
            best_sim = sim
            matched = {
                "key": src.get("key", ""),
                "hotkey": src.get("hotkey", ""),
                "model_uri": src.get("model_uri", ""),
            }

    is_dup = best_sim >= threshold
    result = {
        "is_duplicate": is_dup,
        "similarity": best_sim,
        "threshold": threshold,
        "matched_key": matched["key"] if is_dup else "",
        "matched_hotkey": matched["hotkey"] if is_dup else "",
        "matched_model_uri": matched["model_uri"] if is_dup else "",
        "candidates_checked": len(hits),
        "exact_weights_match": False,
    }
    if is_dup:
        log.warning("duplicate: similarity={:.6f} >= {} vs {}", best_sim, threshold, matched["key"])
    return result


def index_fingerprint(
    key: str,
    fp: dict,
    *,
    hotkey: str,
    repo: str,
    digest: str,
    model_uri: str,
    created_at: str,
    weights_hash: str | None = None,
) -> None:
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))
    body = {
        "key": key,
        "hotkey": hotkey,
        "repo": repo,
        "digest": digest,
        "model_uri": model_uri,
        "created_at": created_at,
        "norm_vector": vec,
        "fingerprint": fp,
    }
    if weights_hash:
        body["weights_hash"] = weights_hash
    get_client().index(index=index, id=key, body=body)
