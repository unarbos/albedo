"""Fingerprint deduplication over OpenSearch.

Two-stage, so it both scales and uses the full fingerprint (norms + per-tensor samples):
  1. We compute the fingerprint (norm_vector + tensor_samples) with our own scripts.
  2. kNN-query OpenSearch (cosine on norm_vector) for the nearest existing models — fast
     prefilter — skipping the submitter's own prior models. Per-architecture index, so models
     with a different tensor count are never compared.
  3. Rerank candidates with the exact tensor-based similarity (config_validation.similarity:
     fraction of per-tensor sample vectors whose cosine is ~1). This uses the TENSORS, not just
     the norms.
     - If the best similarity is NOT at/above the threshold (0.95): not a duplicate → index it.
     - If it is: duplicate → return the reason, matched model, and similarity score.
"""
from __future__ import annotations

from loguru import logger as log

from config_validation.fingerprint import similarity

from hippius_validation import config
from hippius_validation.opensearch.client import ensure_index, get_client


def find_duplicate(fp: dict, hotkey: str, threshold: float | None = None,
                   k: int | None = None) -> dict:
    """Decide duplicate-or-not. Returns a result dict:
      {is_duplicate, similarity, threshold, matched_key, matched_hotkey, matched_model_uri,
       candidates_checked}
    """
    threshold = config.SIM_THRESHOLD if threshold is None else threshold
    k = config.KNN_CANDIDATES if k is None else k
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))   # per-dimension index

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
            continue  # a miner's own prior model is not a duplicate of itself
        sim = similarity(fp, src.get("fingerprint", {}))   # exact, tensor-based
        if sim > best_sim:
            best_sim = sim
            matched = {"key": src.get("key", ""), "hotkey": src.get("hotkey", ""),
                       "model_uri": src.get("model_uri", "")}

    is_dup = best_sim >= threshold
    result = {
        "is_duplicate": is_dup,
        "similarity": best_sim,
        "threshold": threshold,
        "matched_key": matched["key"] if is_dup else "",
        "matched_hotkey": matched["hotkey"] if is_dup else "",
        "matched_model_uri": matched["model_uri"] if is_dup else "",
        "candidates_checked": len(hits),
    }
    if is_dup:
        log.warning("duplicate: similarity={:.6f} >= {} vs {}", best_sim, threshold, matched["key"])
    return result


def index_fingerprint(key: str, fp: dict, *, hotkey: str, repo: str, digest: str,
                      model_uri: str, created_at: str) -> None:
    """Index a non-duplicate model's fingerprint into the per-dimension corpus (id=key)."""
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))
    get_client().index(
        index=index,
        id=key,
        body={
            "key": key, "hotkey": hotkey, "repo": repo, "digest": digest,
            "model_uri": model_uri, "created_at": created_at,
            "norm_vector": vec,
            "fingerprint": fp,
        },
    )
