"""weights_hash: backend-independent exact-content dedup gate ahead of the kNN prefilter."""
from __future__ import annotations

import pytest


def test_weights_hash_is_content_only(tmp_path):
    pytest.importorskip("asyncpg")
    from model_validation.validate_worker import _weights_hash

    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    # Same bytes under different shard names (and an extra non-weight file) → same hash.
    (a / "model-00001-of-00001.safetensors").write_bytes(b"weights-1")
    (b / "renamed.safetensors").write_bytes(b"weights-1")
    (b / "config.json").write_text("{}", encoding="utf-8")
    (c / "model-00001-of-00001.safetensors").write_bytes(b"weights-2")
    assert _weights_hash(str(a)) == _weights_hash(str(b))
    assert _weights_hash(str(a)) != _weights_hash(str(c))


class _FakeClient:
    def __init__(self, exact_hits, knn_hits):
        self.exact_hits = exact_hits
        self.knn_hits = knn_hits
        self.exact_bodies = []

    def search(self, index, body):
        if "bool" in body["query"]:
            self.exact_bodies.append(body)
            return {"hits": {"hits": self.exact_hits}}
        return {"hits": {"hits": self.knn_hits}}


def _patch_opensearch(monkeypatch, client):
    from model_validation.opensearch import fingerprints

    monkeypatch.setattr(fingerprints, "ensure_index", lambda dim: "idx")
    monkeypatch.setattr(fingerprints, "get_client", lambda: client)
    return fingerprints


def test_exact_weights_match_beats_saturated_knn(monkeypatch):
    pytest.importorskip("opensearchpy")
    exact_hit = {"_source": {"key": "ns/orig@" + "a" * 40, "hotkey": "hk-orig",
                             "model_uri": "ns/orig@" + "a" * 40}}
    client = _FakeClient(exact_hits=[exact_hit], knn_hits=[])
    fingerprints = _patch_opensearch(monkeypatch, client)

    result = fingerprints.find_duplicate(
        {"norm_vector": [1.0, 2.0]}, "hk-copycat", weights_hash="w" * 64
    )

    assert result["is_duplicate"] is True
    assert result["exact_weights_match"] is True
    assert result["similarity"] == 1.0
    assert result["matched_hotkey"] == "hk-orig"
    # The gate must skip the submitter's own prior models, like the kNN rerank does.
    assert {"term": {"hotkey": "hk-copycat"}} in client.exact_bodies[0]["query"]["bool"]["must_not"]


def test_no_exact_match_falls_through_to_knn(monkeypatch):
    pytest.importorskip("opensearchpy")
    client = _FakeClient(exact_hits=[], knn_hits=[])
    fingerprints = _patch_opensearch(monkeypatch, client)

    result = fingerprints.find_duplicate(
        {"norm_vector": [1.0, 2.0]}, "hk-x", weights_hash="w" * 64
    )

    assert result["is_duplicate"] is False
    assert result["exact_weights_match"] is False


def test_index_fingerprint_stores_weights_hash(monkeypatch):
    pytest.importorskip("opensearchpy")
    from model_validation.opensearch import fingerprints

    indexed = {}

    class _Idx:
        def index(self, index, id, body):
            indexed.update(body)

    monkeypatch.setattr(fingerprints, "ensure_index", lambda dim: "idx")
    monkeypatch.setattr(fingerprints, "get_client", lambda: _Idx())
    fingerprints.index_fingerprint(
        "k", {"norm_vector": [1.0]}, hotkey="hk", repo="ns/m", digest="d",
        model_uri="ns/m@d", created_at="2026-07-10T00:00:00Z", weights_hash="w" * 64,
    )
    assert indexed["weights_hash"] == "w" * 64
