from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from albedo_eval_service.canonical_model_config import canonical_max_model_len
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_models import ModelArtifactResolver, parse_oci_ref


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_parse_oci_ref_accepts_hippius_registry_digest():
    ref = "registry.hippius.com/sota1028/albedo-qwen3-4b-miner_5@sha256:" + "8" * 64

    assert parse_oci_ref(ref) == ("registry.hippius.com", "sota1028/albedo-qwen3-4b-miner_5", "sha256:" + "8" * 64)


def test_model_resolver_passes_through_existing_local_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"bad","max_position_embeddings":4096,"auto_map":{"x":"y"}}', encoding="utf-8")

    resolved = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(tmp_path / "cache"))).resolve(str(model_dir))

    assert resolved.local_path == str(model_dir)
    assert resolved.source == "local"
    assert resolved.file_count == 1
    rewritten = json.loads(Path(resolved.local_path, "config.json").read_text(encoding="utf-8"))
    assert rewritten["model_type"] == "qwen3"
    assert rewritten["max_position_embeddings"] == canonical_max_model_len()
    assert "auto_map" not in rewritten


def test_model_resolver_downloads_oci_layers_with_digest_verification(tmp_path, monkeypatch):
    layer_payload = b"{\n  \"model_type\": \"qwen3\"\n}\n"
    layer_digest = _sha256(layer_payload)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/octet-stream",
                "digest": layer_digest,
                "size": len(layer_payload),
                "annotations": {"org.opencontainers.image.title": "config.json"},
            }
        ],
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_digest = _sha256(manifest_payload)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def stream(self, method, url, headers=None):
            response = self.get(url, headers=headers)

            class ResponseContext:
                def __enter__(self):
                    return response

                def __exit__(self, exc_type, exc, tb):
                    return None

            return ResponseContext()

        def get(self, url, headers=None):
            request = httpx.Request("GET", url, headers=headers or {})
            if "/service/token" in url:
                return httpx.Response(200, json={"token": "token-1"}, request=request)
            if "/manifests/" in url and not (headers or {}).get("Authorization"):
                return httpx.Response(
                    401,
                    headers={"www-authenticate": 'Bearer realm="https://registry.hippius.com/service/token",service="harbor-registry"'},
                    request=request,
                )
            if "/manifests/" in url:
                return httpx.Response(200, content=manifest_payload, request=request)
            if "/blobs/" in url:
                return httpx.Response(200, content=layer_payload, request=request)
            raise AssertionError(url)

    monkeypatch.setattr("albedo_eval_service.remote_models.httpx.Client", lambda **_: FakeClient())
    ref = f"registry.hippius.com/sota1028/albedo-qwen3-4b-miner_5@{manifest_digest}"

    resolved = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(tmp_path / "cache"))).resolve(ref)

    rewritten = json.loads(Path(resolved.local_path, "config.json").read_text(encoding="utf-8"))
    assert rewritten["model_type"] == "qwen3"
    assert rewritten["max_position_embeddings"] == canonical_max_model_len()
    assert resolved.source == "oci"
    assert resolved.cache_hit is False
