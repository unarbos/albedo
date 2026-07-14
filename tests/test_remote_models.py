from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from albedo_eval_service import remote_models
from albedo_eval_service.canonical_model_config import canonical_max_model_len
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_models import ModelArtifactResolver, parse_oci_ref


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_parse_oci_ref_accepts_hippius_registry_digest():
    ref = "registry.hippius.com/sota1028/albedo-qwen3.6-35b-miner_5@sha256:" + "8" * 64

    assert parse_oci_ref(ref) == ("registry.hippius.com", "sota1028/albedo-qwen3.6-35b-miner_5", "sha256:" + "8" * 64)


def test_model_resolver_passes_through_existing_local_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"bad","max_position_embeddings":4096,"auto_map":{"x":"y"}}', encoding="utf-8")

    resolved = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(tmp_path / "cache"))).resolve(str(model_dir))

    assert resolved.local_path == str(model_dir)
    assert resolved.source == "local"
    assert resolved.file_count >= 2
    assert Path(resolved.local_path, "generation_config.json").exists()
    rewritten = json.loads(Path(resolved.local_path, "config.json").read_text(encoding="utf-8"))
    assert rewritten["model_type"] == "qwen3_5_moe"
    assert rewritten["text_config"]["max_position_embeddings"] == canonical_max_model_len()
    assert "auto_map" not in rewritten


def test_model_resolver_downloads_oci_layers_with_digest_verification(tmp_path, monkeypatch):
    layer_payload = b"{\n  \"model_type\": \"qwen3\"\n}\n"
    weights_payload = b"not-real-safetensors"
    layer_digest = _sha256(layer_payload)
    weights_digest = _sha256(weights_payload)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/octet-stream",
                "digest": layer_digest,
                "size": len(layer_payload),
                "annotations": {"org.opencontainers.image.title": "config.json"},
            },
            {
                "mediaType": "application/octet-stream",
                "digest": weights_digest,
                "size": len(weights_payload),
                "annotations": {"org.opencontainers.image.title": "model-00001-of-00001.safetensors"},
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
            if layer_digest in url:
                return httpx.Response(200, content=layer_payload, request=request)
            if weights_digest in url:
                return httpx.Response(200, content=weights_payload, request=request)
            raise AssertionError(url)

    monkeypatch.setattr("albedo_eval_service.remote_models.httpx.Client", lambda **_: FakeClient())
    ref = f"registry.hippius.com/sota1028/albedo-qwen3.6-35b-miner_5@{manifest_digest}"

    resolved = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(tmp_path / "cache"))).resolve(ref)

    rewritten = json.loads(Path(resolved.local_path, "config.json").read_text(encoding="utf-8"))
    assert rewritten["model_type"] == "qwen3_5_moe"
    assert rewritten["text_config"]["max_position_embeddings"] == canonical_max_model_len()
    assert resolved.source == "oci"
    assert resolved.cache_hit is False



def test_model_resolver_redownloads_marker_only_oci_cache(tmp_path, monkeypatch):
    config_payload = b"{\n  \"model_type\": \"qwen3\"\n}\n"
    weights_payload = b"not-real-safetensors"
    config_digest = _sha256(config_payload)
    weights_digest = _sha256(weights_payload)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/octet-stream",
                "digest": config_digest,
                "size": len(config_payload),
                "annotations": {"org.opencontainers.image.title": "config.json"},
            },
            {
                "mediaType": "application/octet-stream",
                "digest": weights_digest,
                "size": len(weights_payload),
                "annotations": {"org.opencontainers.image.title": "model-00001-of-00001.safetensors"},
            },
        ],
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_digest = _sha256(manifest_payload)
    cache_root = tmp_path / "cache"
    cache_dir = (
        cache_root
        / "oci"
        / "registry.hippius.com"
        / "sota1028__albedo-qwen3.6-35b-miner_5"
        / manifest_digest.removeprefix("sha256:")
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / ".albedo-model-cache.json").write_text("{}\n", encoding="utf-8")

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
                    headers={"www-authenticate": "Bearer realm=\"https://registry.hippius.com/service/token\",service=\"harbor-registry\""},
                    request=request,
                )
            if "/manifests/" in url:
                return httpx.Response(200, content=manifest_payload, request=request)
            if config_digest in url:
                return httpx.Response(200, content=config_payload, request=request)
            if weights_digest in url:
                return httpx.Response(200, content=weights_payload, request=request)
            raise AssertionError(url)

    monkeypatch.setattr("albedo_eval_service.remote_models.httpx.Client", lambda **_: FakeClient())
    ref = f"registry.hippius.com/sota1028/albedo-qwen3.6-35b-miner_5@{manifest_digest}"

    resolved = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(cache_root))).resolve(ref)

    assert resolved.cache_hit is False
    assert Path(resolved.local_path, "config.json").exists()
    assert Path(resolved.local_path, "model-00001-of-00001.safetensors").exists()


def test_model_resolver_rejects_oci_download_without_model_files(tmp_path, monkeypatch):
    manifest = {"schemaVersion": 2, "layers": []}
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_digest = _sha256(manifest_payload)
    cache_root = tmp_path / "cache"

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get(self, url, headers=None):
            request = httpx.Request("GET", url, headers=headers or {})
            if "/manifests/" in url:
                return httpx.Response(200, content=manifest_payload, request=request)
            raise AssertionError(url)

    monkeypatch.setattr("albedo_eval_service.remote_models.httpx.Client", lambda **_: FakeClient())
    ref = f"registry.hippius.com/sota1028/albedo-qwen3.6-35b-miner_5@{manifest_digest}"
    resolver = ModelArtifactResolver(
        RemoteSettings(model_cache_dir=str(cache_root), use_canonical_model_config=False)
    )

    with pytest.raises(FileNotFoundError, match="missing loadable model files"):
        resolver.resolve(ref)

    cache_dir = (
        cache_root
        / "oci"
        / "registry.hippius.com"
        / "sota1028__albedo-qwen3.6-35b-miner_5"
        / manifest_digest.removeprefix("sha256:")
    )
    assert not cache_dir.exists()


def test_resolve_lock_is_shared_per_ref():
    from albedo_eval_service.remote_models import _resolve_lock

    assert _resolve_lock("oci://registry/a@sha256:aa") is _resolve_lock("oci://registry/a@sha256:aa")
    assert _resolve_lock("oci://registry/a@sha256:aa") is not _resolve_lock("oci://registry/b@sha256:bb")


def test_model_resolver_downloads_hf_ref(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = {}

    def fake_snapshot_download(**kw):
        calls.update(kw)
        local_dir = Path(kw["local_dir"])
        (local_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"not-real-safetensors")
        return kw["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False)
    revision = "d" * 40
    ref = f"hf://alice/albedo-qwen3.6-35b-hf@{revision}"
    resolver = ModelArtifactResolver(
        RemoteSettings(
            model_cache_dir=str(tmp_path / "cache"),
            use_canonical_model_config=False,
            model_download_out_of_process=False,
        )
    )

    resolved = resolver.resolve(ref)

    assert calls["repo_id"] == "alice/albedo-qwen3.6-35b-hf"
    assert calls["revision"] == revision
    assert resolved.source == "hf"
    assert resolved.cache_hit is False
    assert Path(resolved.local_path, "config.json").exists()
    assert Path(resolved.local_path, ".albedo-model-cache.json").exists()

    again = resolver.resolve(ref)
    assert again.cache_hit is True
    assert again.local_path == resolved.local_path


def test_model_resolver_downloads_bare_hf_chain_ref(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = {}

    def fake_snapshot_download(**kw):
        calls.update(kw)
        local_dir = Path(kw["local_dir"])
        (local_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"not-real-safetensors")
        return kw["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False)
    revision = "d" * 40
    # Chain commits store "<repo>@<git-sha>" with no scheme; the resolver must treat it
    # as an HF ref, not pass it through to vLLM.
    ref = f"alice/albedo-qwen3.6-35b-hf@{revision}"
    resolver = ModelArtifactResolver(
        RemoteSettings(
            model_cache_dir=str(tmp_path / "cache"),
            use_canonical_model_config=False,
            model_download_out_of_process=False,
        )
    )

    resolved = resolver.resolve(ref)

    assert calls["repo_id"] == "alice/albedo-qwen3.6-35b-hf"
    assert calls["revision"] == revision
    assert resolved.source == "hf"
    assert resolved.original_ref == ref
    assert Path(resolved.local_path, "config.json").exists()

    prefixed = resolver.resolve(f"hf://{ref}")
    assert prefixed.cache_hit is True
    assert prefixed.local_path == resolved.local_path


def test_model_resolver_rejects_malformed_hf_ref(tmp_path):
    resolver = ModelArtifactResolver(RemoteSettings(model_cache_dir=str(tmp_path / "cache")))

    with pytest.raises(ValueError, match="hf:// ref must be"):
        resolver.resolve("hf://no-revision-here")


def test_download_supervisor_kills_stalled_child(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_models, "_HEARTBEAT_INTERVAL_S", 0.05)
    launched: list[subprocess.Popen] = []

    def fake_spawn(repo, revision, temp_dir, concurrency, log_path):
        # A child that never writes to temp_dir and refuses to exit — a wedged transfer.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            start_new_session=True,
        )
        launched.append(proc)
        return proc

    monkeypatch.setattr(remote_models, "_spawn_hf_download", fake_spawn)
    temp_dir = tmp_path / "m.partial"
    temp_dir.mkdir()

    with pytest.raises(TimeoutError, match="made no progress"):
        remote_models._run_hf_download_supervised(
            repo="ns/m",
            revision="d" * 40,
            temp_dir=temp_dir,
            label="ns/m",
            concurrency=1,
            stall_seconds=0.2,
            max_attempts=2,
        )

    assert len(launched) == 2  # stalled once, retried, then gave up
    for proc in launched:
        assert proc.poll() is not None  # every stalled child was terminated


def test_download_supervisor_raises_on_child_error(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_models, "_HEARTBEAT_INTERVAL_S", 0.05)

    def fake_spawn(repo, revision, temp_dir, concurrency, log_path):
        Path(log_path).write_text("RepositoryNotFoundError: 404\n", encoding="utf-8")
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])

    monkeypatch.setattr(remote_models, "_spawn_hf_download", fake_spawn)
    temp_dir = tmp_path / "m.partial"
    temp_dir.mkdir()

    with pytest.raises(RuntimeError, match="exited 3"):
        remote_models._run_hf_download_supervised(
            repo="ns/m",
            revision="d" * 40,
            temp_dir=temp_dir,
            label="ns/m",
            concurrency=1,
            stall_seconds=5.0,
            max_attempts=2,
        )


def test_download_supervisor_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_models, "_HEARTBEAT_INTERVAL_S", 0.05)

    def fake_spawn(repo, revision, temp_dir, concurrency, log_path):
        (Path(temp_dir) / "model.safetensors").write_bytes(b"x" * 1024)
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])

    monkeypatch.setattr(remote_models, "_spawn_hf_download", fake_spawn)
    temp_dir = tmp_path / "m.partial"
    temp_dir.mkdir()

    remote_models._run_hf_download_supervised(
        repo="ns/m",
        revision="d" * 40,
        temp_dir=temp_dir,
        label="ns/m",
        concurrency=1,
        stall_seconds=5.0,
        max_attempts=2,
    )
