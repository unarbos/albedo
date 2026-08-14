from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from config_validation.storage import _fastdl

_REV = "c" * 40


def _spec(name: str, payload: bytes) -> _fastdl.ShardSpec:
    return _fastdl.ShardSpec(name, len(payload), hashlib.sha256(payload).hexdigest())


def _install(dest: Path, spec: _fastdl.ShardSpec, payload: bytes) -> None:
    (dest / spec.name).write_bytes(payload)
    _fastdl._write_sidecar(dest, spec, _REV)


def test_plan_shards_filters_and_requires_lfs(monkeypatch):
    siblings = [
        SimpleNamespace(
            rfilename="model-00001-of-00002.safetensors",
            lfs=SimpleNamespace(size=10, sha256="a" * 64),
        ),
        SimpleNamespace(rfilename="config.json", lfs=None),
    ]
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda self, repo, revision, files_metadata: SimpleNamespace(siblings=siblings),
    )
    shards = _fastdl.plan_shards("ns/m", _REV, token=None)
    assert [s.name for s in shards] == ["model-00001-of-00002.safetensors"]

    siblings.append(SimpleNamespace(rfilename="model-00002-of-00002.safetensors", lfs=None))
    with pytest.raises(RuntimeError, match="no LFS metadata"):
        _fastdl.plan_shards("ns/m", _REV, token=None)


def test_fetch_skips_complete_files(tmp_path, monkeypatch):
    payload = b"weights"
    spec = _spec("model.safetensors", payload)
    _install(tmp_path, spec, payload)
    monkeypatch.setattr(_fastdl, "plan_shards", lambda *a, **k: [spec])

    def boom(*a, **k):
        raise AssertionError("complete file must not be re-fetched")

    monkeypatch.setattr(_fastdl, "_fetch_one", boom)
    _fastdl.fetch_shards("ns/m", _REV, tmp_path, token=None)


def test_fetch_redownloads_wrong_size_or_bad_sidecar(tmp_path, monkeypatch):
    payload = b"weights"
    spec = _spec("model.safetensors", payload)
    (tmp_path / spec.name).write_bytes(b"trunc")
    monkeypatch.setattr(_fastdl, "plan_shards", lambda *a, **k: [spec])
    fetched = []

    def fake_fetch(url, tmp, s, headers):
        fetched.append(s.name)
        Path(tmp).write_bytes(payload)

    monkeypatch.setattr(_fastdl, "_fetch_one", fake_fetch)
    _fastdl.fetch_shards("ns/m", _REV, tmp_path, token=None)
    assert fetched == [spec.name]
    assert (tmp_path / spec.name).read_bytes() == payload
    assert _fastdl._sidecar_ok(tmp_path, spec, _REV)


def test_fetch_retries_then_succeeds_on_sha_mismatch(tmp_path, monkeypatch):
    payload = b"weights"
    spec = _spec("model.safetensors", payload)
    monkeypatch.setattr(_fastdl, "plan_shards", lambda *a, **k: [spec])
    monkeypatch.setattr(_fastdl, "RETRY_BACKOFF_S", 0.0)
    calls = []

    def flaky_fetch(url, tmp, s, headers):
        calls.append(1)
        Path(tmp).write_bytes(b"wrights" if len(calls) == 1 else payload)

    monkeypatch.setattr(_fastdl, "_fetch_one", flaky_fetch)
    _fastdl.fetch_shards("ns/m", _REV, tmp_path, token=None)
    assert len(calls) == 2
    assert (tmp_path / spec.name).read_bytes() == payload
    assert not list(tmp_path.glob("*.fastdl"))


def test_fetch_raises_after_file_retries(tmp_path, monkeypatch):
    payload = b"weights"
    spec = _spec("model.safetensors", payload)
    monkeypatch.setattr(_fastdl, "plan_shards", lambda *a, **k: [spec])
    monkeypatch.setattr(_fastdl, "FILE_RETRIES", 2)
    monkeypatch.setattr(_fastdl, "RETRY_BACKOFF_S", 0.0)
    calls = []

    def bad_fetch(url, tmp, s, headers):
        calls.append(1)
        Path(tmp).write_bytes(b"wrights")

    monkeypatch.setattr(_fastdl, "_fetch_one", bad_fetch)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        _fastdl.fetch_shards("ns/m", _REV, tmp_path, token=None)
    assert len(calls) == 2


def test_sha256_heartbeat_writes_blocks_and_correct_digest(tmp_path):
    payload = b"x" * (1 << 20)
    target = tmp_path / "shard"
    target.write_bytes(payload)
    pulse = tmp_path / "verify-progress.fastdl"
    digest = _fastdl._sha256(target, heartbeat=pulse)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert pulse.stat().st_size >= 4096


def test_clean_stale_removes_temp_corpses(tmp_path):
    side = tmp_path / ".cache" / "huggingface" / "download"
    side.mkdir(parents=True)
    (side / "x.abcd1234.incomplete").write_bytes(b"dead")
    (tmp_path / "model.safetensors.fastdl").write_bytes(b"dead")
    keeper = side / "model.safetensors.metadata"
    keeper.write_text("keep")
    _fastdl._clean_stale(tmp_path)
    assert not list(side.glob("*.incomplete"))
    assert not list(tmp_path.glob("*.fastdl"))
    assert keeper.exists()
