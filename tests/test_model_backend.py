"""Backend detection, HF/Hippius dispatch, preflight, and manifest-URI routing."""
import pytest

from config_validation.models import BACKEND_HF, BACKEND_HIPPIUS, ModelRef, detect_backend

_SHA256 = "sha256:" + "a" * 64
_GIT_SHA1 = "b" * 40
_GIT_SHA256 = "c" * 64


def test_detect_backend():
    assert detect_backend(_SHA256) == BACKEND_HIPPIUS
    assert detect_backend(_GIT_SHA1) == BACKEND_HF
    assert detect_backend(_GIT_SHA256) == BACKEND_HF
    assert detect_backend("main") is None
    assert detect_backend("v1.0") is None


def test_modelref_backend_inferred():
    assert ModelRef("ns/m", _SHA256).backend == BACKEND_HIPPIUS
    assert ModelRef("ns/m", _GIT_SHA1).backend == BACKEND_HF
    assert ModelRef("ns/m", _GIT_SHA256).backend == BACKEND_HF


def test_modelref_rejects_mutable_by_default(monkeypatch):
    monkeypatch.delenv("ALBEDO_ALLOW_MUTABLE_REF", raising=False)
    with pytest.raises(ValueError):
        ModelRef("ns/m", "main")


def test_modelref_allows_mutable_when_enabled(monkeypatch):
    monkeypatch.setenv("ALBEDO_ALLOW_MUTABLE_REF", "1")
    monkeypatch.setenv("ALBEDO_MODEL_BACKEND", "hf")
    assert ModelRef("ns/m", "main").backend == BACKEND_HF


def test_modelref_explicit_backend_contradiction():
    with pytest.raises(ValueError):
        ModelRef("ns/m", _SHA256, backend=BACKEND_HF)  # a sha256 digest cannot be hf


def test_cache_dir_namespaced_no_collision():
    from config_validation.storage import cache_dir

    hf = cache_dir(ModelRef("ns/m", _GIT_SHA1))
    hp = cache_dir(ModelRef("ns/m", _SHA256))
    assert hf != hp
    assert "hf" in hf.parts and "hippius" in hp.parts


def test_dispatch_selects_backend():
    from config_validation.storage import _hf, _hippius, dispatch

    assert dispatch._impl(ModelRef("ns/m", _GIT_SHA1)) is _hf
    assert dispatch._impl(ModelRef("ns/m", _SHA256)) is _hippius


def test_hf_download_and_list(monkeypatch, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import config_validation.storage._hf as hf
    from config_validation.storage import _supervise

    calls = {}
    # Exercise the in-process download so the monkeypatched snapshot_download is reached.
    monkeypatch.setattr(_supervise, "OUT_OF_PROCESS", False)
    monkeypatch.setattr(hf, "_cache_dir", lambda ref: tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda **kw: calls.update(kw) or kw["local_dir"], raising=False,
    )
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda **kw: ["config.json", "model.safetensors"], raising=False,
    )
    from config_validation.storage import download_full, list_files

    ref = ModelRef("ns/m", _GIT_SHA1)
    download_full(ref)
    assert calls["repo_id"] == "ns/m" and calls["revision"] == _GIT_SHA1
    assert list_files(ref) == ["config.json", "model.safetensors"]


def test_hf_preflight_dtypes(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import model_validation.storage.preflight as pf

    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda **kw: ["model.safetensors", "config.json"], raising=False,
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda **kw: "https://hf/resolve/x", raising=False)
    monkeypatch.setattr(
        pf, "_read_header",
        lambda client, url, headers: {"w": {"dtype": "BF16"}, "__metadata__": {}},
    )
    out = pf.safetensors_dtypes(ModelRef("ns/m", _GIT_SHA1))
    assert out == {"model.safetensors": {"BF16"}}


def test_manifest_uri_routing():
    pytest.importorskip("asyncpg")
    from model_validation.db import _model_manifest_uri, _storage_backend_for_model_uri

    hf_uri = f"ns/m@{_GIT_SHA1}"
    hp_uri = f"ns/m@{_SHA256}"
    assert _model_manifest_uri(hf_uri) == f"hf://{hf_uri}"
    assert _model_manifest_uri(hp_uri) == f"registry.hippius.com/{hp_uri}"
    assert _storage_backend_for_model_uri(_model_manifest_uri(hf_uri)) == "hf"
    assert _storage_backend_for_model_uri(_model_manifest_uri(hp_uri)) == "hippius"
