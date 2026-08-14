import hashlib
from pathlib import Path

from model_validation.validate import genesis_files
from model_validation.validate.genesis_files import GENESIS_SHA256, check

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_hashes_match_committed_assets():
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert _sha256(_ASSETS / name) == GENESIS_SHA256[name], name


def test_chat_template_is_not_genesis_hashed():
    assert "chat_template.jinja" not in GENESIS_SHA256


def test_all_matching_passes(tmp_path, monkeypatch):
    files = {"a.json": b"alpha", "b.json": b"beta"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(
        genesis_files,
        "GENESIS_SHA256",
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    ok, msg = check(str(tmp_path), list(files))
    assert ok, msg


def test_tampered_file_rejected(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_bytes(b"tampered")
    monkeypatch.setattr(
        genesis_files, "GENESIS_SHA256", {"a.json": hashlib.sha256(b"genesis").hexdigest()}
    )
    ok, msg = check(str(tmp_path), ["a.json"])
    assert not ok
    assert "a.json sha256" in msg and "does not match genesis" in msg


def test_missing_file_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        genesis_files, "GENESIS_SHA256", {"a.json": hashlib.sha256(b"x").hexdigest()}
    )
    ok, msg = check(str(tmp_path), [])
    assert not ok
    assert "missing required genesis file a.json" in msg


def test_index_json_is_not_genesis_hashed():
    assert "model.safetensors.index.json" not in GENESIS_SHA256
