import json
from pathlib import Path

from hippius_validation.validate.chat_template import check

_CANONICAL_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "tokenizers"
    / "Qwen3.6-35B-A3B"
    / "chat_template.jinja"
).read_text()


def _write_tokenizer_config(path, template):
    path.write_text(json.dumps({"chat_template": template}))


def test_canonical_chat_template_ok(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(_CANONICAL_TEMPLATE)
    _write_tokenizer_config(tmp_path / "tokenizer_config.json", _CANONICAL_TEMPLATE)

    ok, msg = check(str(tmp_path), ["chat_template.jinja", "tokenizer_config.json"])

    assert ok, msg


def test_bad_chat_template_file_rejected(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("bad")
    _write_tokenizer_config(tmp_path / "tokenizer_config.json", _CANONICAL_TEMPLATE)

    ok, msg = check(str(tmp_path), ["chat_template.jinja", "tokenizer_config.json"])

    assert not ok
    assert "chat_template.jinja sha256" in msg


def test_missing_chat_template_file_rejected(tmp_path):
    _write_tokenizer_config(tmp_path / "tokenizer_config.json", _CANONICAL_TEMPLATE)

    ok, msg = check(str(tmp_path), ["tokenizer_config.json"])

    assert not ok
    assert "missing required chat_template.jinja" in msg


def test_bad_tokenizer_config_template_rejected(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(_CANONICAL_TEMPLATE)
    _write_tokenizer_config(tmp_path / "tokenizer_config.json", "bad")

    ok, msg = check(str(tmp_path), ["chat_template.jinja", "tokenizer_config.json"])

    assert not ok
    assert "tokenizer_config.json chat_template sha256" in msg


def test_process_model_rejects_template_before_full_download(tmp_path, monkeypatch):
    from hippius_validation import validate_worker as worker

    calls = []
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "bad"}))

    monkeypatch.setattr(worker, "make_ref", lambda repo, digest: object())
    monkeypatch.setattr(
        worker,
        "list_files",
        lambda ref: [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
            "model.safetensors",
        ],
    )
    monkeypatch.setattr(worker, "safetensors_dtypes", lambda ref: {"model.safetensors": {"BF16"}})
    monkeypatch.setattr(
        worker, "download_config", lambda ref: calls.append("config") or str(tmp_path)
    )
    monkeypatch.setattr(worker, "download_full", lambda ref: calls.append("full") or str(tmp_path))

    outcome = worker.process_model("repo@sha256:abc", "hotkey")

    assert outcome.fault_code == "chat_template_hash"
    assert calls == ["config"]


def test_file_manifest_requires_chat_template():
    from hippius_validation.validate.repo import check as check_repo

    ok, msg = check_repo([
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "model.safetensors",
    ])

    assert not ok
    assert "chat_template.jinja" in msg
