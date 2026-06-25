from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from scripts.king_hf_uploader import (
    _UPLOAD_IGNORE_PATTERNS,
    KingUpload,
    Settings,
    _delete_work_copy,
    _iter_model_files,
    _matches_qwen35,
    _missing_layers,
    _upload_model,
    _verify_and_repair,
    already_uploaded,
    eval_dir_path,
    hf_repo_problems,
    hub_repo_url,
    list_crowned_kings,
    model_repo,
    render_albedo_md,
    repo_id_for,
    to_roman,
    work_dir_path,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://example",
        hf_namespace="kigs",
        hf_token="token",
        eval_dir=tmp_path / "eval",
        work_dir=tmp_path / "work",
        lock_path=tmp_path / "lock",
        repo_prefix="albedo-qwen3.6-35b-king",
        poll_interval_s=30.0,
        qwen_patterns=("qwen3.6", "qwen3-6", "qwen3_6"),
        size_patterns=("35b", "35-b"),
        genesis_markers=("qwen3.6-35b-a3b-genesis", "35b-a3b-genesis"),
        force=False,
        verify=False,
        dry_run=False,
    )


class _FakeApi:
    """Minimal HfApi stand-in that records create_repo/create_commit calls."""

    def __init__(self, *, exists: bool = False, files: list[str] | None = None):
        self._exists = exists
        self._files = files or []
        self.created_repos: list[str] = []
        # Each commit recorded as (repo_id, [path_in_repo, ...], commit_message).
        self.commits: list[tuple[str, list[str], str]] = []

    def repo_exists(self, repo_id, repo_type="model"):
        return self._exists

    def list_repo_files(self, repo_id, repo_type="model"):
        return list(self._files)

    def create_repo(self, repo_id, repo_type="model", private=False, exist_ok=True):
        self.created_repos.append(repo_id)

    def create_commit(self, repo_id, repo_type="model", operations=None, commit_message=""):
        self.commits.append(
            (repo_id, [op.path_in_repo for op in (operations or [])], commit_message)
        )


def _make_model_dir(base: Path) -> Path:
    """A model snapshot mixing real files with internal junk the uploader must ignore."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_text("weights", encoding="utf-8")
    (base / ".albedo-model-cache.json").write_text("{}", encoding="utf-8")  # also the done marker
    (base / "shard.download").write_text("partial", encoding="utf-8")
    cache = base / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "meta").write_text("x", encoding="utf-8")
    return base


def _oci_king(eval_or_work_dir: Path, digest_char: str) -> tuple[KingUpload, Path]:
    digest = digest_char * 64
    king = _king(model_uri="registry.hippius.com/alice/albedo-qwen3.6-35b-v1@sha256:" + digest)
    model_dir = (
        eval_or_work_dir
        / "oci"
        / "registry.hippius.com"
        / "alice__albedo-qwen3.6-35b-v1"
        / digest
    )
    return king, model_dir


def _king(*, model_uri: str, roman: str = "I", reign_reason: str = "CORONATION") -> KingUpload:
    return KingUpload(
        king_version_id=uuid4(),
        king_version=7,
        model_hash="sha256:" + "a" * 64,
        model_uri=model_uri,
        artifact_uri=model_uri,
        architecture="Qwen3_5MoeForConditionalGeneration",
        parameter_count=None,
        uid=12,
        hotkey="5EvHrbHz8rT8DrWazxFhzfMsmscFtPE3qhRDeY4ggKZrBcxZ",
        activated_at=datetime(2026, 6, 23, tzinfo=UTC),
        reign_reason=reign_reason,
        roman=roman,
    )


def _row(version: int, uri: str, reason: str) -> dict:
    return {
        "king_version_id": uuid4(),
        "king_version": version,
        "model_hash": "sha256:" + "a" * 64,
        "activated_at": datetime(2026, 6, 23, tzinfo=UTC),
        "reign_reason": reason,
        "model_uri": uri,
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "parameter_count": None,
        "uid": 1,
        "hotkey": "hotkey",
        "artifact_uri": uri,
    }


class _Result:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def fetchall(self) -> list[dict]:
        return self._rows


class _Conn:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def execute(self, *args, **kwargs) -> _Result:
        return _Result(self._rows)


def test_to_roman():
    assert [to_roman(n) for n in (1, 4, 5, 9, 14, 40, 90)] == [
        "I", "IV", "V", "IX", "XIV", "XL", "XC"
    ]


def test_model_repo_strips_scheme_registry_and_digest():
    assert (
        model_repo("registry.hippius.com/alice/albedo-qwen3.6-35b-v1@sha256:" + "b" * 64)
        == "alice/albedo-qwen3.6-35b-v1"
    )
    assert model_repo("oci://alice/foo") == "alice/foo"
    assert model_repo("alice/foo") == "alice/foo"


def test_hub_repo_url():
    assert (
        hub_repo_url("registry.hippius.com/alice/albedo-qwen3.6-35b-v1@sha256:" + "b" * 64)
        == "https://hub.hippius.com/models/alice/albedo-qwen3.6-35b-v1"
    )


def test_matches_only_qwen36_35b(tmp_path: Path):
    settings = _settings(tmp_path)
    assert _matches_qwen35(_king(model_uri="alice/albedo-qwen3.6-35b-v1"), settings)
    assert not _matches_qwen35(_king(model_uri="alice/albedo-qwen3-4b-genesis"), settings)
    assert not _matches_qwen35(_king(model_uri="alice/llama-35b"), settings)


def test_repo_id_uses_roman_suffix(tmp_path: Path):
    settings = _settings(tmp_path)
    king = _king(model_uri="registry.hippius.com/alice/albedo-qwen3.6-35b-v1", roman="IV")
    assert repo_id_for(king, settings) == "kigs/albedo-qwen3.6-35b-king-IV"


def test_numbering_skips_genesis_and_survives_global_version_offset(tmp_path: Path):
    # kv.version is global: a 4B line (v1 genesis, v2 king) precedes the 35B line.
    rows = [
        _row(1, "registry.hippius.com/teutonic/albedo-qwen3-4b-genesis", "GENESIS"),
        _row(2, "registry.hippius.com/alice/albedo-qwen3-4b-v1", "CORONATION"),
        _row(3, "registry.hippius.com/teutonic/qwen3.6-35b-a3b-genesis", "GENESIS"),
        _row(4, "registry.hippius.com/alice/albedo-qwen3.6-35b-v1", "CORONATION"),
        _row(5, "registry.hippius.com/bob/albedo-qwen3.6-35b-v2", "CORONATION"),
        _row(6, "registry.hippius.com/carol/albedo-qwen3.6-35b-v3", "CORONATION"),
    ]
    kings = list_crowned_kings(_Conn(rows), _settings(tmp_path))
    assert [k.roman for k in kings] == ["I", "II", "III"]
    assert [k.king_version for k in kings] == [4, 5, 6]
    assert [repo_id_for(k, _settings(tmp_path)) for k in kings] == [
        "kigs/albedo-qwen3.6-35b-king-I",
        "kigs/albedo-qwen3.6-35b-king-II",
        "kigs/albedo-qwen3.6-35b-king-III",
    ]
    # King I dethroned the genesis seed; each later king dethroned the one before it.
    assert kings[0].opponent_name == "the genesis seed model"
    assert kings[0].opponent_repo == "teutonic/qwen3.6-35b-a3b-genesis"
    assert [k.opponent_name for k in kings] == [
        "the genesis seed model",
        "King I",
        "King II",
    ]
    assert kings[2].opponent_repo == "bob/albedo-qwen3.6-35b-v2"


def test_render_albedo_md_names_the_dethroned_model(tmp_path: Path):
    rows = [
        _row(3, "registry.hippius.com/teutonic/qwen3.6-35b-a3b-genesis", "GENESIS"),
        _row(4, "registry.hippius.com/alice/albedo-qwen3.6-35b-v1", "CORONATION"),
    ]
    king_one = list_crowned_kings(_Conn(rows), _settings(tmp_path))[0]
    doc = render_albedo_md(king_one)
    assert "Dethroned:" in doc
    assert "the genesis seed model" in doc
    assert "https://hub.hippius.com/models/teutonic/qwen3.6-35b-a3b-genesis" in doc


def test_numbering_skips_genesis_by_marker_even_if_reason_wrong(tmp_path: Path):
    rows = [
        _row(3, "registry.hippius.com/teutonic/qwen3.6-35b-a3b-genesis", "CORONATION"),
        _row(4, "registry.hippius.com/alice/albedo-qwen3.6-35b-v1", "CORONATION"),
    ]
    kings = list_crowned_kings(_Conn(rows), _settings(tmp_path))
    assert [k.roman for k in kings] == ["I"]
    assert kings[0].hippius_repo == "alice/albedo-qwen3.6-35b-v1"


def test_eval_dir_path_detects_done_marker(tmp_path: Path):
    settings = _settings(tmp_path)
    king = _king(
        model_uri="registry.hippius.com/alice/albedo-qwen3.6-35b-v1@sha256:" + "c" * 64
    )
    model_dir = (
        settings.eval_dir
        / "oci"
        / "registry.hippius.com"
        / "alice__albedo-qwen3.6-35b-v1"
        / ("c" * 64)
    )
    model_dir.mkdir(parents=True)
    assert eval_dir_path(king, settings) is None  # no done marker yet
    (model_dir / ".albedo-model-cache.json").write_text("{}", encoding="utf-8")
    assert eval_dir_path(king, settings) == model_dir


def test_delete_work_copy_refuses_outside_work_dir(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    eval_model = tmp_path / "eval" / "oci" / "model"
    eval_model.mkdir(parents=True)
    (eval_model / "config.json").write_text("{}", encoding="utf-8")

    _delete_work_copy(eval_model, work_dir)  # must be a no-op
    assert eval_model.exists()

    work_model = work_dir / "oci" / "model"
    work_model.mkdir(parents=True)
    (work_model / "config.json").write_text("{}", encoding="utf-8")
    _delete_work_copy(work_model, work_dir)
    assert not work_model.exists()


def test_render_albedo_md_has_dynamic_fields():
    king = _king(model_uri="registry.hippius.com/alice/albedo-qwen3.6-35b-v1", roman="IV")
    doc = render_albedo_md(king)
    assert "King IV" in doc
    assert "alice/albedo-qwen3.6-35b-v1" in doc
    assert "https://hub.hippius.com/models/alice/albedo-qwen3.6-35b-v1" in doc
    assert king.hotkey in doc


def test_work_dir_path_detects_done_marker(tmp_path: Path):
    settings = _settings(tmp_path)
    king, model_dir = _oci_king(settings.work_dir, "e")
    model_dir.mkdir(parents=True)
    assert work_dir_path(king, settings) is None  # no done marker yet
    (model_dir / ".albedo-model-cache.json").write_text("{}", encoding="utf-8")
    assert work_dir_path(king, settings) == model_dir


def test_iter_model_files_excludes_internal(tmp_path: Path):
    model_dir = _make_model_dir(tmp_path / "m")
    (model_dir / "albedo.md").write_text("doc", encoding="utf-8")  # our own doc, never re-pushed
    assert set(_iter_model_files(model_dir, _UPLOAD_IGNORE_PATTERNS)) == {
        "config.json",
        "model.safetensors",
    }


def test_already_uploaded_treats_only_gitattributes_as_empty():
    assert already_uploaded(_FakeApi(exists=False), "kigs/x") is False
    assert already_uploaded(_FakeApi(exists=True, files=[".gitattributes"]), "kigs/x") is False
    assert already_uploaded(_FakeApi(exists=True, files=[]), "kigs/x") is False
    assert (
        already_uploaded(_FakeApi(exists=True, files=[".gitattributes", "config.json"]), "kigs/x")
        is True
    )


def test_upload_model_makes_one_commit_with_albedo_md(tmp_path: Path):
    settings = _settings(tmp_path)
    model_dir = _make_model_dir(tmp_path / "m")
    king = _king(model_uri="registry.hippius.com/alice/albedo-qwen3.6-35b-v1", roman="IV")
    api = _FakeApi()
    _upload_model(api, king, model_dir, settings, "kigs/albedo-qwen3.6-35b-king-IV")
    # Two commits total: create_repo seeds the initial .gitattributes commit ...
    assert api.created_repos == ["kigs/albedo-qwen3.6-35b-king-IV"]
    # ... and a single create_commit lands every file at once.
    assert len(api.commits) == 1
    _, paths, _ = api.commits[0]
    assert "albedo.md" in paths  # committed together with the miner's files
    assert {"config.json", "model.safetensors"}.issubset(set(paths))
    assert ".albedo-model-cache.json" not in paths
    assert "shard.download" not in paths


def test_verify_and_repair_commits_only_missing_files(tmp_path: Path):
    settings = _settings(tmp_path)
    king, model_dir = _oci_king(settings.eval_dir, "c")
    _make_model_dir(model_dir)  # local source is complete (config.json + model.safetensors)
    # Repo already has config.json but is missing model.safetensors and albedo.md.
    api = _FakeApi(exists=True, files=[".gitattributes", "config.json"])
    assert _verify_and_repair(api, king, settings, "kigs/x") is True
    assert len(api.commits) == 1
    _, paths, _ = api.commits[0]
    assert set(paths) == {"model.safetensors", "albedo.md"}


def test_verify_and_repair_is_noop_when_repo_complete(tmp_path: Path):
    settings = _settings(tmp_path)
    king, model_dir = _oci_king(settings.eval_dir, "d")
    _make_model_dir(model_dir)
    api = _FakeApi(
        exists=True, files=[".gitattributes", "config.json", "model.safetensors", "albedo.md"]
    )
    assert _verify_and_repair(api, king, settings, "kigs/x") is False
    assert api.commits == []


def test_hf_repo_problems_flags_missing_and_passes_complete():
    assert hf_repo_problems(_FakeApi(exists=False), "kigs/x", "tok") == ["repo does not exist"]
    # unsharded + complete -> no problems
    assert (
        hf_repo_problems(
            _FakeApi(exists=True, files=[".gitattributes", "config.json", "model.safetensors", "albedo.md"]),
            "kigs/x",
            "tok",
        )
        == []
    )
    # bare repo -> config + weights + albedo all flagged
    probs = hf_repo_problems(_FakeApi(exists=True, files=[".gitattributes"]), "kigs/x", "tok")
    assert "config.json" in probs and "albedo.md" in probs
    assert any("safetensors" in p for p in probs)
    # multiple shards but no index map -> index flagged
    probs = hf_repo_problems(
        _FakeApi(
            exists=True,
            files=[
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "albedo.md",
            ],
        ),
        "kigs/x",
        "tok",
    )
    assert "model.safetensors.index.json" in probs


def test_hf_repo_problems_detects_missing_shard_via_index(tmp_path: Path, monkeypatch):
    import json

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(index))
    # shard 2 of 2 is missing from the repo
    api = _FakeApi(
        exists=True,
        files=[
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "albedo.md",
        ],
    )
    assert hf_repo_problems(api, "kigs/x", "tok") == ["model-00002-of-00002.safetensors"]


def _layer(title: str, digest_char: str) -> dict:
    return {
        "digest": "sha256:" + digest_char * 64,
        "annotations": {"org.opencontainers.image.title": title},
    }


def test_missing_layers_picks_only_absent_non_ignored_files():
    manifest = {
        "layers": [
            _layer("config.json", "1"),
            _layer("model-00001-of-00002.safetensors", "2"),
            _layer("model-00002-of-00002.safetensors", "3"),
            _layer(".albedo-model-cache.json", "4"),  # internal: must be ignored
        ]
    }
    present = {"config.json", "model-00001-of-00002.safetensors", ".gitattributes"}
    assert _missing_layers(manifest, present, _UPLOAD_IGNORE_PATTERNS) == [
        ("model-00002-of-00002.safetensors", "sha256:" + "3" * 64),
    ]
