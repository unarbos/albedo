import hashlib
import json

import pytest

from albedo_eval_service.dataset_manifest import load_manifest_file


def test_load_manifest_file_verifies_sha256(tmp_path):
    manifest = {"shards": [{"name": "data/train-00000.parquet", "rows": 1}], "total_rows": 1}
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(payload)

    loaded = load_manifest_file(manifest_path, expected_sha256=hashlib.sha256(payload).hexdigest())

    assert loaded == manifest


def test_load_manifest_file_rejects_hash_mismatch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest_file(manifest_path, expected_sha256="0" * 64)
