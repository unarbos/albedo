import json
import struct

from model_validation.validate.safetensors_index import check


def _write_shard(path, tensor_keys):
    """Write a minimal valid safetensors file declaring tensor_keys (zero-size tensors)."""
    header = {k: {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]} for k in tensor_keys}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _write_index(path, weight_map):
    path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_single_file_no_index_ok(tmp_path):
    _write_shard(tmp_path / "model.safetensors", ["model.embed_tokens.weight"])
    ok, msg = check(str(tmp_path), ["model.safetensors"])
    assert ok, msg


def test_sharded_without_index_fails(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", ["a"])
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", ["b"])
    ok, msg = check(str(tmp_path), [])
    assert not ok and "missing model.safetensors.index.json" in msg


def test_clean_sharded_ok(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", ["a", "b"])
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", ["c"])
    _write_index(tmp_path / "model.safetensors.index.json", {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00001-of-00002.safetensors",
        "c": "model-00002-of-00002.safetensors",
    })
    ok, msg = check(str(tmp_path), [])
    assert ok, msg


def test_extra_unreferenced_shard_fails(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", ["a"])
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", ["b"])  # not in index
    _write_index(tmp_path / "model.safetensors.index.json", {
        "a": "model-00001-of-00001.safetensors",
    })
    ok, msg = check(str(tmp_path), [])
    assert not ok and "not used by the model" in msg
    assert "model-00002-of-00002.safetensors" in msg


def test_missing_referenced_shard_fails(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", ["a"])
    _write_index(tmp_path / "model.safetensors.index.json", {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",  # file absent
    })
    ok, msg = check(str(tmp_path), [])
    assert not ok and "references missing shard" in msg


def test_dead_tensor_in_shard_fails(tmp_path):
    _write_shard(tmp_path / "model.safetensors", ["a", "unused"])  # 'unused' not in index
    _write_index(tmp_path / "model.safetensors.index.json", {"a": "model.safetensors"})
    ok, msg = check(str(tmp_path), [])
    assert not ok and "not referenced by the index" in msg
    assert "unused" in msg


def test_index_maps_missing_tensor_fails(tmp_path):
    _write_shard(tmp_path / "model.safetensors", ["a"])
    _write_index(tmp_path / "model.safetensors.index.json", {
        "a": "model.safetensors",
        "ghost": "model.safetensors",  # not in file header
    })
    ok, msg = check(str(tmp_path), [])
    assert not ok and "not present in shard" in msg
    assert "ghost" in msg


def test_malformed_index_fails(tmp_path):
    _write_shard(tmp_path / "model.safetensors", ["a"])
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    ok, msg = check(str(tmp_path), [])
    assert not ok and "malformed" in msg
