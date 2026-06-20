import json
import struct

from hippius_validation.validate.dtype import check


def _write_shard(path, dtypes_by_key):
    """Write a minimal safetensors file whose header declares the given dtypes."""
    header = {k: {"dtype": dt, "shape": [0], "data_offsets": [0, 0]}
              for k, dt in dtypes_by_key.items()}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_bf16_ok(tmp_path):
    _write_shard(tmp_path / "model.safetensors", {"a": "BF16", "b": "BF16"})
    ok, msg = check(str(tmp_path))
    assert ok, msg


def test_fp16_ok(tmp_path):
    _write_shard(tmp_path / "model.safetensors", {"a": "F16"})
    ok, msg = check(str(tmp_path))
    assert ok, msg


def test_fp32_rejected(tmp_path):
    _write_shard(tmp_path / "model.safetensors", {"a": "BF16", "b": "F32"})
    ok, msg = check(str(tmp_path))
    assert not ok and "16-bit" in msg and "F32" in msg


def test_int8_quantized_rejected(tmp_path):
    _write_shard(tmp_path / "model.safetensors", {"a": "I8"})
    ok, msg = check(str(tmp_path))
    assert not ok and "I8" in msg


def test_uint8_quantized_rejected(tmp_path):
    _write_shard(tmp_path / "model.safetensors", {"a": "U8"})
    ok, msg = check(str(tmp_path))
    assert not ok and "U8" in msg


def test_bad_shard_among_many_reported(tmp_path):
    _write_shard(tmp_path / "model-00001-of-00002.safetensors", {"a": "BF16"})
    _write_shard(tmp_path / "model-00002-of-00002.safetensors", {"b": "F32"})
    ok, msg = check(str(tmp_path))
    assert not ok and "model-00002-of-00002.safetensors" in msg
