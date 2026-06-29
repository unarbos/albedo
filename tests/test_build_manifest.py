import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.sampling import multi_source_manifest_sample_ids


def _load_build_manifest():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(data_dir: Path, name: str, rows: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"messages": [["x"]] * rows}), data_dir / name)


def test_parse_weights():
    bm = _load_build_manifest()
    assert bm._parse_weights("swe-zero=0.7,mini-coder=0.3") == {"swe-zero": 0.7, "mini-coder": 0.3}


def test_build_source_counts_rows_and_namespaces(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "swe-zero" / "data", "train-00000.parquet", 3)
    _write_shard(tmp_path / "swe-zero" / "data", "train-00001.parquet", 2)

    source = bm._build_source("swe-zero", 0.7, tmp_path)

    assert source["name"] == "swe-zero"
    assert source["repo"] == "AlienKevin/SWE-ZERO-12M-trajectories"
    assert source["shard_glob"] == "data/train-*.parquet"
    assert source["weight"] == 0.7
    assert source["total_rows"] == 5
    assert [s["path"] for s in source["shards"]] == [
        "swe-zero/data/train-00000.parquet",
        "swe-zero/data/train-00001.parquet",
    ]
    assert all(len(s["sha256"]) == 64 for s in source["shards"])


def test_built_manifest_is_sampler_compatible(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "swe-zero" / "data", "train-00000.parquet", 100)
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000-of-00060.parquet", 100)

    sources = [
        bm._build_source("swe-zero", 0.7, tmp_path),
        bm._build_source("mini-coder", 0.3, tmp_path),
    ]
    manifest = {"version": "t", "sources": sources, "total_rows": 200}

    ids = multi_source_manifest_sample_ids(
        manifest, block_hash="0xabc", sample_count=10, max_turns_per_sample=1
    )
    assert len(ids) == 10
    assert sum(1 for i in ids if i.startswith("swe-zero/")) == 7
    assert sum(1 for i in ids if i.startswith("mini-coder/")) == 3
