import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.sampling import BUCKETS, multi_source_manifest_sample_ids

_BUCKET_TOTAL = sum(count for _, count in BUCKETS)  # 128


def _load_build_manifest():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(data_dir: Path, name: str, rows: int, *, asst: int = 12) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    # each row: a unique instance_id + a conversation with `asst` assistant turns (deep enough for
    # every bucket), so the enriched manifest carries {iid, asst} the sampler needs.
    conversation = [{"role": "system", "content": "s"}] + [
        {"role": role, "content": "c"} for _ in range(asst) for role in ("user", "assistant")
    ]
    table = pa.table(
        {
            "instance_id": [f"{name}-{i}" for i in range(rows)],
            "messages": [conversation for _ in range(rows)],
        }
    )
    pq.write_table(table, data_dir / name)


def test_parse_weights():
    bm = _load_build_manifest()
    assert bm._parse_weights("swe-zero=0.7,mini-coder=0.3") == {"swe-zero": 0.7, "mini-coder": 0.3}


def test_build_source_counts_rows_and_enriches_row_meta(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "swe-zero" / "data", "train-00000.parquet", 3)
    _write_shard(tmp_path / "swe-zero" / "data", "train-00001.parquet", 2)

    source = bm._build_source("swe-zero", 0.7, tmp_path)

    assert source["name"] == "swe-zero"
    assert source["weight"] == 0.7
    assert source["total_rows"] == 5
    assert [s["path"] for s in source["shards"]] == [
        "swe-zero/data/train-00000.parquet",
        "swe-zero/data/train-00001.parquet",
    ]
    assert all(len(s["sha256"]) == 64 for s in source["shards"])
    first = source["shards"][0]
    assert len(first["rows_meta"]) == first["rows"] == 3
    assert first["rows_meta"][0] == {"iid": "train-00000.parquet-0", "asst": 12}


def test_built_manifest_is_sampler_compatible(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "swe-zero" / "data", "train-00000.parquet", 150)
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000-of-00060.parquet", 150)

    sources = [
        bm._build_source("swe-zero", 0.7, tmp_path),
        bm._build_source("mini-coder", 0.3, tmp_path),
    ]
    manifest = {"version": "t", "sources": sources, "total_rows": 300}

    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=_BUCKET_TOTAL)
    assert len(ids) == _BUCKET_TOTAL
    assert sum(1 for i in ids if i.startswith("swe-zero/")) == 90
    assert sum(1 for i in ids if i.startswith("mini-coder/")) == 38
