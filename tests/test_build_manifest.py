import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.shared.sampling import multi_source_manifest_sample_ids


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_build_manifest():
    return _load_script("build_manifest")


_STRATEGIES = ["pr_1", "lm_rewrite__a", "combine_file__b", "func_pm_remove_cond__c"]


def _conversation(asst: int, edit_at: int) -> list[dict]:
    turns = [{"role": "system", "content": "s"}]
    for i in range(1, asst + 1):
        turns.append({"role": "user", "content": "o"})
        turns.append(
            {"role": "assistant", "content": "sed -i s/a/b/ f.py" if i == edit_at else "c"}
        )
    return turns


def _write_shard(data_dir: Path, name: str, rows: int, *, asst: int = 12) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "instance_id": [
                f"owner__repo{i}.abc1234.{_STRATEGIES[i % len(_STRATEGIES)]}" for i in range(rows)
            ],
            "messages": [_conversation(asst, 4 + (i % 5)) for i in range(rows)],
        }
    )
    pq.write_table(table, data_dir / name)


def test_parse_sources():
    bm = _load_build_manifest()
    assert bm._parse_sources("swe-hero, mini-coder") == ["swe-hero", "mini-coder"]


def test_build_source_counts_rows_and_enriches_row_meta(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 3)
    _write_shard(tmp_path / "mini-coder" / "data", "train-00001.parquet", 2)

    source = bm._build_source("mini-coder", tmp_path)

    assert source["name"] == "mini-coder"
    assert "weight" not in source
    assert source["total_rows"] == 5
    assert [s["path"] for s in source["shards"]] == [
        "mini-coder/data/train-00000.parquet",
        "mini-coder/data/train-00001.parquet",
    ]
    assert all(len(s["sha256"]) == 64 for s in source["shards"])
    first = source["shards"][0]
    assert len(first["rows_meta"]) == first["rows"] == 3
    assert first["rows_meta"][0] == {
        "iid": "owner__repo0.abc1234.pr_1",
        "asst": 12,
        "first_edit": 4,
        "family": "pr",
        "repo": "owner__repo0",
        "language": "python",
        "verified": None,
        "chars_at": 26,
        "chars_pre": 5,
    }
    assert [m["family"] for m in first["rows_meta"]] == ["pr", "lm", "combine"]


def test_build_meta_dict_strips_rows_meta_and_aggregates(tmp_path):
    bm = _load_build_manifest()
    bmm = _load_script("build_manifest_meta")
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 4)

    source = bm._build_source("mini-coder", tmp_path)
    source["shards"][0]["rows_meta"][3]["blocked"] = True
    meta = bmm.build_meta_dict({"version": "t", "sources": [source], "total_rows": 4})

    assert meta["version"] == "t" and meta["total_rows"] == 4
    src = meta["sources"][0]
    assert all("rows_meta" not in shard for shard in src["shards"])
    assert src["shards"][0]["rows"] == 4 and len(src["shards"][0]["sha256"]) == 64
    assert src["stats"] == {
        "instances": 3,
        "instances_with_edit": 3,
        "blocked_rows": 1,
        "families": {"pr": 1, "lm": 1, "combine": 1},
        "languages": {"python": 3},
    }
    assert meta["unique_instances"] == 3
    assert [p[0] for p in meta["sampling"]["phases"]] == ["pre_edit", "at_edit", "cold"]
    assert [f[0] for f in meta["sampling"]["families"]] == ["pr", "lm", "combine", "mechanical"]


def test_write_manifest_emits_meta_alongside(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 3)

    out_path, _manifest, _digest = bm.write_manifest(tmp_path, ["mini-coder"])

    meta = json.loads(out_path.with_name("manifest.meta.json").read_text())
    assert meta["sources"][0]["stats"]["instances"] == 3
    assert "rows_meta" not in meta["sources"][0]["shards"][0]


def test_built_manifest_is_sampler_compatible(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 400)
    _write_shard(tmp_path / "swe-hero" / "data", "train-00000-of-00060.parquet", 400)

    sources = [
        bm._build_source("mini-coder", tmp_path),
        bm._build_source("swe-hero", tmp_path),
    ]
    manifest = {"version": "t", "sources": sources, "total_rows": 800}

    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    assert len(ids) == 100 == len(set(ids))
    assert sum(1 for i in ids if i.startswith("mini-coder/")) > 0
    assert sum(1 for i in ids if i.startswith("swe-hero/")) > 0
