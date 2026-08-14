from collections import Counter

import pytest

from albedo_eval_service.shared.sampling import (
    BENCHMARK_LANGUAGE,
    FAMILY_MIX,
    MAX_PREFIX_CHARS,
    NON_BENCHMARK_LANGUAGE_FRACTION,
    REPO_CAP,
    STEP_TRIM,
    _apportion,
    _family_grid,
    _prefix_chars,
    multi_source_manifest_sample_ids,
)

_FAMILIES = [name for name, _ in FAMILY_MIX]


def _shard(source: str, rows: int, *, asst: int = 30):
    return {
        "name": f"{source}/data/train-00000.parquet",
        "rows": rows,
        "rows_meta": [
            {
                "iid": f"{source}-{i}",
                "asst": asst,
                "first_edit": 5 + (i % 8),
                "family": _FAMILIES[i % len(_FAMILIES)],
                "language": BENCHMARK_LANGUAGE,
            }
            for i in range(rows)
        ],
    }


def _manifest(mini_rows: int = 2000, hero_rows: int = 800) -> dict:
    return {
        "version": "test-v1",
        "sources": [
            {
                "name": "mini-coder",
                "shards": [_shard("mini-coder", mini_rows)],
                "total_rows": mini_rows,
            },
            {
                "name": "swe-hero",
                "shards": [_shard("swe-hero", hero_rows)],
                "total_rows": hero_rows,
            },
        ],
        "total_rows": mini_rows + hero_rows,
    }


def _meta_by_coord(manifest: dict) -> dict[tuple[str, int], dict]:
    return {
        (shard["name"], row): entry
        for source in manifest["sources"]
        for shard in source["shards"]
        for row, entry in enumerate(shard["rows_meta"])
    }


def test_rejects_single_source_manifest():
    legacy = {"shards": [{"name": "data/train-00000.parquet", "rows": 3}], "total_rows": 3}
    with pytest.raises(ValueError, match="sources"):
        multi_source_manifest_sample_ids(legacy, block_hash="0xabc", sample_count=100)


def test_requires_rows_meta():
    no_meta = {
        "sources": [
            {
                "name": "mini-coder",
                "shards": [{"name": "mini-coder/data/train-0.parquet", "rows": 1}],
            },
        ]
    }
    with pytest.raises(ValueError, match="rows_meta"):
        multi_source_manifest_sample_ids(no_meta, block_hash="0xabc", sample_count=100)


def test_rejects_nonpositive_sample_count():
    with pytest.raises(ValueError, match="positive"):
        multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=0)


def test_returns_requested_count_of_unique_samples():
    ids = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=100)
    assert len(ids) == 100 == len(set(ids))
    assert len({i.rsplit(":", 1)[0] for i in ids}) == 100


def test_stable_and_seed_sensitive():
    first = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=100)
    second = multi_source_manifest_sample_ids(_manifest(), block_hash="0xabc", sample_count=100)
    third = multi_source_manifest_sample_ids(_manifest(), block_hash="0xdef", sample_count=100)
    assert first == second
    assert first != third


def test_family_mix_matches_spec():
    manifest = _manifest()
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    got = Counter(meta[(n, int(r))]["family"] for n, r, _ in (i.rsplit(":", 2) for i in ids))

    assert got == Counter(_apportion(FAMILY_MIX, 100))


def test_step_trim_anchors_on_first_edit():
    manifest = _manifest()
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)

    phases = Counter()
    for sid in ids:
        name, row, turn = sid.rsplit(":", 2)
        entry = meta[(name, int(row))]
        turn, first_edit = int(turn), entry["first_edit"]
        if turn == first_edit:
            phases["at_edit"] += 1
        elif turn == max(1, first_edit - 2):
            phases["pre_edit"] += 1
        elif turn in (1, 2):
            phases["cold"] += 1
        else:
            pytest.fail(f"{sid}: turn {turn} matches no phase for first_edit={first_edit}")
        assert entry["asst"] > turn, f"{sid}: no assistant turn remains after the cut"

    for phase, target in STEP_TRIM:
        assert abs(phases[phase] - target) <= 2, f"{phase}: {phases[phase]} vs {target}"


def test_pools_across_sources_so_shared_instances_are_not_double_drawn():
    shared = _manifest(mini_rows=400, hero_rows=400)
    for shard in shared["sources"][1]["shards"]:
        for entry in shard["rows_meta"]:
            entry["iid"] = entry["iid"].replace("swe-hero-", "mini-coder-")
    ids = multi_source_manifest_sample_ids(shared, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(shared)
    iids = [meta[(n, int(r))]["iid"] for n, r, _ in (i.rsplit(":", 2) for i in ids)]
    assert len(iids) == len(set(iids)) == 100


def test_missing_first_edit_falls_back_to_a_shallow_cut():
    manifest = _manifest()
    for source in manifest["sources"]:
        for shard in source["shards"]:
            for entry in shard["rows_meta"]:
                entry["first_edit"] = 0
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    assert {int(i.rsplit(":", 1)[1]) for i in ids} <= {1, 2}


def test_infeasible_stratum_raises():
    manifest = _manifest(mini_rows=40, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    with pytest.raises(ValueError, match="infeasible stratum"):
        multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)


def test_shallow_trajectories_are_skipped_not_mis_cut():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for i, entry in enumerate(shard["rows_meta"]):
            entry["asst"] = 30 if i % 3 == 0 else 3
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    for sid in ids:
        name, row, turn = sid.rsplit(":", 2)
        assert meta[(name, int(row))]["asst"] > int(turn)


def test_rejects_traversal_shard_name():
    bad = {
        "sources": [
            {
                "name": "x",
                "shards": [
                    {
                        "name": "../data/train-0.parquet",
                        "rows": 1,
                        "rows_meta": [{"iid": "a", "asst": 3, "first_edit": 1, "family": "pr"}],
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="data/train"):
        multi_source_manifest_sample_ids(bad, block_hash="0xabc", sample_count=100)


def test_apportion_largest_remainder_sums_exactly():
    for count in (7, 13, 64, 100, 137):
        assert sum(_apportion(STEP_TRIM, count).values()) == count
        assert sum(_apportion(FAMILY_MIX, count).values()) == count


def test_family_grid_margins_are_exact_on_both_axes():
    for count in (13, 64, 100, 137, 250):
        grid = _family_grid(count)
        phase_want = _apportion(STEP_TRIM, count)
        family_want = _apportion(FAMILY_MIX, count)
        assert {p: sum(r.values()) for p, r in grid.items()} == phase_want
        totals = Counter()
        for row in grid.values():
            totals.update(row)
        assert dict(totals) == family_want
        assert sum(totals.values()) == count


def test_repo_cap_limits_how_many_tasks_one_codebase_supplies():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for i, entry in enumerate(shard["rows_meta"]):
            entry["repo"] = f"owner__repo{i % 200}"
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    counts = Counter(meta[(n, int(r))]["repo"] for n, r, _ in (i.rsplit(":", 2) for i in ids))
    assert max(counts.values()) <= REPO_CAP


def test_repo_cap_infeasibility_is_reported_not_silently_short_drawn():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for entry in shard["rows_meta"]:
            entry["repo"] = "owner__only"
    with pytest.raises(ValueError, match="REPO_CAP"):
        multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)


def test_oversized_prefixes_are_skipped_for_anchored_phases():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for i, entry in enumerate(shard["rows_meta"]):
            entry["repo"] = f"owner__repo{i}"
            huge = i % 3 == 0
            entry["chars_at"] = MAX_PREFIX_CHARS * 2 if huge else 1_000
            entry["chars_pre"] = MAX_PREFIX_CHARS * 2 if huge else 1_000
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    for sid in ids:
        name, row, turn = sid.rsplit(":", 2)
        entry, turn = meta[(name, int(row))], int(turn)
        if turn in (1, 2) and turn != entry["first_edit"]:
            continue
        assert entry["chars_at"] <= MAX_PREFIX_CHARS


def test_cold_phase_ignores_the_prefix_gate():
    entry = {"chars_at": MAX_PREFIX_CHARS * 9, "chars_pre": MAX_PREFIX_CHARS * 9}
    assert _prefix_chars("cold", entry) == 0
    assert _prefix_chars("at_edit", entry) == MAX_PREFIX_CHARS * 9


def test_prefix_gate_tolerates_manifests_without_chars_fields():
    assert _prefix_chars("at_edit", {}) == 0


def test_non_benchmark_languages_are_capped():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for i, entry in enumerate(shard["rows_meta"]):
            entry["repo"] = f"owner__repo{i}"
            entry["language"] = "rust" if i % 3 else BENCHMARK_LANGUAGE
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    other = sum(
        1
        for n, r, _ in (i.rsplit(":", 2) for i in ids)
        if meta[(n, int(r))]["language"] != BENCHMARK_LANGUAGE
    )
    assert other <= round(100 * NON_BENCHMARK_LANGUAGE_FRACTION)


def test_language_cap_does_not_break_the_family_mix():
    manifest = _manifest(mini_rows=4000, hero_rows=0)
    manifest["sources"] = manifest["sources"][:1]
    for shard in manifest["sources"][0]["shards"]:
        for i, entry in enumerate(shard["rows_meta"]):
            entry["repo"] = f"owner__repo{i}"
            entry["language"] = "go" if i % 3 else BENCHMARK_LANGUAGE
    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    meta = _meta_by_coord(manifest)
    got = Counter(meta[(n, int(r))]["family"] for n, r, _ in (i.rsplit(":", 2) for i in ids))
    assert got == Counter(_apportion(FAMILY_MIX, 100))


def test_missing_language_defaults_to_the_benchmark_language():
    manifest = _manifest()
    for source in manifest["sources"]:
        for shard in source["shards"]:
            for entry in shard["rows_meta"]:
                entry.pop("language", None)
    assert (
        len(multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)) == 100
    )
