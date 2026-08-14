from __future__ import annotations

import json
import shutil

from config import Config
from state import State
from store import append_rows


def migrate_if_needed(cfg: Config, state: State) -> None:
    legacy_json = cfg.data_dir / "pipeline_state.json"
    if legacy_json.exists():
        data = json.loads(legacy_json.read_text())
        for run_id, meta in data.get("processed_runs", {}).items():
            state.mark_processed(
                run_id,
                meta.get("rows", 0),
                meta.get("source", "db"),
                meta.get("ingested_at"),
                meta.get("started_at"),
            )
        for key in ("anchor", "hf_repo", "last_new_dir_at", "last_db_check_at"):
            if data.get(key):
                state.set_meta(key, data[key])
        if data.get("readme_models"):
            state.set_meta("readme_models", json.dumps(data["readme_models"]))
        uploaded = set(data.get("uploaded", []))
        legacy_json.rename(legacy_json.with_suffix(".json.migrated"))
        print(f"migrated pipeline_state.json ({len(data.get('processed_runs', {}))} runs)")
    else:
        uploaded = set()

    if cfg.out_dir.is_dir():
        for f in sorted(cfg.out_dir.glob("*/*.parquet")):
            rel = f"data/{f.parent.name}/{f.name}"
            if not state.chunk_exists(rel):
                state.record_chunk(
                    rel, f.parent.name, uploaded_at="legacy" if rel in uploaded else None
                )

    all_dir = cfg.data_dir / "all"
    if all_dir.is_dir():
        import pandas as pd

        for p in sorted(all_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            rem = df.iloc[(len(df) // cfg.chunk_size) * cfg.chunk_size :]
            if len(rem):
                rows = [
                    {
                        "sample_id": rec["sample_id"],
                        "messages": [dict(m) for m in rec["messages"]],
                        "_run_id": rec["_run_id"],
                    }
                    for rec in rem.to_dict("records")
                ]
                for run_id in dict.fromkeys(r["_run_id"] for r in rows):
                    append_rows(cfg, p.stem, [r for r in rows if r["_run_id"] == run_id])
            print(f"migrated all/{p.name}: {len(rem)} remainder rows -> pending")
        shutil.rmtree(all_dir)
