from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from config import Config
from state import State

_CHUNK_NO_RE = re.compile(r"-(\d+)\.parquet$")


def _pending_path(cfg: Config, model: str) -> Path:
    return cfg.pending_dir / f"{model}.json"


def _load_pending(cfg: Config, model: str) -> list[dict]:
    path = _pending_path(cfg, model)
    return json.loads(path.read_text()) if path.exists() else []


def _save_pending(cfg: Config, model: str, rows: list[dict]) -> None:
    path = _pending_path(cfg, model)
    if not rows:
        path.unlink(missing_ok=True)
        return
    cfg.pending_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows))
    tmp.rename(path)


def append_rows(cfg: Config, model: str, rows: list[dict]) -> None:
    pending = _load_pending(cfg, model)
    run_id = rows[0]["_run_id"]
    pending = [r for r in pending if r["_run_id"] != run_id] + rows
    _save_pending(cfg, model, pending)


def _repo_path(model: str, name: str) -> str:
    return f"data/{model}/{name}"


def _write_chunk(cfg: Config, model: str, no: int, rows: list[dict]) -> str:
    model_dir = cfg.out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    name = f"{model}-{no:04d}.parquet"
    tmp = model_dir / (name + ".tmp")
    pd.DataFrame(rows)[["sample_id", "messages"]].to_parquet(tmp, index=False)
    tmp.rename(model_dir / name)
    return name


def _reconcile_orphans(cfg: Config, state: State, model: str, pending: list[dict]) -> list[dict]:
    model_dir = cfg.out_dir / model
    if not model_dir.is_dir():
        return pending
    for f in sorted(model_dir.glob("*.parquet")):
        rel = _repo_path(model, f.name)
        if state.chunk_exists(rel):
            continue
        chunk_ids = list(pd.read_parquet(f, columns=["sample_id"])["sample_id"])
        head_ids = [r["sample_id"] for r in pending[: len(chunk_ids)]]
        if head_ids == chunk_ids:
            pending = pending[len(chunk_ids) :]
            _save_pending(cfg, model, pending)
        state.record_chunk(rel, model)
    return pending


def _next_chunk_no(cfg: Config, model: str) -> int:
    model_dir = cfg.out_dir / model
    nos = (
        [int(m.group(1)) for f in model_dir.glob("*.parquet") if (m := _CHUNK_NO_RE.search(f.name))]
        if model_dir.is_dir()
        else []
    )
    return max(nos, default=0) + 1


def cut_chunks(cfg: Config, state: State) -> None:
    models = (
        sorted({p.stem for p in cfg.pending_dir.glob("*.json")} | set(state.chunk_models()))
        if cfg.pending_dir.exists()
        else state.chunk_models()
    )
    for model in models:
        pending = _reconcile_orphans(cfg, state, model, _load_pending(cfg, model))
        cut = 0
        while len(pending) >= cfg.chunk_size:
            rows, pending = pending[: cfg.chunk_size], pending[cfg.chunk_size :]
            name = _write_chunk(cfg, model, _next_chunk_no(cfg, model), rows)
            _save_pending(cfg, model, pending)
            state.record_chunk(_repo_path(model, name), model)
            cut += 1
        print(
            f"  {model}: {state.chunk_count(model)} chunks"
            + (f" (+{cut} new)" if cut else "")
            + f", {len(pending)} waiting"
        )
