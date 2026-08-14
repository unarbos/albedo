from __future__ import annotations

import subprocess
from pathlib import Path

from config import Config

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
COMPLETE_MARKERS = {"verdict.json", "generated-samples.jsonl", "scoring-results.jsonl"}


def _require_dir(cfg: Config) -> str:
    if not cfg.eval_artifacts_dir:
        raise RuntimeError("DATASET_CREATOR_EVAL_ARTIFACTS_DIR env var not set")
    return cfg.eval_artifacts_dir


def list_remote_runs(cfg: Config) -> dict[str, dict]:
    art_dir = _require_dir(cfg)
    if not cfg.eval_ssh_host:
        return _list_local(Path(art_dir))
    cmd = f"find {art_dir} -mindepth 2 -maxdepth 2 -type f -printf '%P\\t%s\\t%T@\\n'"
    res = subprocess.run(
        ["ssh", *SSH_OPTS, cfg.eval_ssh_host, cmd], capture_output=True, text=True, timeout=60
    )
    if res.returncode != 0:
        raise RuntimeError(f"eval-machine listing failed: {res.stderr.strip()[:300]}")
    runs: dict[str, dict] = {}
    for line in res.stdout.splitlines():
        try:
            rel, size, mtime = line.split("\t")
            run_id, name = rel.split("/", 1)
        except ValueError:
            continue
        run = runs.setdefault(run_id, {"files": {}, "mtime": 0.0})
        run["files"][name] = int(size)
        run["mtime"] = max(run["mtime"], float(mtime))
    return runs


def _list_local(art_dir: Path) -> dict[str, dict]:
    if not art_dir.is_dir():
        raise RuntimeError(f"artifact dir not found: {art_dir}")
    runs: dict[str, dict] = {}
    for run_dir in art_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run = runs.setdefault(run_dir.name, {"files": {}, "mtime": 0.0})
        for f in run_dir.iterdir():
            if f.is_file():
                st = f.stat()
                run["files"][f.name] = st.st_size
                run["mtime"] = max(run["mtime"], st.st_mtime)
    return runs


def is_complete(files: dict[str, int]) -> bool:
    return COMPLETE_MARKERS.issubset(files)


def fetch_run(cfg: Config, run_id: str, files: dict[str, int]) -> Path:
    art_dir = _require_dir(cfg)
    if not cfg.eval_ssh_host:
        return Path(art_dir) / run_id
    dest = cfg.run_dir(run_id)
    if dest.exists() and all(
        (dest / n).exists() and (dest / n).stat().st_size == s for n, s in files.items()
    ):
        return dest
    res = subprocess.run(
        [
            "scp",
            "-q",
            *SSH_OPTS,
            "-r",
            f"{cfg.eval_ssh_host}:{art_dir}/{run_id}",
            str(cfg.data_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(f"scp failed for {run_id}: {res.stderr.strip()[:300]}")
    for name, size in files.items():
        got = (dest / name).stat().st_size if (dest / name).exists() else -1
        if got != size:
            raise RuntimeError(f"size mismatch after scp: {run_id}/{name} {got} != {size}")
    return dest
