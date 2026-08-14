#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from albedo_eval_service.shared.sampling import _SHARD_RE, SAMPLING_ALGO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest_meta import write_meta
from prepare_datasets import SOURCES

DEFAULT_VERSION = "mini-coder+open-swe+smith-rs+hero-v1"


def _parse_sources(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("--sources must be like 'mini-coder,open-swe-traces'")
    return names


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_EDIT_RE = re.compile(
    r"sed\s+-i|tee\s+[\w./-]|cat\s*>|str_replace|git apply|patch\s+-p|applypatch|"
    r"cp\s+[\w./-]|mv\s+[\w./-]|(?<![-\d&])>>?\s*(?!/dev/)[\w.][\w./-]*"
)


def _family(instance_id: str) -> str:
    if "." not in instance_id:
        return "pr"
    tail = instance_id.rsplit(".", 1)[-1]
    for prefix, family in (("pr_", "pr"), ("lm_", "lm"), ("combine", "combine")):
        if tail.startswith(prefix):
            return family
    return "mechanical"


def _row_meta(path: Path, blocked: frozenset[str] = frozenset(), language: str = "") -> list[dict]:
    metas: list[dict] = []
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    columns = ["instance_id", "messages"] + [
        c for c in ("first_edit", "family", "repo", "language", "verified") if c in available
    ]
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        for row in batch.to_pylist():
            iid = str(row.get("instance_id") or "")
            asst = 0
            regex_first_edit = 0
            running = 0
            chars_by_turn: list[int] = []
            for turn in row.get("messages") or []:
                if not isinstance(turn, dict):
                    continue
                running += len(str(turn.get("content") or ""))
                if str(turn.get("role", "")).lower() != "assistant":
                    continue
                asst += 1
                chars_by_turn.append(running)
                if not regex_first_edit and _EDIT_RE.search(str(turn.get("content") or "")):
                    regex_first_edit = asst
            given = row.get("first_edit")
            first_edit = int(given) if given is not None else regex_first_edit

            def _chars(cut: int, _by_turn: list[int] = chars_by_turn) -> int:
                if not _by_turn:
                    return 0
                return _by_turn[min(max(cut, 1), len(_by_turn)) - 1]

            metas.append(
                {
                    **({"blocked": True} if iid in blocked else {}),
                    "iid": iid,
                    "asst": asst,
                    "first_edit": first_edit,
                    "family": str(row.get("family") or _family(iid)),
                    "repo": str(row.get("repo") or iid.split(".")[0]),
                    "language": str(row.get("language") or language or "unknown"),
                    "verified": bool(row.get("verified")) if "verified" in available else None,
                    "chars_at": _chars(first_edit or 1),
                    "chars_pre": _chars(max(1, (first_edit or 1) - 2)),
                }
            )
    return metas


def _build_source(name: str, root: Path, *, max_workers: int = 8) -> dict:
    if name not in SOURCES:
        raise SystemExit(f"{name}: unknown source (not in prepare_datasets.SOURCES)")
    repo = SOURCES[name]["repos"][0]
    shard_glob = SOURCES[name]["shard_glob"]
    data_dir = root / name / "data"
    name_pattern = shard_glob.rsplit("/", 1)[-1]
    files = sorted(data_dir.glob(name_pattern), key=lambda p: p.name)
    if not files:
        raise SystemExit(
            f"{name}: no parquet shards under {data_dir} (run prepare_datasets.py first)"
        )

    def _shard(path: Path) -> dict:
        shard_path = f"{name}/data/{path.name}"
        if not _SHARD_RE.match(shard_path):
            raise SystemExit(
                f"{name}: shard name {shard_path!r} is not a valid (<source>/)data/train-*.parquet"
            )
        rows_meta = _row_meta(
            path, frozenset(SOURCES[name].get("exclude_ids", ())), SOURCES[name].get("language", "")
        )
        return {
            "path": shard_path,
            "rows": len(rows_meta),
            "sha256": _sha256(path),
            "rows_meta": rows_meta,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shards = list(pool.map(_shard, files))
    total_rows = sum(s["rows"] for s in shards)

    return {
        "name": name,
        "repo": repo,
        "shard_glob": shard_glob,
        "shards": shards,
        "total_rows": total_rows,
    }


def build_manifest_dict(
    root: Path, names: list[str], *, version: str = DEFAULT_VERSION, max_workers: int = 8
) -> dict:
    sources = [_build_source(name, root, max_workers=max_workers) for name in names]
    sources.sort(key=lambda s: s["name"])
    return {
        "version": version,
        "sources": sources,
        "total_rows": sum(s["total_rows"] for s in sources),
    }


def write_manifest(
    root: Path,
    names: list[str],
    *,
    out_path: Path | None = None,
    version: str = DEFAULT_VERSION,
    max_workers: int = 8,
) -> tuple[Path, dict, str]:
    out_path = Path(out_path) if out_path else Path(root) / "manifest.json"
    manifest = build_manifest_dict(Path(root), names, version=version, max_workers=max_workers)
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    out_path.write_bytes(payload)
    write_meta(manifest, out_path.with_name("manifest.meta.json"))
    return out_path, manifest, hashlib.sha256(payload).hexdigest()


def print_manifest_summary(out_path: Path, manifest: dict, digest: str) -> None:
    sources = manifest["sources"]
    print(f"wrote {out_path} ({manifest['total_rows']} rows across {len(sources)} sources)")
    meta_path = out_path.with_name("manifest.meta.json")
    print(f"wrote {meta_path} (dashboard meta; upload as datasets/manifest.meta.json)")
    for source in sources:
        print(f"  {source['name']}: rows={source['total_rows']} shards={len(source['shards'])}")
    print()
    print(f"sha256: {digest}")
    print()
    print("Update these to pin the new manifest:")
    print(f"  ALBEDO_EVAL_DATASET_MANIFEST_HASH={digest}")
    print(f"  SANITY_DISPATCH_DATASET_MANIFEST_HASH={digest}")
    print(f"  ALBEDO_EVAL_SAMPLING_ALGO={SAMPLING_ALGO}")
    print("  src/albedo_eval_service/config.py  -> dataset_manifest_hash default")
    print("  src/sanity_service/settings.py     -> dataset_manifest_hash default")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined eval dataset manifest.")
    parser.add_argument(
        "--dataset-root", required=True, help="Root dir holding <source>/data/*.parquet."
    )
    parser.add_argument("--sources", required=True, help="e.g. 'mini-coder,open-swe-traces'")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Manifest version label.")
    parser.add_argument("--out", default=None, help="Output path (default: <root>/manifest.json).")
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Parallel shard-hashing workers (default: 8)."
    )
    args = parser.parse_args()

    out_path, manifest, digest = write_manifest(
        Path(args.dataset_root),
        _parse_sources(args.sources),
        out_path=Path(args.out) if args.out else None,
        version=args.version,
        max_workers=args.max_workers,
    )
    print_manifest_summary(out_path, manifest, digest)


if __name__ == "__main__":
    main()
