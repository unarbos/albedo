"""Deterministic trajectory sampling from a local SWE-ZERO parquet corpus.

The full dataset is prefetched once to disk (`scripts/prefetch_dataset.py`).
Sampling picks rows uniformly at random across *all* shards by drawing global
row indices in `[0, total_rows)` — no shard-size bias.

The seed argument fully pins which rows and which turns within each row
will be evaluated. Validators running on the same `(block_hash, challenger_hotkey)`
will pick the exact same fixture set.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import chain_config

log = logging.getLogger("albedo.sampler")

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Sample:
    """One (trajectory prefix, turn-to-evaluate) pair.

    `messages_prefix` is what gets sent to BOTH contestant models. The
    original assistant turn at `turn_idx` is held aside as
    `original_reply` so we can show diffs in the dashboard if we ever want
    to (judging itself only sees the candidate replies, not the original).
    """
    instance_id: str
    repo: str
    global_idx: int       # row index across the full corpus, deterministic
    shard_idx: int        # index into the sorted shard list
    shard_name: str       # basename of the parquet file
    sample_idx: int       # row index inside the shard (alias for local row)
    turn_idx: int         # index of the assistant message in `messages`
    messages_prefix: list[dict]
    original_reply: str


@dataclass(frozen=True)
class _ShardInfo:
    path: Path
    name: str
    rows: int
    sha256: str = ""


@dataclass
class DatasetCatalog:
    """Index over all local parquet shards."""

    dataset_dir: Path
    shards: tuple[_ShardInfo, ...]
    total_rows: int
    _cum_rows: np.ndarray  # exclusive cumulative row counts per shard

    @classmethod
    def from_disk(cls, dataset_dir: str | Path) -> DatasetCatalog:
        root = Path(dataset_dir)
        if not root.is_dir():
            raise FileNotFoundError(
                f"dataset directory not found at {root}; "
                "run scripts/prefetch_dataset.py first"
            )
        manifest_path = root / MANIFEST_NAME
        if manifest_path.exists():
            cls._verify_manifest(manifest_path)
            shards = cls._shards_from_manifest(root, manifest_path)
        else:
            log.warning(
                "%s missing; building catalog from parquet metadata on disk",
                manifest_path,
            )
            shards = cls._shards_from_glob(root, chain_config.DATASET_SHARD_GLOB)
        if not shards:
            raise FileNotFoundError(
                f"no parquet shards under {root} matching "
                f"{chain_config.DATASET_SHARD_GLOB!r}; "
                "run scripts/prefetch_dataset.py first"
            )
        rows = [s.rows for s in shards]
        total = sum(rows)
        cum = np.cumsum(rows, dtype=np.int64)
        log.info(
            "dataset catalog: dir=%s shards=%d total_rows=%d",
            root,
            len(shards),
            total,
        )
        return cls(
            dataset_dir=root,
            shards=tuple(shards),
            total_rows=total,
            _cum_rows=cum,
        )

    @staticmethod
    def _verify_manifest(manifest_path: Path) -> None:
        pinned = (chain_config.DATASET_MANIFEST_SHA256 or "").strip().lower()
        if not pinned or pinned == "0" * 64:
            log.warning(
                "chain.toml [dataset].manifest_sha256 is unset; skipping integrity check"
            )
            return
        h = hashlib.sha256()
        with open(manifest_path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        got = h.hexdigest()
        if got != pinned:
            raise RuntimeError(
                f"dataset manifest sha256 mismatch:\n  pinned={pinned}\n  on-disk={got}\n"
                f"manifest={manifest_path}\nrefetch with scripts/prefetch_dataset.py."
            )

    @staticmethod
    def _parquet_row_count(path: Path) -> int:
        return pq.ParquetFile(path).metadata.num_rows

    @classmethod
    def _shards_from_manifest(cls, root: Path, manifest_path: Path) -> list[_ShardInfo]:
        doc = json.loads(manifest_path.read_text())
        out: list[_ShardInfo] = []
        for entry in doc.get("shards", []):
            rel = entry["path"]
            p = root / rel if not Path(rel).is_absolute() else Path(rel)
            if not p.exists():
                raise FileNotFoundError(f"manifest shard missing on disk: {p}")
            rows = int(entry["rows"])
            meta_rows = cls._parquet_row_count(p)
            if meta_rows != rows:
                raise RuntimeError(
                    f"shard row count mismatch for {p}: manifest={rows} parquet={meta_rows}"
                )
            out.append(
                _ShardInfo(
                    path=p,
                    name=p.name,
                    rows=rows,
                    sha256=str(entry.get("sha256") or ""),
                )
            )
        return out

    @classmethod
    def _shards_from_glob(cls, root: Path, glob_pattern: str) -> list[_ShardInfo]:
        paths = sorted(root.glob(glob_pattern))
        if not paths and "/" not in glob_pattern:
            paths = sorted(root.glob(f"**/{glob_pattern}"))
        out: list[_ShardInfo] = []
        for p in paths:
            if not p.is_file():
                continue
            out.append(
                _ShardInfo(
                    path=p,
                    name=p.name,
                    rows=cls._parquet_row_count(p),
                )
            )
        return out

    def global_to_shard(self, global_idx: int) -> tuple[int, int]:
        """Map a corpus-wide row index to (shard_idx, local_row_idx)."""
        if global_idx < 0 or global_idx >= self.total_rows:
            raise IndexError(f"global_idx {global_idx} out of range [0, {self.total_rows})")
        shard_idx = int(np.searchsorted(self._cum_rows, global_idx, side="right"))
        start = int(self._cum_rows[shard_idx - 1]) if shard_idx > 0 else 0
        return shard_idx, global_idx - start


_CATALOG: DatasetCatalog | None = None
_CATALOG_DIR: Path | None = None
_SHARD_TABLES: dict[int, pq.Table] = {}


def load_catalog(dataset_dir: str | Path) -> DatasetCatalog:
    """Build or return the cached dataset catalog for `dataset_dir`."""
    global _CATALOG, _CATALOG_DIR
    root = Path(dataset_dir)
    if _CATALOG is not None and _CATALOG_DIR == root:
        return _CATALOG
    _CATALOG = DatasetCatalog.from_disk(root)
    _CATALOG_DIR = root
    _SHARD_TABLES.clear()
    return _CATALOG


def _load_shard_table(catalog: DatasetCatalog, shard_idx: int) -> pq.Table:
    cached = _SHARD_TABLES.get(shard_idx)
    if cached is not None:
        return cached
    shard = catalog.shards[shard_idx]
    log.info("mmap shard %s (%d rows)", shard.path, shard.rows)
    table = pq.read_table(shard.path, memory_map=True)
    _SHARD_TABLES[shard_idx] = table
    return table


def _row(table: pq.Table, idx: int) -> dict:
    return {col: table.column(col)[idx].as_py() for col in table.column_names}


def _seed_to_rng(seed: bytes) -> np.random.Generator:
    digest = hashlib.blake2b(seed, digest_size=32).digest()
    entropy = np.frombuffer(digest, dtype=np.uint64).tolist()
    seq = np.random.SeedSequence(entropy=entropy)
    return np.random.Generator(np.random.PCG64DXSM(seq))


def _assistant_turn_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def sample(
    seed: bytes,
    *,
    n_samples: int | None = None,
    max_turns_per_sample: int | None = None,
    dataset_dir: str | Path,
) -> list[Sample]:
    """Pick `n_samples` rows uniformly from the full corpus, then up to
    `max_turns_per_sample` assistant turns from each.

    Row selection uses global indices in `[0, total_rows)` so every trajectory
    has equal probability regardless of which shard it lives in.

    Deterministic in `seed`. Same seed + same on-disk corpus => same Sample list.
    """
    n_samples = n_samples if n_samples is not None else chain_config.DUEL_N_SAMPLES
    max_turns = (
        max_turns_per_sample
        if max_turns_per_sample is not None
        else chain_config.DUEL_MAX_TURNS_PER_SAMPLE
    )

    catalog = load_catalog(dataset_dir)
    total_rows = catalog.total_rows
    rng = _seed_to_rng(seed)

    pool = min(total_rows, max(n_samples * 4, n_samples + 16))
    candidates = rng.choice(total_rows, size=pool, replace=False)

    picked: list[Sample] = []
    for gidx in candidates:
        if len(picked) >= n_samples:
            break
        gidx = int(gidx)
        shard_idx, local_idx = catalog.global_to_shard(gidx)
        shard = catalog.shards[shard_idx]
        table = _load_shard_table(catalog, shard_idx)
        row = _row(table, local_idx)
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            continue
        asst = _assistant_turn_indices(messages)
        if len(asst) < 1:
            continue
        k = min(len(asst), max_turns)
        turn_idxs = rng.choice(asst, size=k, replace=False).tolist()
        turn_idxs.sort()
        for ti in turn_idxs:
            picked.append(
                Sample(
                    instance_id=str(row.get("instance_id") or ""),
                    repo=str(row.get("repo") or ""),
                    global_idx=gidx,
                    shard_idx=shard_idx,
                    shard_name=shard.name,
                    sample_idx=local_idx,
                    turn_idx=int(ti),
                    messages_prefix=messages[:ti],
                    original_reply=str(messages[ti].get("content") or ""),
                )
            )

    if not picked:
        raise RuntimeError(
            f"sampler produced 0 turns from corpus rows={total_rows} "
            f"(n_samples={n_samples}, max_turns={max_turns}). "
            "Is the dataset the expected schema (messages: list[{role,content}])?"
        )
    log.info(
        "sampled %d (sample, turn) pairs from %d candidate rows across %d shards (seed=%s)",
        len(picked),
        n_samples,
        len(catalog.shards),
        hashlib.blake2b(seed, digest_size=8).hexdigest(),
    )
    return picked


def build_manifest(dataset_dir: str | Path, *, shard_glob: str | None = None) -> Path:
    """Scan local shards and write `manifest.json`. Returns the manifest path."""
    root = Path(dataset_dir)
    pattern = shard_glob or chain_config.DATASET_SHARD_GLOB
    shards = DatasetCatalog._shards_from_glob(root, pattern)
    if not shards:
        raise FileNotFoundError(f"no shards under {root} matching {pattern!r}")

    entries = []
    for shard in shards:
        rel = str(shard.path.relative_to(root))
        sha = hashlib.sha256()
        with open(shard.path, "rb") as f:
            while chunk := f.read(1 << 20):
                sha.update(chunk)
        entries.append(
            {
                "path": rel,
                "rows": shard.rows,
                "sha256": sha.hexdigest(),
            }
        )

    doc = {
        "repo": chain_config.DATASET_REPO,
        "shard_glob": pattern,
        "total_rows": sum(e["rows"] for e in entries),
        "shards": entries,
    }
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return manifest_path
