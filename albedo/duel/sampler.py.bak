"""Deterministic sampling from SWE-ZERO parquet shards via a manifest.json index."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from albedo.config import DATASET_MANIFEST_SHA256


@dataclass
class Sample:
    global_idx:      int
    shard_idx:       int
    shard_name:      str
    sample_idx:      int
    turn_idx:        int
    instance_id:     str
    repo:            str
    messages_prefix: list[dict]   # conversation history up to this turn
    messages_prompt: list[dict]   # current user turn (single entry)
    original_reply:  str


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Return all rows from a parquet file as plain dicts."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        return table.to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(str(path)).to_dict(orient="records")


def _verify_manifest_sha256(manifest_path: Path, expected: str) -> None:
    """Raise ValueError if manifest sha256 doesn't match expected."""
    if not expected:
        return
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"manifest.json sha256 mismatch: "
            f"expected {expected!r}, got {digest!r}"
        )


def _extract_turns(row: dict[str, Any], shard_name: str, shard_idx: int, row_idx: int) -> list[Sample]:
    """Expand a parquet row into one Sample per assistant turn."""
    messages: list[dict] = row.get("messages") or []
    instance_id: str = row.get("instance_id", "")
    repo: str = row.get("repo", "")

    samples: list[Sample] = []
    prefix: list[dict] = []
    turn_idx = 0

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            user_prompt = prefix[-1:] if prefix and prefix[-1].get("role") == "user" else []
            samples.append(
                Sample(
                    global_idx=-1,          # filled in by TrajectoryDataset.sample
                    shard_idx=shard_idx,
                    shard_name=shard_name,
                    sample_idx=row_idx,
                    turn_idx=turn_idx,
                    instance_id=instance_id,
                    repo=repo,
                    messages_prefix=list(prefix[:-1]) if user_prompt else list(prefix),
                    messages_prompt=user_prompt,
                    original_reply=msg.get("content", ""),
                )
            )
            turn_idx += 1
        prefix.append(msg)

    return samples


class TrajectoryDataset:
    """Lazy-loading wrapper around SWE-ZERO parquet shards with manifest verification."""

    def __init__(self, dataset_dir: str, *, verify_manifest: bool = True) -> None:
        self._root = Path(dataset_dir)
        manifest_path = self._root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self._root}")

        if verify_manifest:
            _verify_manifest_sha256(manifest_path, DATASET_MANIFEST_SHA256)

        with manifest_path.open() as fh:
            self._manifest: dict[str, Any] = json.load(fh)

        # manifest schema: {"shards": [{"name": "...", "rows": N}, ...], "total_rows": N}
        self._shards: list[dict[str, Any]] = self._manifest.get("shards", [])
        self._total_rows: int = self._manifest.get("total_rows", sum(s.get("rows", 0) for s in self._shards))

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def sample(self, seed: bytes, n_samples: int, max_turns: int) -> list[Sample]:
        """Deterministically select n_samples (instance, turn) pairs from seed.

        Seeds random.Random from the first 8 bytes of seed (little-endian).
        Only loads shards that contain selected rows.
        """
        if not self._shards:
            return []

        entropy = int.from_bytes(seed[:8], "little")
        rng = random.Random(entropy)

        flat_index: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self._shards):
            n_rows = shard.get("rows", 0)
            flat_index.extend((shard_idx, row_idx) for row_idx in range(n_rows))

        if not flat_index:
            return []

        # Oversample to allow for deduplication and skipped rows.
        chosen_positions = rng.choices(range(len(flat_index)), k=n_samples * 4)

        shard_to_rows: dict[int, list[int]] = {}
        for pos in chosen_positions:
            shard_idx, row_idx = flat_index[pos]
            shard_to_rows.setdefault(shard_idx, []).append(row_idx)

        row_cache: dict[tuple[int, int], dict[str, Any]] = {}
        for shard_idx, row_indices in shard_to_rows.items():
            shard_name = self._shards[shard_idx]["name"]
            shard_path = self._root / shard_name           # name is relative to dataset root
            if not shard_path.exists():
                shard_path = self._root / Path(shard_name).name
            rows = _load_parquet_rows(shard_path)
            for row_idx in set(row_indices):
                if row_idx < len(rows):
                    row_cache[(shard_idx, row_idx)] = rows[row_idx]

        collected: list[Sample] = []
        seen: set[tuple[int, int, int]] = set()  # (shard_idx, row_idx, turn_idx)

        for pos in chosen_positions:
            if len(collected) >= n_samples:
                break
            shard_idx, row_idx = flat_index[pos]
            row = row_cache.get((shard_idx, row_idx))
            if row is None:
                continue

            shard_name = self._shards[shard_idx]["name"]
            turns = _extract_turns(row, shard_name, shard_idx, row_idx)
            if not turns:
                continue

            eligible = [t for t in turns if t.turn_idx < max_turns]
            if not eligible:
                continue
            turn = rng.choice(eligible)

            key = (shard_idx, row_idx, turn.turn_idx)
            if key in seen:
                continue
            seen.add(key)

            turn.global_idx = len(collected)
            collected.append(turn)

        return collected
