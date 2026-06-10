from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    prompt: str
    target: str | None = None


def load_swe_zero_samples(*, dataset_root: str | Path, sample_ids: list[str]) -> list[EvalSample]:
    root = Path(dataset_root)
    return [_load_sample(root, sample_id) for sample_id in sample_ids]


def _load_sample(root: Path, sample_id: str) -> EvalSample:
    shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
    row = _read_parquet_row(root / shard_name, row_idx)
    prompt, target = _prompt_from_row(row, turn_idx=turn_idx)
    return EvalSample(sample_id=sample_id, prompt=prompt, target=target)


def _parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    shard_name, row_idx_raw, turn_idx_raw = sample_id.rsplit(":", 2)
    if not shard_name.startswith("data/train-") or not shard_name.endswith(".parquet"):
        raise ValueError(f"unsupported SWE-ZERO shard in sample_id: {sample_id}")
    return shard_name, int(row_idx_raw), int(turn_idx_raw)


def _read_parquet_row(path: Path, row_idx: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"SWE-ZERO shard not found: {path}")
    if row_idx < 0:
        raise ValueError("row_idx must be non-negative")

    parquet_file = pq.ParquetFile(path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=1024):
        if seen + batch.num_rows <= row_idx:
            seen += batch.num_rows
            continue
        local_idx = row_idx - seen
        return batch.slice(local_idx, 1).to_pydict() | {"__row_idx": [row_idx]}
    raise IndexError(f"row_idx {row_idx} out of range for shard {path}")


def _prompt_from_row(row: dict[str, Any], *, turn_idx: int) -> tuple[str, str | None]:
    normalized = {key: _unwrap_column(value) for key, value in row.items()}
    turns = _extract_turns(normalized)
    if turns:
        assistant_turns = [index for index, turn in enumerate(turns) if _role(turn) == "assistant"]
        source_index = assistant_turns[turn_idx] if turn_idx < len(assistant_turns) else min(turn_idx, len(turns) - 1)
        prompt_turns = turns[:source_index]
        target = _content(turns[source_index]) if _role(turns[source_index]) == "assistant" else None
        return _format_prompt(prompt_turns), target

    for key in ("prompt", "instruction", "question", "input", "text"):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            return value, None
    return json.dumps({key: value for key, value in normalized.items() if not key.startswith("__")}, sort_keys=True), None


def _extract_turns(row: dict[str, Any]) -> list[Any]:
    for key in ("messages", "turns", "conversation", "trajectory"):
        value = row.get(key)
        parsed = _maybe_json(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for nested_key in ("messages", "turns", "conversation"):
                nested = parsed.get(nested_key)
                if isinstance(nested, list):
                    return nested
    return []


def _format_prompt(turns: list[Any]) -> str:
    lines = []
    for turn in turns:
        role = _role(turn) or "user"
        content = _content(turn)
        if content:
            lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _unwrap_column(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _role(turn: Any) -> str | None:
    if isinstance(turn, dict):
        role = turn.get("role") or turn.get("speaker") or turn.get("from")
        return str(role).lower() if role is not None else None
    return None


def _content(turn: Any) -> str:
    if isinstance(turn, dict):
        for key in ("content", "text", "value", "message"):
            value = turn.get(key)
            if value is not None:
                return str(value)
    return str(turn) if turn is not None else ""
