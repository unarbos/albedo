from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\S\s]*?)\s*```", re.IGNORECASE)


def extract_json(raw: str, prefer_keys: tuple[str, ...] = ()) -> Any | None:
    if not raw:
        return None
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for match in _JSON_FENCE_RE.finditer(raw):
        try:
            obj = json.loads(match.group(1).strip())
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            candidates.append(obj)
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        if index > 0 and raw[index - 1] == "`":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            candidates.append(obj)
    if prefer_keys:
        keyed = [c for c in candidates if isinstance(c, dict) and set(prefer_keys) & set(c)]
        if keyed:
            return keyed[-1]
    return candidates[0] if candidates else None
