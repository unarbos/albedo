"""albedo.judge.verdict — Per-metric verdict dataclass and JSON parser (pairwise).

The judge answers 1/2/0 per dimension (1 = MODEL 1 = king, 2 = MODEL 2 = challenger,
0 = draw). We parse that into challenger-perspective scores:

    challenger wins metric -> 1.0
    draw                   -> 0.5
    king wins metric       -> 0.0
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from albedo.judge.rubric import METRIC_KEYS

# Challenger-perspective score for each role outcome.
METRIC_SCORES: dict[str, float] = {
    "challenger": 1.0,
    "draw":       0.5,
    "king":       0.0,
}

_VALID_TOKENS = {"1", "2", "0", "draw", "tie", "model 1", "model_1", "model 2", "model_2"}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_KEY_ALIASES: dict[str, str] = {
    "correctness": "correctness",
    "correct": "correctness",
    "grounding": "grounding",
    "groundedness": "grounding",
    "faithfulness": "grounding",
    "progress": "progress",
    "task_progress": "progress",
    "protocol": "protocol",
    "format": "protocol",
    "instruction_following": "protocol",
    "efficiency": "efficiency",
    "conciseness": "efficiency",
}


@dataclass
class MetricVerdict:
    """Pairwise per-metric verdict for one turn from one judge model (challenger perspective)."""
    metric_scores: dict[str, float]   # {metric: 1.0/0.5/0.0}, one per METRIC_KEYS
    judge_mean:    float               # mean of the 5 metric scores
    raw:           str
    parse_ok:      bool                # True only if all 5 metrics parsed
    model:         str = field(default="")


def _extract_json(raw: str) -> dict | None:
    """Robustly pull the verdict object out of (possibly verbose) judge text.

    Scans every '{' position, decodes the longest valid JSON object there, and
    prefers an object carrying our expected metric keys (handles thinking-model
    preamble and trailing prose).
    """
    if not raw:
        return None
    dec = json.JSONDecoder()
    cands: list[dict] = []

    for match in _JSON_FENCE_RE.finditer(raw):
        body = match.group(1).strip()
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                cands.append(obj)
        except Exception:
            pass

    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = dec.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                cands.append(obj)
        except Exception:
            pass
        idx = raw.find("{", idx + 1)
    if not cands:
        return None
    keyed = [c for c in cands if set(METRIC_KEYS) & set(c.keys())]
    return (keyed or cands)[-1]


def _map_token(tok: str) -> str | None:
    """Map a numeric MODEL 1 / MODEL 2 / draw answer to king/challenger/draw.

    MODEL 1 (first) carries the king's reply, MODEL 2 (second) the challenger's,
    so 1 -> king, 2 -> challenger, 0 (or "draw"/"tie") -> draw.
    """
    tok = str(tok).strip().lower()
    tok = re.sub(r"[\s_-]+", " ", tok)
    if tok in ("0", "draw", "tie"):
        return "draw"
    if tok in (
        "1", "model 1", "model1", "response 1", "candidate 1",
        "a", "model a", "response a", "candidate a", "reply a", "king",
    ):
        return "king"
    if tok in (
        "2", "model 2", "model2", "response 2", "candidate 2",
        "b", "model b", "response b", "candidate b", "reply b", "challenger",
    ):
        return "challenger"
    if re.search(r"\b(draw|tie|equal|same)\b", tok):
        return "draw"
    if re.search(r"\b(model|response|candidate|reply)\s*1\b|\b1\b", tok):
        return "king"
    if re.search(r"\b(model|response|candidate|reply)\s*2\b|\b2\b", tok):
        return "challenger"
    return None


def _normalise_metric_obj(obj: dict) -> dict[str, object]:
    """Return values keyed by canonical metric names, accepting common aliases."""
    out: dict[str, object] = {}
    for key, value in obj.items():
        norm = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        canonical = _KEY_ALIASES.get(norm)
        if canonical and canonical not in out:
            out[canonical] = value
    return out


def parse_metric_verdict(raw: str) -> MetricVerdict:
    """Parse a judge's per-metric JSON into challenger-perspective scores.

    A metric that is missing or malformed scores 0.0 and flips parse_ok to False.
    """
    obj = _normalise_metric_obj(_extract_json(raw) or {})
    scores: dict[str, float] = {}
    ok = True
    for k in METRIC_KEYS:
        tok = str(obj.get(k, "")).strip().lower()
        role = _map_token(tok)
        if role is None:
            scores[k] = 0.0
            ok = False
        else:
            scores[k] = METRIC_SCORES[role]
    judge_mean = round(sum(scores.values()) / len(scores), 6) if scores else 0.0
    return MetricVerdict(metric_scores=scores, judge_mean=judge_mean, raw=raw, parse_ok=ok)
