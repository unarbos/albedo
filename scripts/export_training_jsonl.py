#!/usr/bin/env python3
"""Export Albedo eval traces to flat training JSONL (SFT / DPO).

Reads one or more `.jsonl.gz` eval trace files (local paths or HTTPS URLs)
and writes a single JSONL with one row per usable turn.

SFT rows (default):
    {"messages": [...], "source": "gold"|"king"|"chal", ...metadata}

DPO rows (--format dpo):
    {"prompt": [...], "chosen": str, "rejected": str, ...metadata}

Usage:
    source .venv/bin/activate
    python scripts/export_training_jsonl.py \\
        --input https://us-east-1.hippius.com/albedo/evals/2026-05-26/eval-0005.jsonl.gz \\
        --output training.jsonl

    python scripts/export_training_jsonl.py \\
        --input /var/albedo/evals/2026-05-26/*.jsonl.gz \\
        --format dpo --min-chal-score 0.5 --output dpo.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path


def _open_trace(path: str):
    if path.startswith("http://") or path.startswith("https://"):
        resp = urllib.request.urlopen(path)
        return gzip.GzipFile(fileobj=resp)
    return gzip.open(path, "rt", encoding="utf-8")


def _iter_records(path: str):
    with _open_trace(path) as f:
        for line in f:
            line = line.decode() if isinstance(line, bytes) else line
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_valid_turn(rec: dict) -> bool:
    if rec.get("type") != "turn":
        return False
    if rec.get("error"):
        return False
    king = (rec.get("king") or {}).get("reply") or ""
    chal = (rec.get("chal") or {}).get("reply") or ""
    return bool(king.strip() and chal.strip())


def _best_judge_scores(rec: dict) -> tuple[float, float]:
    judges = rec.get("judges") or []
    if not judges:
        return rec.get("king_score_avg", 0.0), rec.get("chal_score_avg", 0.0)
    king = sum(j.get("king_score", 0.0) for j in judges) / len(judges)
    chal = sum(j.get("chal_score", 0.0) for j in judges) / len(judges)
    return king, chal


def _base_meta(rec: dict, eval_id: str, accepted: bool | None) -> dict:
    return {
        "eval_id": eval_id,
        "hotkey": rec.get("hotkey"),
        "challenger": rec.get("challenger"),
        "sample_idx": rec.get("sample_idx"),
        "global_idx": rec.get("global_idx"),
        "shard_idx": rec.get("shard_idx"),
        "shard_name": rec.get("shard_name"),
        "turn_idx": rec.get("turn_idx"),
        "instance_id": rec.get("instance_id"),
        "repo": rec.get("repo"),
        "prompt_truncated": rec.get("prompt_truncated", False),
        "accepted": accepted,
        "schema_version": rec.get("schema_version"),
    }


def export_sft(
    paths: list[str],
    *,
    include_gold: bool,
    include_king: bool,
    include_chal: bool,
    min_chal_score: float,
    require_accept: bool,
) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        meta: dict | None = None
        verdict: dict | None = None
        for rec in _iter_records(path):
            if rec.get("type") == "duel_meta":
                meta = rec
            elif rec.get("type") == "verdict":
                verdict = rec
            elif rec.get("type") == "turn" and _is_valid_turn(rec):
                eval_id = rec.get("eval_id") or (meta or {}).get("eval_id", "?")
                accepted = (verdict or {}).get("accepted")
                if require_accept and not accepted:
                    continue
                prefix = rec.get("messages_prompt") or rec.get("messages_prefix") or []
                king_score, chal_score = _best_judge_scores(rec)
                if chal_score < min_chal_score:
                    continue
                base = _base_meta(rec, eval_id, accepted)

                if include_gold:
                    gold = rec.get("original_reply") or ""
                    if gold.strip():
                        rows.append({
                            **base,
                            "source": "gold",
                            "messages": prefix + [
                                {"role": "assistant", "content": gold},
                            ],
                        })

                if include_king:
                    king_reply = (rec.get("king") or {}).get("reply") or ""
                    if king_reply.strip():
                        rows.append({
                            **base,
                            "source": "king",
                            "king_score": king_score,
                            "chal_score": chal_score,
                            "messages": prefix + [
                                {"role": "assistant", "content": king_reply},
                            ],
                        })

                if include_chal:
                    chal_reply = (rec.get("chal") or {}).get("reply") or ""
                    if chal_reply.strip():
                        rows.append({
                            **base,
                            "source": "chal",
                            "king_score": king_score,
                            "chal_score": chal_score,
                            "messages": prefix + [
                                {"role": "assistant", "content": chal_reply},
                            ],
                        })
    return rows


def export_dpo(
    paths: list[str],
    *,
    min_chal_score: float,
    min_margin: float,
    require_accept: bool,
) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        meta: dict | None = None
        verdict: dict | None = None
        for rec in _iter_records(path):
            if rec.get("type") == "duel_meta":
                meta = rec
            elif rec.get("type") == "verdict":
                verdict = rec
            elif rec.get("type") == "turn" and _is_valid_turn(rec):
                eval_id = rec.get("eval_id") or (meta or {}).get("eval_id", "?")
                accepted = (verdict or {}).get("accepted")
                if require_accept and not accepted:
                    continue
                prefix = rec.get("messages_prompt") or rec.get("messages_prefix") or []
                king_reply = (rec.get("king") or {}).get("reply") or ""
                chal_reply = (rec.get("chal") or {}).get("reply") or ""
                king_score, chal_score = _best_judge_scores(rec)
                if chal_score < min_chal_score:
                    continue
                if chal_score - king_score < min_margin:
                    continue
                rows.append({
                    **_base_meta(rec, eval_id, accepted),
                    "prompt": prefix,
                    "chosen": chal_reply,
                    "rejected": king_reply,
                    "king_score": king_score,
                    "chal_score": chal_score,
                    "margin": chal_score - king_score,
                })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Export Albedo eval traces to training JSONL")
    p.add_argument("--input", nargs="+", required=True,
                   help="Local .jsonl.gz paths or HTTPS URLs")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--format", choices=("sft", "dpo"), default="sft")
    p.add_argument("--min-chal-score", type=float, default=0.0)
    p.add_argument("--min-margin", type=float, default=0.0,
                   help="DPO only: min chal_score - king_score")
    p.add_argument("--require-accept", action="store_true",
                   help="Only export from accepted duels")
    p.add_argument("--no-gold", action="store_true")
    p.add_argument("--no-king", action="store_true")
    p.add_argument("--no-chal", action="store_true")
    args = p.parse_args()

    paths: list[str] = []
    for spec in args.input:
        if any(c in spec for c in "*?[]"):
            paths.extend(str(p) for p in Path().glob(spec))
        else:
            paths.append(spec)
    if not paths:
        print("No input files matched", file=sys.stderr)
        return 1

    if args.format == "dpo":
        rows = export_dpo(
            paths,
            min_chal_score=args.min_chal_score,
            min_margin=args.min_margin,
            require_accept=args.require_accept,
        )
    else:
        rows = export_sft(
            paths,
            include_gold=not args.no_gold,
            include_king=not args.no_king,
            include_chal=not args.no_chal,
            min_chal_score=args.min_chal_score,
            require_accept=args.require_accept,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
