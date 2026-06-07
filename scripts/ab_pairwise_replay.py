#!/usr/bin/env python3
"""A/B replay: independent rubric scoring vs pairwise head-to-head judging.

Phase 1 (offline): recompute turn/duel outcomes from stored per-judge scores
and quantify cases where independent scoring is ambiguous (equal labels, etc.).

Phase 2 (live): re-judge stored king/challenger reply pairs with a pairwise
prompt via Chutes and compare per-turn winners + simulated duel outcomes.

Usage:
    source .venv/bin/activate
    doppler run -p arbos -c dev -- python scripts/ab_pairwise_replay.py \\
        --manifest-url https://us-east-1.hippius.com/albedo/evals/2026-05-26/manifest.jsonl

    # Offline only (no judge API spend):
    python scripts/ab_pairwise_replay.py --manifest-url ... --offline-only

    # Smaller live sample:
    python scripts/ab_pairwise_replay.py --manifest-url ... \\
        --live-eval-ids eval-0098,eval-0009 --live-max-turns 30
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chain_config
import judge as judge_mod
from eval import dethrone_by_judge_dimensions, judge_dimension_outcome, paired_bootstrap_lcb

PAIRWISE_SYSTEM = """You are a strict code-agent trajectory judge doing a head-to-head comparison.

You will be shown:
  1. The agent's SYSTEM PROMPT.
  2. The CONVERSATION SO FAR (prior turns, excluding the candidate replies).
  3. Two candidate replies for the very next assistant turn: REPLY A and REPLY B.

Pick exactly one outcome:
  - "A"       : reply A is the better next move (more competent, on-task, correct format).
  - "B"       : reply B is the better next move.
  - "tie"     : genuinely indistinguishable in quality, or both are equally flawed.

Reply with strict JSON on one line, no prose around it:

  {"winner": "A" | "B" | "tie", "rationale": "<one short sentence>"}

Do NOT include code fences. Do NOT add extra fields."""

_WINNER_JSON_RE = re.compile(
    r'\{\s*"winner"\s*:\s*"(?:A|B|tie)"[^}]*\}',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class TurnRecord:
    eval_id: str
    sample_idx: int
    turn_idx: int
    context: list[dict]
    king_reply: str
    chal_reply: str
    per_judge: list[dict]
    accepted: bool | None = None


@dataclass
class EvalBundle:
    eval_id: str
    url: str
    accepted: bool
    n_valid_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    verdict: dict | None = None
    meta: dict | None = None


def _open_trace(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = urllib.request.urlopen(path_or_url)
        return gzip.GzipFile(fileobj=resp)
    return gzip.open(path_or_url, "rt", encoding="utf-8")


def _iter_records(path_or_url: str):
    with _open_trace(path_or_url) as f:
        for line in f:
            line = line.decode() if isinstance(line, bytes) else line
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_valid_turn(rec: dict) -> bool:
    if rec.get("type") != "turn" or rec.get("error"):
        return False
    king = (rec.get("king") or {}).get("reply") or ""
    chal = (rec.get("chal") or {}).get("reply") or ""
    return bool(king.strip() and chal.strip())


def load_eval(url: str, *, eval_id: str, accepted: bool) -> EvalBundle:
    bundle = EvalBundle(eval_id=eval_id, url=url, accepted=accepted, n_valid_turns=0)
    for rec in _iter_records(url):
        if rec.get("type") == "duel_meta":
            bundle.meta = rec
        elif rec.get("type") == "verdict":
            bundle.verdict = rec
            bundle.accepted = bool(rec.get("accepted"))
        elif _is_valid_turn(rec):
            bundle.turns.append(
                TurnRecord(
                    eval_id=eval_id,
                    sample_idx=rec["sample_idx"],
                    turn_idx=rec["turn_idx"],
                    context=rec.get("messages_prompt") or rec.get("messages_prefix") or [],
                    king_reply=(rec.get("king") or {}).get("reply") or "",
                    chal_reply=(rec.get("chal") or {}).get("reply") or "",
                    per_judge=rec.get("judges") or [],
                    accepted=bundle.accepted,
                )
            )
    bundle.n_valid_turns = len(bundle.turns)
    return bundle


def _turn_winner_from_scores(king_score: float, chal_score: float) -> str:
    if chal_score > king_score:
        return "chal"
    if king_score > chal_score:
        return "king"
    return "tie"


def _turn_winner_from_labels(king_label: str, chal_label: str) -> str:
    order = {"reject": 0, "weak_pass": 1, "accept": 2}
    ks = order.get(king_label, 0)
    cs = order.get(chal_label, 0)
    if cs > ks:
        return "chal"
    if ks > cs:
        return "king"
    return "tie"


def _duel_outcome_from_judge_deltas(
    per_judge_deltas: dict[str, list[float]],
    *,
    tie_band: float,
    min_turns: int,
    n_valid: int,
) -> tuple[bool, list[str], list[dict]]:
    judge_models = list(per_judge_deltas.keys())
    judge_outcomes: list[str] = []
    judges_final: list[dict] = []
    for jm in judge_models:
        deltas = per_judge_deltas[jm]
        n = max(len(deltas), 1)
        mean_delta = sum(deltas) / n if deltas else 0.0
        outcome = judge_dimension_outcome(mean_delta, tie_band=tie_band)
        judge_outcomes.append(outcome)
        judges_final.append({
            "model": jm,
            "n": len(deltas),
            "delta": mean_delta,
            "outcome": outcome,
        })
    accepted, detail = dethrone_by_judge_dimensions(
        judge_outcomes, min_turns=min_turns, n_done=n_valid, n_valid=n_valid,
    )
    return accepted, judge_outcomes, judges_final


def _duel_outcome_from_pairwise_turns(
    per_judge_winners: dict[str, list[str]],
    *,
    tie_band: float,
    min_turns: int,
    n_valid: int,
) -> tuple[bool, list[str], list[dict]]:
    """Map pairwise {king,tie,chal} per turn to duel outcome via mean signed delta."""
    per_judge_deltas: dict[str, list[float]] = {}
    for jm, winners in per_judge_winners.items():
        per_judge_deltas[jm] = [
            1.0 if w == "chal" else (-1.0 if w == "king" else 0.0)
            for w in winners
        ]
    return _duel_outcome_from_judge_deltas(
        per_judge_deltas, tie_band=tie_band, min_turns=min_turns, n_valid=n_valid,
    )


def build_pairwise_messages(
    context_msgs: list[dict],
    reply_a: str,
    reply_b: str,
) -> list[dict]:
    agent_system = ""
    for m in context_msgs:
        if m.get("role") == "system":
            agent_system = m.get("content") or ""
            break
    conversation = judge_mod._format_conversation(context_msgs)
    user_block = (
        "AGENT SYSTEM PROMPT:\n------\n"
        f"{agent_system}\n------\n\n"
        "CONVERSATION SO FAR:\n------\n"
        f"{conversation}\n------\n\n"
        "REPLY A (candidate next assistant turn):\n------\n"
        f"{reply_a.rstrip()}\n------\n\n"
        "REPLY B (candidate next assistant turn):\n------\n"
        f"{reply_b.rstrip()}\n------\n\n"
        'Respond with strict JSON: {"winner": "A"|"B"|"tie", "rationale": "..."}'
    )
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": user_block},
    ]


def parse_pairwise_winner(raw: str) -> tuple[str, str, bool]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    for match in _WINNER_JSON_RE.finditer(raw):
        candidates.append(match.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        w = str(obj.get("winner", "")).strip().upper()
        if w == "A":
            return "A", str(obj.get("rationale", ""))[:500], True
        if w == "B":
            return "B", str(obj.get("rationale", ""))[:500], True
        if w.lower() == "tie" or obj.get("winner", "").lower() == "tie":
            return "tie", str(obj.get("rationale", ""))[:500], True
    return "tie", "parse_failure", False


def _pairwise_side(
    winner: str,
    *,
    a_is_king: bool,
) -> str:
    if winner == "tie":
        return "tie"
    if winner == "A":
        return "king" if a_is_king else "chal"
    if winner == "B":
        return "chal" if a_is_king else "king"
    return "tie"


async def pairwise_judge_turn(
    client: judge_mod.ChutesJudge,
    turn: TurnRecord,
    *,
    model: str,
    seed: bytes,
) -> dict:
    key = f"{turn.eval_id}:{turn.sample_idx}:{turn.turn_idx}:{model}"
    h = int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)
    a_is_king = bool(h & 1)
    if a_is_king:
        reply_a, reply_b = turn.king_reply, turn.chal_reply
    else:
        reply_a, reply_b = turn.chal_reply, turn.king_reply

    messages = build_pairwise_messages(turn.context, reply_a, reply_b)
    use_model = model
    max_tokens = judge_mod._max_tokens_for_model(use_model)
    body = {
        "model": use_model,
        "messages": messages,
        "temperature": chain_config.JUDGE_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    delay = chain_config.JUDGE_RETRY_INITIAL_BACKOFF_S
    last_exc: Exception | None = None
    for attempt in range(chain_config.JUDGE_RETRY_MAX + 1):
        try:
            resp = await client._client.post("/chat/completions", json=body)
        except Exception as exc:
            last_exc = exc
        else:
            if resp.status_code < 400:
                try:
                    choice = resp.json()["choices"][0]
                    text = judge_mod._message_text(choice["message"])
                except Exception:
                    text = ""
                winner_raw, rationale, parse_ok = parse_pairwise_winner(text)
                side = _pairwise_side(winner_raw, a_is_king=a_is_king)
                return {
                    "model": model,
                    "winner": side,
                    "winner_raw": winner_raw,
                    "a_is_king": a_is_king,
                    "rationale": rationale,
                    "parse_ok": parse_ok,
                    "raw": text[:2000],
                }
            if resp.status_code in (408, 425, 429) or 500 <= resp.status_code < 600:
                last_exc = Exception(f"http {resp.status_code}")
            else:
                return {
                    "model": model,
                    "winner": "tie",
                    "winner_raw": "tie",
                    "a_is_king": a_is_king,
                    "rationale": f"http_{resp.status_code}",
                    "parse_ok": False,
                    "raw": resp.text[:500],
                }
        if attempt >= chain_config.JUDGE_RETRY_MAX:
            break
        await asyncio.sleep(delay * (0.5 + random.random()))
        delay *= 2
    return {
        "model": model,
        "winner": "tie",
        "winner_raw": "tie",
        "a_is_king": a_is_king,
        "rationale": f"retries_exhausted: {last_exc}",
        "parse_ok": False,
        "raw": "",
    }


def offline_analysis(bundles: list[EvalBundle], *, tie_band: float) -> dict:
    min_turns = max(8, chain_config.DUEL_N_SAMPLES // 4)
    turn_stats = Counter()
    label_vs_score_mismatch = 0
    same_label_diff_score = 0
    duel_flips_vs_stored: list[dict] = []

    for bundle in bundles:
        per_judge_deltas: dict[str, list[float]] = defaultdict(list)
        for turn in bundle.turns:
            for pj in turn.per_judge:
                jm = pj["model"]
                ks = float(pj.get("king_score", 0.0))
                cs = float(pj.get("chal_score", 0.0))
                per_judge_deltas[jm].append(cs - ks)
                score_w = _turn_winner_from_scores(ks, cs)
                label_w = _turn_winner_from_labels(
                    pj.get("king_verdict", "reject"),
                    pj.get("chal_verdict", "reject"),
                )
                if score_w != label_w:
                    label_vs_score_mismatch += 1
                if pj.get("king_verdict") == pj.get("chal_verdict") and score_w != "tie":
                    same_label_diff_score += 1
                turn_stats["turns"] += 1
                if score_w == "tie":
                    turn_stats["score_tie"] += 1
                if label_w == "tie":
                    turn_stats["label_tie"] += 1
                if pj.get("king_verdict") == pj.get("chal_verdict"):
                    turn_stats["same_verdict_label"] += 1

        recomputed, outcomes, _ = _duel_outcome_from_judge_deltas(
            dict(per_judge_deltas),
            tie_band=tie_band,
            min_turns=min_turns,
            n_valid=bundle.n_valid_turns,
        )
        stored = bool((bundle.verdict or {}).get("accepted"))
        if recomputed != stored:
            duel_flips_vs_stored.append({
                "eval_id": bundle.eval_id,
                "stored_accepted": stored,
                "recomputed_accepted": recomputed,
                "outcomes": outcomes,
                "n_turns": bundle.n_valid_turns,
            })

    return {
        "n_evals": len(bundles),
        "n_turns": turn_stats["turns"],
        "score_tie_rate": turn_stats["score_tie"] / max(turn_stats["turns"], 1),
        "label_tie_rate": turn_stats["label_tie"] / max(turn_stats["turns"], 1),
        "same_verdict_label_rate": turn_stats["same_verdict_label"] / max(turn_stats["turns"], 1),
        "same_label_diff_score": same_label_diff_score,
        "label_vs_score_mismatch": label_vs_score_mismatch,
        "recompute_mismatch_evals": duel_flips_vs_stored,
    }


async def live_pairwise_analysis(
    bundles: list[EvalBundle],
    *,
    judge_model: str,
    tie_band: float,
    max_turns: int | None,
    concurrency: int,
) -> dict:
    min_turns = max(8, chain_config.DUEL_N_SAMPLES // 4)
    sem = asyncio.Semaphore(concurrency)
    turn_agree = 0
    turn_disagree = 0
    turn_parse_fail = 0
    disagree_examples: list[dict] = []
    duel_rows: list[dict] = []

    async with judge_mod.ChutesJudge(model=judge_model) as client:
        for bundle in bundles:
            turns = bundle.turns
            if max_turns and len(turns) > max_turns:
                rng = random.Random(bundle.eval_id)
                turns = rng.sample(turns, max_turns)

            stored_deltas: dict[str, list[float]] = defaultdict(list)
            pairwise_winners: dict[str, list[str]] = defaultdict(list)

            async def _one(turn: TurnRecord) -> tuple[TurnRecord, dict]:
                async with sem:
                    live = await pairwise_judge_turn(
                        client, turn, model=judge_model,
                        seed=f"{turn.eval_id}|{turn.sample_idx}|{turn.turn_idx}".encode(),
                    )
                return turn, live

            results = await asyncio.gather(*[_one(t) for t in turns])
            for turn, live in results:
                pj = next((j for j in turn.per_judge if j["model"] == judge_model), None)
                if not pj:
                    pj = turn.per_judge[0] if turn.per_judge else {}
                ks = float(pj.get("king_score", 0.0))
                cs = float(pj.get("chal_score", 0.0))
                stored_deltas[judge_model].append(cs - ks)
                score_w = _turn_winner_from_scores(ks, cs)
                live_w = live["winner"]
                pairwise_winners[judge_model].append(live_w)
                if not live.get("parse_ok", True):
                    turn_parse_fail += 1
                if score_w == live_w:
                    turn_agree += 1
                else:
                    turn_disagree += 1
                    if len(disagree_examples) < 8:
                        disagree_examples.append({
                            "eval_id": turn.eval_id,
                            "sample_idx": turn.sample_idx,
                            "turn_idx": turn.turn_idx,
                            "stored_king_verdict": pj.get("king_verdict"),
                            "stored_chal_verdict": pj.get("chal_verdict"),
                            "stored_scores": (ks, cs),
                            "stored_winner": score_w,
                            "pairwise_winner": live_w,
                            "pairwise_rationale": live.get("rationale"),
                        })

            stored_accepted, _, _ = _duel_outcome_from_judge_deltas(
                {judge_model: stored_deltas[judge_model]},
                tie_band=tie_band,
                min_turns=min_turns,
                n_valid=len(stored_deltas[judge_model]),
            )
            pairwise_accepted, pw_outcomes, _ = _duel_outcome_from_pairwise_turns(
                {judge_model: pairwise_winners[judge_model]},
                tie_band=tie_band,
                min_turns=min_turns,
                n_valid=len(pairwise_winners[judge_model]),
            )
            verdict_accepted = bool((bundle.verdict or {}).get("accepted"))
            duel_rows.append({
                "eval_id": bundle.eval_id,
                "n_turns_replayed": len(turns),
                "verdict_accepted": verdict_accepted,
                "stored_single_judge_accepted": stored_accepted,
                "pairwise_single_judge_accepted": pairwise_accepted,
                "pairwise_outcomes": pw_outcomes,
                "stored_judge_outcome": judge_dimension_outcome(
                    sum(stored_deltas[judge_model]) / max(len(stored_deltas[judge_model]), 1),
                    tie_band=tie_band,
                ),
            })

    n = max(turn_agree + turn_disagree, 1)
    return {
        "judge_model": judge_model,
        "n_evals": len(bundles),
        "n_turns": turn_agree + turn_disagree,
        "turn_agreement_rate": turn_agree / n,
        "turn_disagree": turn_disagree,
        "turn_parse_fail": turn_parse_fail,
        "disagree_examples": disagree_examples,
        "duel_rows": duel_rows,
        "duel_flip_count": sum(
            1 for r in duel_rows
            if r["stored_single_judge_accepted"] != r["pairwise_single_judge_accepted"]
        ),
    }


def _load_manifest(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _pick_live_evals(manifest: list[dict], explicit: str | None) -> list[dict]:
    if explicit:
        want = set(explicit.split(","))
        return [e for e in manifest if e.get("eval_id") in want]
    accepted = [e for e in manifest if e.get("accepted") and str(e.get("eval_id", "")).startswith("eval-")]
    rejected = [
        e for e in manifest
        if not e.get("accepted")
        and str(e.get("eval_id", "")).startswith("eval-")
        and int(e.get("n_valid_turns") or 0) >= 30
    ]
    # Fixed seed for reproducible pick.
    rng = random.Random(42)
    pick = rng.sample(accepted, min(3, len(accepted))) + rng.sample(rejected, min(3, len(rejected)))
    return pick


async def _async_main(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest_url)
    eval_entries = [
        e for e in manifest
        if str(e.get("eval_id", "")).startswith("eval-")
        and int(e.get("n_valid_turns") or 0) >= args.min_valid_turns
    ]
    print(f"manifest evals (n_valid>={args.min_valid_turns}): {len(eval_entries)}")

    print("\n=== Loading traces for offline analysis ===")
    bundles: list[EvalBundle] = []
    for entry in eval_entries:
        url = entry.get("url") or ""
        if not url:
            continue
        try:
            bundles.append(load_eval(
                url,
                eval_id=entry["eval_id"],
                accepted=bool(entry.get("accepted")),
            ))
        except Exception as exc:
            print(f"  skip {entry['eval_id']}: {exc}")

    tie_band = chain_config.JUDGE_TIE_BAND
    offline = offline_analysis(bundles, tie_band=tie_band)
    print("\n=== OFFLINE (stored independent scores) ===")
    print(json.dumps({
        k: v for k, v in offline.items()
        if k != "recompute_mismatch_evals"
    }, indent=2))
    if offline["recompute_mismatch_evals"]:
        print("recompute mismatches:", json.dumps(offline["recompute_mismatch_evals"], indent=2))
    else:
        print("recompute mismatches: none (stored verdicts match replay from trace scores)")

    if args.offline_only:
        return 0

    live_entries = _pick_live_evals(manifest, args.live_eval_ids)
    print(f"\n=== LIVE pairwise replay on {len(live_entries)} evals ===")
    for e in live_entries:
        print(f"  {e['eval_id']} accepted={e.get('accepted')} n_valid={e.get('n_valid_turns')}")

    live_bundles = [
        load_eval(e["url"], eval_id=e["eval_id"], accepted=bool(e.get("accepted")))
        for e in live_entries
    ]
    live = await live_pairwise_analysis(
        live_bundles,
        judge_model=args.judge_model,
        tie_band=tie_band,
        max_turns=args.live_max_turns,
        concurrency=args.concurrency,
    )
    print("\n=== LIVE pairwise vs stored score winner ===")
    print(json.dumps({
        k: v for k, v in live.items()
        if k not in ("disagree_examples", "duel_rows")
    }, indent=2))
    print("\nduel_rows:")
    print(json.dumps(live["duel_rows"], indent=2))
    if live["disagree_examples"]:
        print("\ndisagree_examples:")
        print(json.dumps(live["disagree_examples"], indent=2))

    out_path = Path(args.output) if args.output else None
    if out_path:
        out_path.write_text(json.dumps({"offline": offline, "live": live}, indent=2))
        print(f"\nWrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="A/B independent vs pairwise judge replay")
    p.add_argument(
        "--manifest-url",
        default="https://us-east-1.hippius.com/albedo/evals/2026-05-26/manifest.jsonl",
    )
    p.add_argument("--offline-only", action="store_true")
    p.add_argument("--live-eval-ids", default=None,
                   help="Comma-separated eval ids for live replay (default: 3 accepted + 3 rejected)")
    p.add_argument("--live-max-turns", type=int, default=None,
                   help="Cap turns per eval for live replay (default: all valid turns)")
    p.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V3.2-TEE")
    p.add_argument("--min-valid-turns", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", default="/tmp/albedo-ab/results.json")
    args = p.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
