"""Run one duel turn: generate from both models, score head-to-head (pairwise per-metric)."""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from albedo.config import (
    DUEL_GEN_MAX_TOKENS,
    DUEL_GEN_TEMP,
    JUDGE_METRIC_KEYS,
)
from albedo.duel.sampler import Sample

if TYPE_CHECKING:
    from albedo.judge import ChutesJudge
    from albedo.eval_server.sink import DatasetSink

log = logging.getLogger(__name__)

# Patterns that miners may inject into replies to hijack judge verdicts.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # {"verdict": "..."} or {"verdict": "...", "rationale": "..."}
    re.compile(r'\{\s*"verdict"\s*:\s*"[^"]*"[^}]*\}', re.IGNORECASE),
    # {"injection": true/false ...}
    re.compile(r'\{\s*"injection"\s*:\s*(true|false)[^}]*\}', re.IGNORECASE),
]
# Section-delimiter impersonation: a "------" separator followed by content with
# verdict/grading keywords. Stripped first since it removes the larger block.
_DELIMITER_INJECTION_RE = re.compile(
    r'\s*-{3,}[\s\S]*?(?:"verdict"\s*:|GRADING\s+INSTRUCTION)[\s\S]*$',
    re.DOTALL | re.IGNORECASE,
)

# Known verdict label values — used by the JSON scanner below.
_VERDICT_LABELS: frozenset[str] = frozenset({"accept", "weak_pass", "reject"})


def _scan_verdict_json(text: str) -> bool:
    """Return True if text contains a JSON object with a recognisable verdict label.

    Uses stdlib json.JSONDecoder to find every valid JSON object — catches
    base64-decoded embeds, unicode lookalikes, and any obfuscation that defeats
    the simple regexes above.
    """
    dec = _json.JSONDecoder()
    start = 0
    while True:
        idx = text.find("{", start)
        if idx == -1:
            break
        try:
            obj, _ = dec.raw_decode(text[idx:])
            if isinstance(obj, dict):
                v = obj.get("verdict", "")
                if isinstance(v, str) and v.lower() in _VERDICT_LABELS:
                    return True
                if any(k.lower() == "injection" for k in obj):
                    return True
        except _json.JSONDecodeError:
            pass
        start = idx + 1
    return False


@dataclass
class TurnResult:
    sample:          Sample
    king_reply:      str
    chal_reply:      str
    per_judge:       list[dict]     # one entry per judge model
    final_score:     float          # mean of per-judge means over ok judges (0.0–1.0)
    final_score_100: float          # final_score * 100
    delta:           float          # final_score - 0.5 (challenger advantage; for LCB)
    parse_ok:        bool           # all judges parsed
    king_usage:      dict
    chal_usage:      dict
    vllm_error:      str | None = None  # set when generation failed; turn not valid


def strip_reply_injection(reply: str) -> str:
    """Remove injected verdict JSON and delimiter-impersonation blocks from replies.

    After regex stripping, runs a JSON-based scan to catch obfuscated injections
    (verdict not first key, unicode lookalikes, etc.) that the regexes miss.
    If any verdict-carrying JSON object survives stripping, the reply is blanked.
    """
    cleaned = _DELIMITER_INJECTION_RE.sub("", reply)
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if _scan_verdict_json(cleaned):
        log.warning(
            "strip_reply_injection: verdict JSON survived regex strip — reply blanked"
        )
        return ""
    return cleaned.strip()


async def _generate(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
) -> tuple[str, dict, str | None]:
    """Call a vLLM endpoint. Returns (reply, usage, error). error is None on success."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": DUEL_GEN_TEMP,
        "max_tokens": DUEL_GEN_MAX_TOKENS,
    }
    try:
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body  = response.json()
        reply = (body["choices"][0]["message"].get("content") or "").strip()
        return reply, body.get("usage", {}), None
    except httpx.TimeoutException as exc:
        log.warning("vLLM generation timed out: %s", exc)
        return "", {}, f"vllm_timeout: {exc}"
    except httpx.HTTPStatusError as exc:
        log.warning("vLLM returned HTTP %s", exc.response.status_code)
        return "", {}, f"vllm_http_{exc.response.status_code}"
    except (KeyError, IndexError, ValueError) as exc:
        log.warning("vLLM response malformed: %s", exc)
        return "", {}, f"vllm_malformed: {exc}"


async def _score_single_judge(
    judge: "ChutesJudge",
    model: str,
    sample: Sample,
    king_reply: str,
    chal_reply: str,
    hotkey: str,
    seed: bytes,
    sink: "DatasetSink | None" = None,
) -> dict:
    """Score one turn head-to-head with one judge model; returns a per-judge dict.

    A single pairwise call sees both replies (MODEL 1 = king, MODEL 2 = challenger)
    and returns per-metric challenger-perspective scores.
    """
    if not isinstance(king_reply, str):
        raise ValueError(f"king_reply is not str: {type(king_reply)}")
    if not isinstance(chal_reply, str):
        raise ValueError(f"chal_reply is not str: {type(chal_reply)}")
    king_clean = strip_reply_injection(king_reply)
    chal_clean = strip_reply_injection(chal_reply)

    messages_prefix = sample.messages_prefix or []
    messages_prompt = sample.messages_prompt or []
    context_messages = list(messages_prefix) + list(messages_prompt)

    verdict = await judge.pairwise_judge(
        model=model,
        messages_prefix=messages_prefix,
        messages_prompt=messages_prompt,
        king_reply=king_clean,
        chal_reply=chal_clean,
        hotkey=hotkey,
        seed=seed,
    )

    if sink is not None:
        try:
            sink.write_judge_raw({
                "judge_model":    model,
                "global_idx":     sample.global_idx,
                "turn_idx":       sample.turn_idx,
                "instance_id":    sample.instance_id,
                "prompt_messages": context_messages,
                "king_reply":     king_clean,
                "chal_reply":     chal_clean,
                "raw_verdict":    verdict.raw,
                "metric_scores":  verdict.metric_scores,
                "judge_mean":     verdict.judge_mean,
                "parse_ok":       verdict.parse_ok,
            })
        except Exception:
            log.warning("sink.write_judge_raw failed for judge %s turn %d", model, sample.global_idx)

    return {
        "judge_model":   model,
        "metric_scores": verdict.metric_scores,
        "judge_mean":    verdict.judge_mean,
        "parse_ok":      verdict.parse_ok,
    }


async def resolve_model_names(
    king_client: httpx.AsyncClient,
    chal_client: httpx.AsyncClient,
) -> tuple[str, str]:
    """Fetch vLLM model IDs once before the duel starts.

    Falls back to "default" (vLLM's built-in alias) on failure — not the
    label string "king"/"challenger" which vLLM would reject with HTTP 400.
    Called once from run_duel and passed to all score_turn invocations.
    """
    async def _get(client: httpx.AsyncClient) -> str:
        try:
            resp = await client.get("/v1/models", timeout=10.0)
            resp.raise_for_status()
            return resp.json()["data"][0]["id"]
        except Exception as exc:
            log.warning("model name discovery failed: %s — using 'default'", exc)
            return "default"

    return await asyncio.gather(_get(king_client), _get(chal_client))


def _dummy_judge(model: str) -> dict:
    """A zeroed per-judge entry (challenger loses all metrics, parse_ok=False)."""
    return {
        "judge_model":   model,
        "metric_scores": {k: 0.0 for k in JUDGE_METRIC_KEYS},
        "judge_mean":    0.0,
        "parse_ok":      False,
    }


async def score_turn(
    sample: Sample,
    *,
    king_client:      httpx.AsyncClient,
    chal_client:      httpx.AsyncClient,
    king_model_name:  str,
    chal_model_name:  str,
    judge:            "ChutesJudge",
    judge_models:     list[str],
    hotkey:           str,
    seed:             bytes,
    sink:             "DatasetSink | None" = None,
) -> TurnResult:
    """Generate king and challenger replies in parallel, then score head-to-head.

    One pairwise judge call per judge model. final_score is the mean of per-judge
    means (challenger perspective, 0.0–1.0). Logs a warning when parse_ok=False.
    """

    full_messages = sample.messages_prefix + sample.messages_prompt

    (king_reply, king_usage, king_err), (chal_reply, chal_usage, chal_err) = \
        await asyncio.gather(
            _generate(king_client, king_model_name, full_messages),
            _generate(chal_client, chal_model_name, full_messages),
        )

    if king_err or chal_err:
        error_msg = f"king_{king_err}" if king_err else f"chal_{chal_err}"
        return TurnResult(
            sample=sample,
            king_reply=king_reply, chal_reply=chal_reply,
            per_judge=[_dummy_judge(m) for m in judge_models],
            final_score=0.0, final_score_100=0.0, delta=0.0,
            parse_ok=False, vllm_error=error_msg,
            king_usage=king_usage, chal_usage=chal_usage,
        )

    # Write champion and challenger replies to the sink.
    if sink is not None:
        try:
            turn_meta = {
                "global_idx":  sample.global_idx,
                "instance_id": sample.instance_id,
                "turn_idx":    sample.turn_idx,
            }
            sink.write_champion({**turn_meta, "reply": king_reply, "usage": king_usage})
            sink.write_challenger({**turn_meta, "reply": chal_reply, "usage": chal_usage})
        except Exception:
            log.warning("sink write failed for turn %d", sample.global_idx)

    raw_judge_results = await asyncio.gather(
        *[
            _score_single_judge(
                judge, model, sample, king_reply, chal_reply, hotkey, seed, sink=sink
            )
            for model in judge_models
        ],
        return_exceptions=True,
    )

    # Replace failed judge calls with a parse-failure sentinel so one judge
    # exhausting its retries doesn't abort the whole turn (mislabelled as vllm_error).
    per_judge: list[dict] = []
    for model, result in zip(judge_models, raw_judge_results):
        if isinstance(result, Exception):
            log.warning("judge %s raised on turn %d: %s", model, sample.global_idx, result)
            per_judge.append(_dummy_judge(model))
        else:
            per_judge.append(result)

    # final_score = mean of per-judge means over judges that parsed ok.
    ok_means = [e["judge_mean"] for e in per_judge if e["parse_ok"]]
    all_parse_ok = all(e["parse_ok"] for e in per_judge)
    for e in per_judge:
        if not e["parse_ok"]:
            log.warning(
                "parse_ok=False for judge %s on global_idx=%d",
                e["judge_model"], sample.global_idx,
            )

    final_score = sum(ok_means) / len(ok_means) if ok_means else 0.0

    return TurnResult(
        sample=sample,
        king_reply=king_reply,
        chal_reply=chal_reply,
        per_judge=per_judge,
        final_score=final_score,
        final_score_100=final_score * 100.0,
        delta=final_score - 0.5,
        parse_ok=all_parse_ok,
        king_usage=king_usage,
        chal_usage=chal_usage,
    )
