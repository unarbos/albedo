"""albedo.judge.client — unified LLM-as-judge transport.

query_judges(models, messages, accept, max_tokens_fn) is the one task-level primitive used
by BOTH the duel judges (turn.py) and the pre-eval injection probe (injection.py):

  PER-JUDGE SEQUENTIAL: resolve judges one at a time, fully finishing one before
  starting the next (judge1 -> judge2 -> judge3). For each judge:
    - Chutes, stream-gated: open a streaming request; within JUDGE_CHUTES_TRY_S decide
      accept vs reject (429/non-200/no first chunk -> reject); if accepted, read to
      completion within JUDGE_CHUTES_MAX_S and keep iff accept(raw).
    - on Chutes failure, fall back to OpenRouter for THAT judge (retrying on
      network/429/parse-fail). The judge's whole Chutes+OR budget is bounded by
      JUDGE_TOTAL_S, after which it resolves to None.
  Circuit-breaker: after JUDGE_CHUTES_GIVEUP_TASKS consecutive tasks with no Chutes
    success, this instance turns Chutes off (OpenRouter-only) for the rest of the eval.

Returns {model: raw_text | None}; callers parse raw with their own validator. There is no
whole-duel deadline cap here (that lived in the old _chat and caused cascade failures); the
duel-level budget lives in duel/stream.py and the validator hard-timeout is the last resort.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Callable

import httpx

from albedo.config import (
    JUDGE_API_KEY_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MAX_TOKENS,
    JUDGE_METRIC_KEYS,
    JUDGE_RETRY_BACKOFF,
    JUDGE_SCORE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    JUDGE_THINKING_MODELS,
    JUDGE_THINKING_TOKENS,
    JUDGE_CHUTES_TRY_S,
    JUDGE_CHUTES_MAX_S,
    JUDGE_OR_TIMEOUT_S,
    JUDGE_OR_RETRIES,
    JUDGE_TOTAL_S,
    JUDGE_CHUTES_GIVEUP_TASKS,
    JUDGE_MAX_CONCURRENCY_PER_MODEL,
    JUDGE_FALLBACK_ENABLED,
    JUDGE_FALLBACK_BASE_URL,
    JUDGE_FALLBACK_API_KEY_ENV,
    JUDGE_FALLBACK_MODEL_MAP,
    JUDGE_FALLBACK_REASONING_MODELS,
    JUDGE_RATE_LIMITS,
)
from albedo.eval_server.notifications import notify_problem


from albedo.judge.rubric import PAIRWISE_RUBRIC_SYSTEM, PROBE_SYSTEM, build_pairwise_user
from albedo.judge.verdict import MetricVerdict, parse_metric_verdict

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://llm.chutes.ai"


def _chutes_enabled() -> bool:
    return os.environ.get("ALBEDO_JUDGE_CHUTES_ENABLED", "1").lower() not in ("0", "false", "no", "off")


def _is_thinking(model: str) -> bool:
    return model in JUDGE_THINKING_MODELS


def _max_tokens_for(model: str) -> int:
    return JUDGE_THINKING_TOKENS if _is_thinking(model) else JUDGE_SCORE_MAX_TOKENS


def _message_content(choices: list[dict]) -> str:
    """Return only the content field from the first choice message."""
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content") or ""


def _sse_delta(line: str) -> tuple[str, str, bool]:
    """Parse one SSE 'data:' line -> (content, reasoning, done)."""
    line = line.strip()
    if not line.startswith("data:"):
        return "", "", False
    body = line[len("data:"):].strip()
    if body == "[DONE]":
        return "", "", True
    try:
        d = json.loads(body)
        delta = (d.get("choices") or [{}])[0].get("delta", {}) or {}
        return (delta.get("content") or ""), (delta.get("reasoning_content") or delta.get("reasoning") or ""), False
    except Exception:
        return "", "", False


_INJECTION_RE = re.compile(r'\{[^{}]*"injection"\s*:\s*(?:true|false)[^{}]*\}', re.DOTALL)


def _parse_injection(raw: str) -> bool | None:
    """Extract the injection bool from a probe verdict; None if unparseable."""
    obj = None
    matches = _INJECTION_RE.findall(raw or "")
    if matches:
        try:
            obj = json.loads(matches[-1])
        except Exception:
            obj = None
    if obj is None:
        s = (raw or "").strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
    return None if obj is None else bool(obj.get("injection", False))


def _injection_accept(raw: str) -> bool:
    return _parse_injection(raw) is not None


def _parse_injection_evidence(raw: str) -> str | None:
    """Extract the evidence field from a probe verdict if present."""
    obj = None
    matches = _INJECTION_RE.findall(raw or "")
    if matches:
        try:
            obj = json.loads(matches[-1])
        except Exception:
            obj = None
    if obj is None:
        s = (raw or "").strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
    if obj is None:
        return None
    evidence = obj.get("evidence")
    return None if evidence is None else str(evidence)


class ChutesJudge:
    """Unified judge client — one instance per eval (duel or probe); shared across models."""

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = (base_url or os.environ.get(JUDGE_BASE_URL_ENV, "") or _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get(JUDGE_API_KEY_ENV, "")
        ch_headers = {"Authorization": f"Bearer {self._api_key}"}
        # Chutes clients (regular vs thinking read budgets); stream calls override read=CHUTES_MAX_S.
        self._chutes = httpx.AsyncClient(
            base_url=self._base_url, headers=ch_headers,
            timeout=httpx.Timeout(connect=10.0, read=JUDGE_CHUTES_MAX_S, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        self._chutes_think = self._chutes  # same budget; stream gate bounds it

        # OpenRouter fallback clients.
        self._fb_key = os.environ.get(JUDGE_FALLBACK_API_KEY_ENV, "")
        self._fb_enabled = bool(JUDGE_FALLBACK_ENABLED and self._fb_key)
        self._fb = self._fb_think = None
        if self._fb_enabled:
            fb_headers = {"Authorization": f"Bearer {self._fb_key}"}
            fb_base = JUDGE_FALLBACK_BASE_URL.rstrip("/")
            self._fb = httpx.AsyncClient(
                base_url=fb_base, headers=fb_headers,
                timeout=httpx.Timeout(connect=10.0, read=JUDGE_OR_TIMEOUT_S, write=30.0, pool=10.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
            self._fb_think = self._fb

        # OpenRouter per-model concurrency (lazy, created in the running loop).
        self._or_sems: dict[str, asyncio.Semaphore] = {}

        # Circuit-breaker state (per eval instance).
        self._chutes_dry_tasks = 0
        self._chutes_off = not _chutes_enabled()
        if self._chutes_off:
            log.info("ChutesJudge: Chutes disabled via ALBEDO_JUDGE_CHUTES_ENABLED=0")

    # ----- helpers -----

    def _or_sem(self, model: str) -> asyncio.Semaphore:
        s = self._or_sems.get(model)
        if s is None:
            limit_cfg = JUDGE_RATE_LIMITS.get(model, {}) if isinstance(JUDGE_RATE_LIMITS, dict) else {}
            limit = int(limit_cfg.get("max_concurrency", JUDGE_MAX_CONCURRENCY_PER_MODEL))
            s = self._or_sems[model] = asyncio.Semaphore(max(1, limit))
        return s

    def _or_model(self, model: str) -> str | None:
        return JUDGE_FALLBACK_MODEL_MAP.get(model)

    def _build_pairwise_messages(self, context_messages, king_reply, chal_reply) -> list[dict]:
        return [
            {"role": "system", "content": PAIRWISE_RUBRIC_SYSTEM},
            {"role": "user", "content": build_pairwise_user(context_messages, king_reply, chal_reply)},
        ]

    def _build_probe_messages(self, messages, reply) -> list[dict]:
        combined = json.dumps({"conversation": messages, "candidate_reply": reply}, ensure_ascii=False)
        return [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": combined}]

    # ----- providers -----

    async def _chutes_stream(self, model: str, messages: list[dict], max_tokens: int) -> str | None:
        """Stream-gated Chutes. Return raw text if Chutes ACCEPTS and completes, else None."""
        client = self._chutes_think if _is_thinking(model) else self._chutes
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": JUDGE_TEMPERATURE,
            "max_tokens": max_tokens, "stream": True,
        }

        async def _run() -> str | None:
            async with client.stream(
                "POST", "/v1/chat/completions", json=payload,
                timeout=httpx.Timeout(connect=10.0, read=JUDGE_CHUTES_MAX_S, write=30.0, pool=10.0),
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    await notify_problem(
                        title=f"Chutes judge HTTP {resp.status_code}",
                        message=f"Chutes returned HTTP {resp.status_code} for judge model `{model}`.",
                        dedupe_key=f"judge:chutes:{model}:{resp.status_code}:{body[:120]}",
                        details=body,
                        status_code=resp.status_code,
                        source="judge",
                    )
                    return None  # 429 / capacity / error -> Chutes not taking it
                agen = resp.aiter_lines()

                async def _first() -> str | None:
                    async for ln in agen:
                        if ln.strip().startswith("data:"):
                            return ln
                    return None

                # accept/reject gate: first streamed chunk within JUDGE_CHUTES_TRY_S
                try:
                    first = await asyncio.wait_for(_first(), timeout=JUDGE_CHUTES_TRY_S)
                except asyncio.TimeoutError:
                    return None
                if first is None:
                    return None
                # ACCEPTED — read to completion (bounded by the outer wait_for)
                cparts: list[str] = []
                rparts: list[str] = []
                c, r, done = _sse_delta(first)
                cparts.append(c); rparts.append(r)
                if not done:
                    async for ln in agen:
                        c, r, done = _sse_delta(ln)
                        cparts.append(c); rparts.append(r)
                        if done:
                            break
                return "".join(rparts) + "\n" + "".join(cparts)

        try:
            return await asyncio.wait_for(_run(), timeout=JUDGE_CHUTES_MAX_S)
        except asyncio.TimeoutError:
            log.warning("chutes_stream %s: accepted but exceeded %.0fs — falling back", model, JUDGE_CHUTES_MAX_S)
            return None
        except Exception as exc:
            log.debug("chutes_stream %s rejected/error: %s", model, exc)
            return None

    async def _openrouter(self, model: str, messages: list[dict], max_tokens: int, *,
                          accept: Callable[[str], bool], deadline: float) -> str | None:
        """OpenRouter, retried on any non-2xx (4xx/5xx), no-response/network error, or
        parse-fail, up to JUDGE_OR_RETRIES times, the whole phase capped by `deadline`."""
        if not self._fb_enabled:
            return None
        or_model = self._or_model(model)
        if not or_model:
            return None
        client = self._fb_think if _is_thinking(model) else self._fb
        payload: dict[str, Any] = {
            "model": or_model, "messages": messages, "temperature": JUDGE_TEMPERATURE, "max_tokens": max_tokens,
        }
        # Models not in reasoning_models get reasoning excluded so their full max_tokens
        # budget goes to content. deepseek-v4-flash (in reasoning_models) reasons freely —
        # we read only content, so reasoning tokens never corrupt the verdict.
        if model not in JUDGE_FALLBACK_REASONING_MODELS:
            payload["reasoning"] = {"enabled": False, "exclude": True}

        sem = self._or_sem(model)
        attempt = 0
        last = ""
        while time.monotonic() < deadline and attempt <= JUDGE_OR_RETRIES:
            await sem.acquire()
            try:
                t = min(JUDGE_OR_TIMEOUT_S, max(1.0, deadline - time.monotonic()))
                resp = await client.post("/v1/chat/completions", json=payload, timeout=t)
                resp.raise_for_status()
                raw = _message_content(resp.json().get("choices", []))
                if accept(raw):
                    return raw
                last = "parse_fail"
            except httpx.HTTPStatusError as exc:
                last = f"http_{exc.response.status_code}"
                await notify_problem(
                    title=f"OpenRouter judge HTTP {exc.response.status_code}",
                    message=f"OpenRouter returned HTTP {exc.response.status_code} for judge model `{model}`.",
                    dedupe_key=(
                        f"judge:openrouter:{model}:{exc.response.status_code}:{exc.response.text[:120]}"
                    ),
                    details=exc.response.text,
                    status_code=exc.response.status_code,
                    source="judge",
                )
            except Exception as exc:
                last = type(exc).__name__
            finally:
                sem.release()
            attempt += 1
            wait = min(JUDGE_RETRY_BACKOFF * (2 ** attempt), 8.0) * random.uniform(0.8, 1.3)
            wait = min(wait, max(0.0, deadline - time.monotonic()))
            if wait > 0:
                await asyncio.sleep(wait)
        log.warning("openrouter %s exhausted (%s)", model, last)
        return None

    # ----- task-level orchestrator -----

    async def query_judges(self, models: list[str], messages: list[dict], *,
                       accept: Callable[[str], bool],
                       max_tokens_fn: Callable[[str], int] = _max_tokens_for) -> dict[str, str | None]:
        """All judges run CONCURRENTLY — asyncio.gather fires all 3 at once.

        Each judge independently: Chutes stream-gate -> OpenRouter fallback.
        If DeepSeek is rate-limited (429), Qwen3 and Kimi finish independently
        rather than waiting behind it. Total per-turn latency = max(judge times),
        not sum. Returns {model: raw|None}.
        """

        async def _resolve_one(m: str) -> tuple[str, str | None, bool]:
            """Race Chutes and OR simultaneously — first valid response wins.

            Both providers are started at the same time. Whichever returns a
            parseable verdict first cancels the other. This eliminates the
            sequential fallback delay: OR starts immediately, not after Chutes
            fails its 2s gate.
            """
            deadline = time.monotonic() + JUDGE_TOTAL_S

            chutes_task = asyncio.create_task(
                self._chutes_stream(m, messages, max_tokens_fn(m))
            ) if not self._chutes_off else None

            or_task = asyncio.create_task(
                self._openrouter(m, messages, max_tokens_fn(m),
                                 accept=accept, deadline=deadline)
            ) if self._fb_enabled else None

            # If only one provider available, await it directly
            if chutes_task is None and or_task is None:
                return m, None, False
            if chutes_task is None:
                raw = await or_task
                return m, raw, False
            if or_task is None:
                c = await chutes_task
                ok = c is not None and accept(c)
                return m, c if ok else None, ok

            # Race: check each task as it completes
            pending = {chutes_task, or_task}

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    raw = task.result()
                    if raw is not None and accept(raw):
                        # Winner found — cancel the other
                        for t in pending:
                            t.cancel()
                        return m, raw, (task is chutes_task)

            # Both failed
            return m, None, False

        triples = await asyncio.gather(*[_resolve_one(m) for m in models])

        for m, raw, hit in triples:
            provider = "chutes" if hit else ("openrouter" if raw is not None else "none")
            log.info("[judge] %s -> %s via %s", m, "ok" if raw is not None else "miss", provider)

        results = {m: raw for m, raw, _ in triples}
        chutes_hit = any(hit for _, _, hit in triples)

        # Circuit-breaker bookkeeping: count tasks where Chutes scored zero judges.
        if not self._chutes_off:
            if chutes_hit:
                self._chutes_dry_tasks = 0
            else:
                self._chutes_dry_tasks += 1
                if self._chutes_dry_tasks >= JUDGE_CHUTES_GIVEUP_TASKS:
                    self._chutes_off = True
                    log.warning(
                        "query_judges: Chutes dry for %d tasks — OpenRouter-only for rest of eval",
                        JUDGE_CHUTES_GIVEUP_TASKS,
                    )

        return {m: results.get(m) for m in models}

    # ----- back-compat wrappers -----

    async def pairwise_judge(self, context_messages=None, king_reply="", chal_reply="", *, model,
                             messages_prefix=None, messages_prompt=None, hotkey="", seed=b"") -> MetricVerdict:
        if context_messages is None:
            context_messages = list(messages_prefix or []) + list(messages_prompt or [])
        msgs = self._build_pairwise_messages(context_messages, king_reply, chal_reply)
        raws = await self.query_judges([model], msgs, accept=lambda r: parse_metric_verdict(r).parse_ok)
        raw = raws.get(model)
        if raw is None:
            return MetricVerdict(metric_scores={k: 0.0 for k in JUDGE_METRIC_KEYS},
                                 judge_mean=0.0, raw="", parse_ok=False, model=model)
        v = parse_metric_verdict(raw)
        v.model = model
        return v

    async def probe_batch(self, messages: list[dict], reply: str, models: list[str]) -> dict[str, bool | None]:
        """Injection probe for all judges in ONE query_judges (per-judge sequential: each judge
        Chutes -> OpenRouter). Returns {model: is_injected | None}; None = unresolved by both
        providers within budget -> the caller fails closed (marks the turn untested)."""
        detailed = await self.probe_batch_with_raw(messages, reply, models)
        out: dict[str, bool | None] = {}
        for m in models:
            entry = detailed.get(m) or {}
            out[m] = entry.get("injection")
        return out

    async def probe_batch_with_raw(
        self,
        messages: list[dict],
        reply: str,
        models: list[str],
    ) -> dict[str, dict[str, str | bool | None]]:
        """Return parsed and raw injection-probe verdicts for every judge model."""
        probe_msgs = self._build_probe_messages(messages, reply)
        raws = await self.query_judges(
            models, probe_msgs, accept=_injection_accept,
            max_tokens_fn=lambda m: JUDGE_THINKING_TOKENS if _is_thinking(m) else JUDGE_MAX_TOKENS,
        )
        out: dict[str, dict[str, str | bool | None]] = {}
        for m in models:
            raw = raws.get(m)
            injection = None if raw is None else _parse_injection(raw)
            out[m] = {
                "injection": injection,
                "raw": raw,
                "evidence": None if raw is None else _parse_injection_evidence(raw),
            }
        return out

    async def probe(self, messages: list[dict], reply: str, *, model: str) -> tuple[bool, str]:
        """Single-judge injection probe (back-compat). Raises RuntimeError on exhaustion."""
        res = await self.probe_batch(messages, reply, [model])
        if res.get(model) is None:
            raise RuntimeError(f"probe {model}: no definitive answer (both providers exhausted)")
        return bool(res[model]), ""

    async def aclose(self) -> None:
        await self._chutes.aclose()
        if self._fb is not None:
            await self._fb.aclose()

    async def __aenter__(self) -> "ChutesJudge":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
