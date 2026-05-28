"""Albedo eval server.

Runs on the GPU box. Manages two long-/short-lived vLLM subprocesses
(king, challenger), pulls samples from the local SWE-ZERO parquet corpus,
queries both contestants for the same `(messages_prefix, turn)`, scores
each reply via Chutes LLM-as-judge with the 3-tier rubric, and emits an
SSE stream of progress + a final per-judge dimensional verdict (ensemble
paired-bootstrap stats included for diagnostics).

Endpoints:
    GET  /health         — vLLM ready states + GPU mem + dataset state.
    POST /set_king       — point king vLLM at a new Hippius ref. Idempotent.
    POST /eval           — start a duel, SSE stream `progress`/`verdict`.

Process model:
    king_vllm        long-lived, GPUs configured by ALBEDO_KING_GPUS
    challenger_vllm  spun up per duel,  ALBEDO_CHAL_GPUS

Both are vanilla `vllm serve` subprocesses with the
VLLM_USE_FLASHINFER_SAMPLER=0 + VLLM_USE_DEEP_GEMM=0 envs we already
proved out on the H200 box (no nvcc + no vendored deep_gemm).
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import chain_config
import judge as judge_mod
import trajectory_sampler
from model_store import (
    MODEL_CACHE_DIR,
    ModelRef,
    disk_free_bytes,
    ensure_chat_template,
    materialize_model,
    prune_model_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("albedo.eval")


# ---------------------------------------------------------------------------
# Config (env)
# ---------------------------------------------------------------------------

KING_PORT      = int(os.environ.get("ALBEDO_KING_PORT", "8001"))
CHAL_PORT      = int(os.environ.get("ALBEDO_CHAL_PORT", "8002"))
KING_GPUS      = os.environ.get("ALBEDO_KING_GPUS", "0,1,2,3")
CHAL_GPUS      = os.environ.get("ALBEDO_CHAL_GPUS", "4,5,6,7")
GPU_MEM_UTIL   = float(os.environ.get("ALBEDO_GPU_MEMORY_UTILIZATION", "0.85"))
VLLM_DTYPE     = os.environ.get("ALBEDO_VLLM_DTYPE", "bfloat16")
VLLM_STARTUP_TIMEOUT_S = int(os.environ.get("ALBEDO_VLLM_STARTUP_TIMEOUT_S", "300"))
VLLM_MAX_MODEL_LEN = int(os.environ.get(
    "ALBEDO_VLLM_MAX_MODEL_LEN", str(chain_config.DUEL_GEN_MAX_MODEL_LEN)
))
# Extra headroom: local tokenizer counts often under-estimate vLLM's chat-template
# tokenization, causing 400s when prompt + max_tokens exceeds max_model_len.
VLLM_CONTEXT_SAFETY_MARGIN = int(
    os.environ.get("ALBEDO_VLLM_CONTEXT_SAFETY_MARGIN", "512")
)
# Reserve headroom for generation + vLLM overhead when truncating long trajectories.
VLLM_PROMPT_TOKEN_BUDGET = max(
    512,
    VLLM_MAX_MODEL_LEN
    - chain_config.DUEL_GEN_MAX_TOKENS
    - 64
    - VLLM_CONTEXT_SAFETY_MARGIN,
)
# Reject duels where too many turns fail vLLM generation (unfair comparison).
MIN_VALID_TURN_FRAC = float(os.environ.get("ALBEDO_MIN_VALID_TURN_FRAC", "0.8"))
# Headroom for one challenger snapshot (~3.5 GB) + vLLM temp files.
MIN_DISK_BYTES = int(os.environ.get("ALBEDO_MIN_DISK_BYTES", str(6 * 1024**3)))
# Overlay `/tmp` on the eval box is often full (teutonic data). Triton JIT and
# vLLM torch.compile write temp files there unless redirected to /root.
TMP_DIR = os.environ.get("ALBEDO_TMP_DIR", "/root/albedo/tmp")

DATASET_DIR = os.environ.get("ALBEDO_DATASET_DIR", "/var/albedo/dataset")

# Per-duel concurrency caps. Each "task" = (one model query + one judge call)
# for one (sample, turn). Two tasks per sample (king + challenger) run side
# by side; the gather is bounded so we don't open thousands of judge sockets.
MAX_PARALLEL_TURNS = int(os.environ.get("ALBEDO_MAX_PARALLEL_TURNS", "8"))
SSE_HEARTBEAT_S    = float(os.environ.get("ALBEDO_SSE_HEARTBEAT_S", "5.0"))

# Eval-trace sink. Every duel's full (messages_prefix, king_reply, chal_reply,
# judge_verdict, rationale, original_reply) records are gzipped + uploaded
# to Hippius S3 so the corpus is mineable for distillation training. The
# sink is best-effort: a failed upload does NOT fail the duel — the
# in-memory record is also kept on local disk under EVALS_LOCAL_DIR so an
# operator can re-upload manually.
EVALS_ENABLED     = os.environ.get("ALBEDO_EVALS_ENABLED", "1") not in ("", "0", "false", "False")
EVALS_S3_ENDPOINT = os.environ.get("ALBEDO_EVALS_S3_ENDPOINT", "https://s3.hippius.com")
EVALS_S3_BUCKET   = os.environ.get("ALBEDO_EVALS_S3_BUCKET", "")
EVALS_S3_ACCESS   = os.environ.get("ALBEDO_EVALS_S3_ACCESS_KEY", "")
EVALS_S3_SECRET   = os.environ.get("ALBEDO_EVALS_S3_SECRET_KEY", "")
EVALS_S3_PREFIX   = os.environ.get("ALBEDO_EVALS_S3_PREFIX", "evals").strip("/")
EVALS_LOCAL_DIR   = os.environ.get("ALBEDO_EVALS_LOCAL_DIR", "/var/albedo/evals")
EVALS_PUBLIC_BASE = os.environ.get(
    "ALBEDO_EVALS_PUBLIC_BASE",
    # us-east-1.hippius.com is path-style + public-read for this bucket;
    # override if you use a different region or a private bucket with a CDN.
    "https://us-east-1.hippius.com",
).rstrip("/")
EVALS_JUDGE_RAW_MAX_CHARS = int(os.environ.get("ALBEDO_EVALS_JUDGE_RAW_MAX_CHARS", "8192"))

# Bump when turn / duel_meta fields change so training exporters can branch.
EVAL_TRACE_SCHEMA_VERSION = 1


def _all_keep_refs(req: "EvalRequest") -> list[ModelRef]:
    """Full model cache keep-set: current king + past kings + 5 recent
    evaluated challengers + any still-queued challengers."""
    refs: list[ModelRef] = []
    for entry in [req.king] + req.king_chain + req.recent_challengers + req.queued_challengers:
        if not entry:
            continue
        repo   = entry.get("repo") or entry.get("model_repo", "")
        digest = entry.get("digest") or entry.get("king_digest", "")
        if repo and digest:
            try:
                refs.append(ModelRef(repo, digest))
            except Exception:
                pass
    return refs


def _ensure_disk_for_duel(
    king_ref: ModelRef,
    chal_ref: ModelRef,
    keep_refs: list[ModelRef] | None = None,
) -> None:
    """Prune stale caches then verify free space before downloading weights."""
    from model_store import ensure_disk_bytes

    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    for path in (MODEL_CACHE_DIR, TMP_DIR):
        if disk_free_bytes(path) >= MIN_DISK_BYTES:
            continue
        if path == MODEL_CACHE_DIR:
            keep = list(keep_refs) if keep_refs else [king_ref]
            freed = prune_model_cache(*keep)
            log.warning(
                "low disk before duel (need %d bytes); pruned %.2f GB keeping %d models",
                MIN_DISK_BYTES,
                freed / 1e9,
                len(keep),
            )
        ensure_disk_bytes(MIN_DISK_BYTES, path)


async def _post_duel_cache_cleanup(
    king_ref: ModelRef,
    keep_refs: list[ModelRef] | None = None,
) -> None:
    """Drop stale models after each duel; kings + recent challengers + queue stay cached."""
    try:
        keep = list(keep_refs) if keep_refs else [king_ref]
        freed = await asyncio.to_thread(prune_model_cache, *keep)
        if freed:
            log.info(
                "post-duel cache prune freed %.2f GB (kept %d models)",
                freed / 1e9,
                len(keep),
            )
    except Exception:
        log.exception("post-duel cache prune failed (non-fatal)")


# ---------------------------------------------------------------------------
# Eval-trace sink (publish per-turn judge data to Hippius for distillation)
# ---------------------------------------------------------------------------

@dataclass
class DatasetSink:
    """Writes one `.jsonl.gz` per duel to Hippius S3 + a local backup.

    File shape:
        line 1   : {"type": "duel_meta", ...}
        line 2..N: {"type": "turn", ...}                   one per (sample, turn)
        last line: {"type": "verdict", ...}                duel-level outcome
    Every record is independently parseable so a partial file (eval.py
    crashed mid-duel) is still usable training data.
    """
    eval_id: str
    enabled: bool = EVALS_ENABLED
    s3_bucket: str = EVALS_S3_BUCKET
    s3_prefix: str = EVALS_S3_PREFIX
    public_base: str = EVALS_PUBLIC_BASE
    _local_path: Path | None = None
    _records: list[dict] = field(default_factory=list)
    _client: object | None = None  # boto3 client; lazy-imported

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        # Stamp the day in the key so the prefix is easy to browse and
        # cheap to enumerate (S3 list-objects pagination by prefix).
        self._day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        local_dir = Path(EVALS_LOCAL_DIR) / self._day
        local_dir.mkdir(parents=True, exist_ok=True)
        self._local_path = local_dir / f"{self.eval_id}.jsonl.gz"

    @property
    def s3_key(self) -> str:
        return f"{self.s3_prefix}/{self._day}/{self.eval_id}.jsonl.gz"

    @property
    def public_url(self) -> str | None:
        if not (self.enabled and self.s3_bucket):
            return None
        return f"{self.public_base}/{self.s3_bucket}/{self.s3_key}"

    def append(self, record: dict) -> None:
        if not self.enabled:
            return
        self._records.append(record)

    async def flush(self) -> dict:
        """Compress in-memory records and (best-effort) upload to S3.
        Always writes the local file even if S3 is misconfigured."""
        if not self.enabled or not self._records:
            return {"enabled": self.enabled, "n_records": len(self._records),
                    "uploaded": False, "local_path": None, "url": None}

        body = io.BytesIO()
        with gzip.GzipFile(fileobj=body, mode="wb") as gz:
            for rec in self._records:
                gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
        data = body.getvalue()

        if self._local_path:
            try:
                self._local_path.write_bytes(data)
            except Exception:
                log.exception("eval sink local write failed (non-fatal)")

        uploaded = False
        url = None
        if self.s3_bucket and EVALS_S3_ACCESS and EVALS_S3_SECRET:
            try:
                client = await asyncio.to_thread(self._boto_client)
                await asyncio.to_thread(
                    client.put_object,
                    Bucket=self.s3_bucket,
                    Key=self.s3_key,
                    Body=data,
                    ContentType="application/gzip",
                    ContentEncoding="gzip",
                    CacheControl="public, max-age=31536000, immutable",
                )
                uploaded = True
                url = self.public_url
                log.info("eval traces uploaded: %s (%d records, %d bytes)",
                         url, len(self._records), len(data))
            except Exception:
                log.exception("eval sink S3 upload failed; record kept locally at %s",
                              self._local_path)
        else:
            log.info("eval sink S3 creds missing; wrote local-only %s "
                     "(%d records, %d bytes)",
                     self._local_path, len(self._records), len(data))
        try:
            await self._append_manifest(self._summarize_for_manifest())
        except Exception:
            log.exception("manifest append failed (non-fatal)")
        return {
            "enabled": True,
            "n_records": len(self._records),
            "uploaded": uploaded,
            "local_path": str(self._local_path) if self._local_path else None,
            "url": url,
            "key": self.s3_key,
            "bytes": len(data),
        }

    def _summarize_for_manifest(self) -> dict:
        turns = [r for r in self._records if r.get("type") == "turn"]
        verdict = next((r for r in self._records if r.get("type") == "verdict"), None)
        meta = next((r for r in self._records if r.get("type") == "duel_meta"), None)
        n_valid = sum(
            1 for t in turns
            if t.get("king", {}).get("reply") and t.get("chal", {}).get("reply")
        )
        parse_by_judge: dict[str, int] = {}
        for t in turns:
            if t.get("error"):
                continue
            for j in t.get("judges", []):
                if not j.get("parse_ok"):
                    parse_by_judge[j.get("model", "?")] = (
                        parse_by_judge.get(j.get("model", "?"), 0) + 1
                    )
        return {
            "type": "manifest_entry",
            "schema_version": EVAL_TRACE_SCHEMA_VERSION,
            "eval_id": self.eval_id,
            "day": self._day,
            "url": self.public_url,
            "key": self.s3_key,
            "hotkey": (meta or {}).get("hotkey"),
            "challenger": (meta or {}).get("challenger"),
            "king": (meta or {}).get("king"),
            "n_turns": len(turns),
            "n_valid_turns": n_valid,
            "n_vllm_error": sum(1 for t in turns if t.get("error")),
            "n_truncated": sum(1 for t in turns if t.get("prompt_truncated")),
            "parse_failures_by_judge": parse_by_judge,
            "accepted": (verdict or {}).get("accepted"),
            "completed_at": (verdict or {}).get("completed_at"),
        }

    async def _append_manifest(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        manifest_name = "manifest.jsonl"
        local_manifest = Path(EVALS_LOCAL_DIR) / self._day / manifest_name
        try:
            local_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(local_manifest, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            log.exception("manifest local append failed (non-fatal)")

        if not (self.s3_bucket and EVALS_S3_ACCESS and EVALS_S3_SECRET):
            return
        s3_key = f"{self.s3_prefix}/{self._day}/{manifest_name}"
        try:
            client = await asyncio.to_thread(self._boto_client)
            existing = b""
            try:
                obj = await asyncio.to_thread(
                    client.get_object, Bucket=self.s3_bucket, Key=s3_key,
                )
                existing = await asyncio.to_thread(obj["Body"].read)
            except Exception:
                pass
            body = existing + line.encode()
            await asyncio.to_thread(
                client.put_object,
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=body,
                ContentType="application/x-ndjson",
                CacheControl="public, max-age=300",
            )
        except Exception:
            log.exception("manifest S3 append failed (non-fatal)")

    def _boto_client(self):
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config as BotoConfig
        self._client = boto3.client(
            "s3", endpoint_url=EVALS_S3_ENDPOINT,
            aws_access_key_id=EVALS_S3_ACCESS,
            aws_secret_access_key=EVALS_S3_SECRET,
            region_name="decentralized",
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        return self._client


# ---------------------------------------------------------------------------
# vLLM subprocess manager
# ---------------------------------------------------------------------------

@dataclass
class VLLMProcess:
    """One vLLM-serve subprocess pinned to a fixed GPU set + port.

    The same envs that worked in the live smoke (CUDA_VISIBLE_DEVICES +
    disabling flashinfer/deep_gemm). `tensor_parallel_size` defaults to the
    number of GPUs in the pin set so a 4B model can split when GPU memory
    is tight; for mini-coder-1.7b a single GPU is enough but we keep the
    split optional.
    """
    role: str                 # "king" | "challenger"
    port: int
    gpus: str
    model_path: str = ""
    model_name: str = ""      # ModelRef.immutable_ref, surfaced in /health
    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    base_url: str = ""

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def _wait_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                if not self.is_alive():
                    raise RuntimeError(
                        f"{self.role} vllm exited during startup (rc={self.proc.returncode})"
                    )
                try:
                    r = await client.get(f"{self.base_url}/v1/models")
                    if r.status_code == 200:
                        log.info("%s vllm ready at %s after %.1fs",
                                 self.role, self.base_url, time.monotonic() - self.started_at)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
        raise asyncio.TimeoutError(
            f"{self.role} vllm did not come up within {timeout_s}s"
        )

    async def start(self, model_path: str, model_name: str) -> None:
        if self.is_alive():
            await self.stop()
        self.model_path = model_path
        self.model_name = model_name
        self.base_url = f"http://127.0.0.1:{self.port}"

        await asyncio.to_thread(ensure_chat_template, model_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["VLLM_USE_DEEP_GEMM"] = "0"
        env["VLLM_MOE_USE_DEEP_GEMM"] = "0"
        Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = TMP_DIR
        env["TRITON_CACHE_DIR"] = os.path.join(TMP_DIR, "triton_cache")
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(TMP_DIR, "torchinductor"))

        n_gpus = max(1, len([g for g in self.gpus.split(",") if g.strip()]))
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--max-model-len", str(VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEM_UTIL),
            "--dtype", VLLM_DTYPE,
            "--tensor-parallel-size", str(n_gpus),
            "--served-model-name", model_name,
            "--no-enable-log-requests",
        ]
        log.info("starting %s vllm: %s", self.role, " ".join(cmd))
        log_dir = Path("/var/albedo/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"vllm_{self.role}.log"
        self._log_file = open(self._log_path, "ab")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started_at = time.monotonic()
        await self._wait_ready(VLLM_STARTUP_TIMEOUT_S)

    async def stop(self) -> None:
        if self.proc is None:
            return
        if self.is_alive():
            log.info("stopping %s vllm (pid=%d)", self.role, self.proc.pid)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.proc.wait), timeout=30.0
                )
            except asyncio.TimeoutError:
                log.warning("%s vllm did not exit on SIGTERM, sending SIGKILL", self.role)
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if hasattr(self, "_log_file"):
            with contextlib.suppress(Exception):
                self._log_file.close()
        self.proc = None
        self.base_url = ""
        self.model_path = ""
        self.model_name = ""

    def health(self) -> dict:
        return {
            "role": self.role,
            "port": self.port,
            "gpus": self.gpus,
            "alive": self.is_alive(),
            "model_name": self.model_name,
            "pid": self.proc.pid if self.proc else None,
            "uptime_s": (time.monotonic() - self.started_at) if self.is_alive() else 0,
        }


# ---------------------------------------------------------------------------
# Eval state (singleton)
# ---------------------------------------------------------------------------

@dataclass
class EvalState:
    king_proc: VLLMProcess = field(default_factory=lambda: VLLMProcess("king", KING_PORT, KING_GPUS))
    chal_proc: VLLMProcess = field(default_factory=lambda: VLLMProcess("challenger", CHAL_PORT, CHAL_GPUS))
    eval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    current_eval_id: str | None = None


STATE = EvalState()


# ---------------------------------------------------------------------------
# Contestant query (vLLM OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------

_TOKENIZER_BY_PATH: dict[str, object] = {}


def _chat_prompt_tokens(tokenizer, messages: list[dict]) -> int:
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    return len(ids)


def _fit_messages_for_vllm(
    messages: list[dict], model_path: str,
) -> tuple[list[dict], dict]:
    """Drop or trim prefix turns so the prompt fits `VLLM_PROMPT_TOKEN_BUDGET`.

    Returns (fitted_messages, truncation_info).
    """
    from transformers import AutoTokenizer

    original_n = len(messages)
    trimmed_chars = 0

    tok = _TOKENIZER_BY_PATH.get(model_path)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _TOKENIZER_BY_PATH[model_path] = tok

    msgs = [dict(m) for m in messages]
    if _chat_prompt_tokens(tok, msgs) <= VLLM_PROMPT_TOKEN_BUDGET:
        return msgs, {
            "truncated": False,
            "original_n_messages": original_n,
            "fitted_n_messages": len(msgs),
            "last_message_trimmed_chars": 0,
            "prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
        }

    while len(msgs) > 1 and _chat_prompt_tokens(tok, msgs) > VLLM_PROMPT_TOKEN_BUDGET:
        drop = next((i for i, m in enumerate(msgs) if m.get("role") != "system"), 0)
        if drop >= len(msgs) - 1:
            break
        msgs.pop(drop)

    if _chat_prompt_tokens(tok, msgs) <= VLLM_PROMPT_TOKEN_BUDGET:
        log.warning(
            "truncated trajectory prefix to %d messages for vLLM budget=%d",
            len(msgs), VLLM_PROMPT_TOKEN_BUDGET,
        )
        return msgs, {
            "truncated": len(msgs) != original_n,
            "original_n_messages": original_n,
            "fitted_n_messages": len(msgs),
            "last_message_trimmed_chars": 0,
            "prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
        }

    last = msgs[-1]
    content = last.get("content") or ""
    if not isinstance(content, str) or not content:
        return msgs, {
            "truncated": len(msgs) != original_n,
            "original_n_messages": original_n,
            "fitted_n_messages": len(msgs),
            "last_message_trimmed_chars": 0,
            "prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
        }

    lo, hi = 0, len(content)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = msgs[:-1] + [{**last, "content": content[-mid:]}]
        if _chat_prompt_tokens(tok, trial) <= VLLM_PROMPT_TOKEN_BUDGET:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best <= 0:
        # Last-resort: drop non-system prefix turns until the prompt fits.
        while len(msgs) > 2 and _chat_prompt_tokens(tok, msgs) > VLLM_PROMPT_TOKEN_BUDGET:
            drop = next((i for i, m in enumerate(msgs) if m.get("role") != "system"), 0)
            if drop >= len(msgs) - 1:
                break
            msgs.pop(drop)
        return msgs, {
            "truncated": len(msgs) != original_n,
            "original_n_messages": original_n,
            "fitted_n_messages": len(msgs),
            "last_message_trimmed_chars": 0,
            "prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
        }
    trimmed_chars = len(content) - best
    msgs[-1] = {**last, "content": content[-best:]}
    log.warning(
        "trimmed last message to %d chars for vLLM budget=%d",
        best, VLLM_PROMPT_TOKEN_BUDGET,
    )
    return msgs, {
        "truncated": True,
        "original_n_messages": original_n,
        "fitted_n_messages": len(msgs),
        "last_message_trimmed_chars": trimmed_chars,
        "prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
    }


def _clip_judge_raw(raw: str) -> tuple[str, bool]:
    if len(raw) <= EVALS_JUDGE_RAW_MAX_CHARS:
        return raw, False
    return raw[:EVALS_JUDGE_RAW_MAX_CHARS], True


async def query_contestant(
    client: httpx.AsyncClient,
    proc: VLLMProcess,
    messages: list[dict],
    *,
    fitted_messages: list[dict] | None = None,
) -> tuple[str, dict]:
    fitted = fitted_messages
    if fitted is None:
        fitted, _ = await asyncio.to_thread(
            _fit_messages_for_vllm, messages, proc.model_path,
        )
    body = {
        "model": proc.model_name,
        "messages": fitted,
        "temperature": chain_config.DUEL_GEN_TEMPERATURE,
        "max_tokens": chain_config.DUEL_GEN_MAX_TOKENS,
    }
    r = await client.post(f"{proc.base_url}/v1/chat/completions", json=body, timeout=300.0)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    return content, usage


# ---------------------------------------------------------------------------
# Paired bootstrap on per-turn deltas
# ---------------------------------------------------------------------------

def paired_bootstrap_lcb(
    deltas: list[float],
    *,
    resamples: int,
    alpha: float,
    rng_seed: bytes,
) -> tuple[float, float, float]:
    """Return (mean_delta, lcb_at_1-alpha, se).

    Standard one-sided bootstrap: resample the per-turn deltas with
    replacement, take the lower `alpha` percentile of resample means.
    """
    if not deltas:
        return 0.0, 0.0, 0.0
    arr = np.asarray(deltas, dtype=np.float64)
    mean = float(arr.mean())
    # Use the bootstrap RNG independently from the sampler RNG so resample
    # noise can't leak into fixture selection.
    import hashlib as _h
    digest = _h.blake2b(rng_seed + b"|bootstrap", digest_size=32).digest()
    entropy = np.frombuffer(digest, dtype=np.uint64).tolist()
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy=entropy)))
    n = len(arr)
    means = arr[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    lcb = float(np.quantile(means, alpha))
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, lcb, se


def judge_dimension_outcome(mean_delta: float, *, tie_band: float) -> str:
    """Per-judge win/tie/lose from mean challenger-minus-king score."""
    if mean_delta > tie_band:
        return "win"
    if mean_delta < -tie_band:
        return "lose"
    return "tie"


def dethrone_by_judge_dimensions(
    judge_outcomes: list[str],
    *,
    min_turns: int,
    n_done: int,
    n_valid: int,
) -> tuple[bool, dict]:
    """Match-and-exceed across judge dimensions (scores per judge).

    Crown the challenger when:
      - at least one judge dimension is a strict win (beat the king), and
      - every other judge dimension is a tie or win (no dimension where the
        king clearly beats the challenger).
    """
    wins = sum(1 for o in judge_outcomes if o == "win")
    ties = sum(1 for o in judge_outcomes if o == "tie")
    loses = sum(1 for o in judge_outcomes if o == "lose")
    min_valid = max(min_turns, int(n_done * MIN_VALID_TURN_FRAC))
    accepted = (
        n_valid >= min_turns
        and n_valid >= min_valid
        and wins >= 1
        and loses == 0
    )
    return accepted, {
        "rule": "match_exceed_one_dimension",
        "n_wins": wins,
        "n_ties": ties,
        "n_loses": loses,
        "min_turns": min_turns,
        "n_valid": n_valid,
        "n_done": n_done,
        "min_valid_turns": min_valid,
    }


# ---------------------------------------------------------------------------
# Duel runner
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    king:                 dict           # {"repo": str, "digest": str}
    challenger:           dict
    seed_hex:             str            # hex-encoded seed bytes (e.g. blake2b(blockhash||hotkey))
    eval_id:              str
    hotkey:               str | None = None
    n_samples:            int | None = None
    max_turns:            int | None = None
    king_chain:           list[dict] = []   # past kings [{"repo":…, "digest":…}]
    recent_challengers:   list[dict] = []   # last 5 evaluated [{"repo":…, "digest":…}]
    queued_challengers:   list[dict] = []   # still in queue (not yet evaluated)


class SetKingRequest(BaseModel):
    king: dict                        # {"repo": str, "digest": str}


def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def _heartbeat_pump(out: asyncio.Queue, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=SSE_HEARTBEAT_S)
        except asyncio.TimeoutError:
            await out.put(_sse_event("heartbeat", {"ts": time.time()}))


async def _score_one_turn(
    sample: trajectory_sampler.Sample,
    vllm_client: httpx.AsyncClient,
    judge_client: judge_mod.ChutesJudge,
    sem: asyncio.Semaphore,
    judge_models: tuple[str, ...],
    *,
    hotkey: str | None = None,
    challenger: dict | None = None,
) -> dict:
    """Run king + challenger generation once, then fan out across every
    judge in `judge_models` (two judge calls per judge — king side and
    challenger side, all in flight at once). Returns a record with one
    `per_judge` entry per judge plus ensemble aggregates."""
    async with sem:
        fitted, trunc_info = await asyncio.to_thread(
            _fit_messages_for_vllm,
            sample.messages_prefix,
            STATE.king_proc.model_path,
        )
        judge_context = fitted

        # 1. Generation in parallel (same fitted prompt for both models).
        king_task = asyncio.create_task(
            query_contestant(
                vllm_client, STATE.king_proc, sample.messages_prefix,
                fitted_messages=fitted,
            )
        )
        chal_task = asyncio.create_task(
            query_contestant(
                vllm_client, STATE.chal_proc, sample.messages_prefix,
                fitted_messages=fitted,
            )
        )
        try:
            (king_reply, king_usage), (chal_reply, chal_usage) = await asyncio.gather(
                king_task, chal_task
            )
        except Exception as exc:
            log.warning("vllm error on sample %d turn %d: %s",
                        sample.sample_idx, sample.turn_idx, exc)
            # On vLLM failure: emit one reject record per judge so downstream
            # accumulators stay uniform across all judges.
            per_judge_fail = [
                {"model": jm, "king_verdict": "reject", "chal_verdict": "reject",
                 "king_score": 0.0, "chal_score": 0.0,
                 "king_rationale": "vllm_error", "chal_rationale": "vllm_error",
                 "parse_ok": True, "vllm_error": True}
                for jm in judge_models
            ]
            return {
                "global_idx": sample.global_idx,
                "shard_idx": sample.shard_idx,
                "shard_name": sample.shard_name,
                "sample_idx": sample.sample_idx,
                "turn_idx": sample.turn_idx,
                "instance_id": sample.instance_id,
                "repo": sample.repo,
                "hotkey": hotkey,
                "challenger": challenger,
                "messages_prefix": sample.messages_prefix,
                "messages_prompt": fitted,
                "prompt_truncated": trunc_info.get("truncated", False),
                "prompt_truncation": trunc_info,
                "original_reply": sample.original_reply,
                "king_reply": "",
                "chal_reply": "",
                "per_judge": per_judge_fail,
                "king_score_avg": 0.0,
                "chal_score_avg": 0.0,
                "delta_avg": 0.0,
                "parse_ok": False,
                "error": f"vllm_error: {exc}",
                "king_usage": {},
                "chal_usage": {},
            }

        # 2. Fan out across all judges. Two judge calls per judge per turn,
        #    all in flight at once. Judges see the same prompt vLLM used.
        tasks: list[asyncio.Task] = []
        for jm in judge_models:
            tasks.append(asyncio.create_task(
                judge_client.score(judge_context, king_reply, model=jm)
            ))
            tasks.append(asyncio.create_task(
                judge_client.score(judge_context, chal_reply, model=jm)
            ))
        verdicts = await asyncio.gather(*tasks)

        per_judge: list[dict] = []
        king_sum = 0.0
        chal_sum = 0.0
        any_parse_fail = False
        for i, jm in enumerate(judge_models):
            k_v = verdicts[2 * i]
            c_v = verdicts[2 * i + 1]
            king_raw, king_raw_trunc = _clip_judge_raw(k_v.raw)
            chal_raw, chal_raw_trunc = _clip_judge_raw(c_v.raw)
            per_judge.append({
                "model": jm,
                "king_verdict": k_v.label,
                "chal_verdict": c_v.label,
                "king_score": k_v.score,
                "chal_score": c_v.score,
                "king_rationale": k_v.rationale,
                "chal_rationale": c_v.rationale,
                "king_raw": king_raw,
                "chal_raw": chal_raw,
                "king_raw_truncated": king_raw_trunc,
                "chal_raw_truncated": chal_raw_trunc,
                "parse_ok": k_v.parse_ok and c_v.parse_ok,
            })
            king_sum += k_v.score
            chal_sum += c_v.score
            if not (k_v.parse_ok and c_v.parse_ok):
                any_parse_fail = True

        n = max(1, len(judge_models))
        king_avg = king_sum / n
        chal_avg = chal_sum / n
        return {
            "global_idx": sample.global_idx,
            "shard_idx": sample.shard_idx,
            "shard_name": sample.shard_name,
            "sample_idx": sample.sample_idx,
            "turn_idx": sample.turn_idx,
            "instance_id": sample.instance_id,
            "repo": sample.repo,
            "hotkey": hotkey,
            "challenger": challenger,
            "messages_prefix": sample.messages_prefix,
            "messages_prompt": fitted,
            "prompt_truncated": trunc_info.get("truncated", False),
            "prompt_truncation": trunc_info,
            "original_reply": sample.original_reply,
            "king_reply": king_reply,
            "chal_reply": chal_reply,
            "per_judge": per_judge,
            "king_score_avg": king_avg,
            "chal_score_avg": chal_avg,
            "delta_avg": chal_avg - king_avg,
            "parse_ok": not any_parse_fail,
            "king_usage": king_usage,
            "chal_usage": chal_usage,
        }


async def _safe_flush_sink(sink: "DatasetSink", flushed_ref: list[bool]) -> dict:
    if flushed_ref[0]:
        return {"enabled": sink.enabled, "uploaded": False, "url": None,
                "note": "already_flushed"}
    flushed_ref[0] = True
    try:
        return await asyncio.wait_for(sink.flush(), timeout=60.0)
    except Exception:
        log.exception("sink flush failed in cleanup (non-fatal)")
        return {"enabled": sink.enabled, "uploaded": False, "url": None,
                "error": "flush_failed"}


async def run_duel(req: EvalRequest) -> AsyncIterator[bytes]:
    """The full duel. Yields SSE-framed bytes for StreamingResponse.

    Wrapped in try/finally so the dataset sink is flushed on any exit
    path — normal completion, mid-stream exception, or client disconnect
    (StreamingResponse closes the async generator, the finally runs).
    """
    eval_id = req.eval_id
    seed = bytes.fromhex(req.seed_hex)
    n_samples = req.n_samples or chain_config.DUEL_N_SAMPLES
    max_turns = req.max_turns or chain_config.DUEL_MAX_TURNS_PER_SAMPLE

    sink = DatasetSink(eval_id=eval_id)
    flushed_ref = [False]   # mutable cell so the finally block can see it
    sink.append({
        "type": "duel_meta",
        "schema_version": EVAL_TRACE_SCHEMA_VERSION,
        "eval_id": eval_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hotkey": req.hotkey,
        "king": req.king,
        "challenger": req.challenger,
        "seed_hex": req.seed_hex,
        "n_samples": n_samples,
        "max_turns_per_sample": max_turns,
        "judge_models": list(chain_config.JUDGE_MODELS),
        "judge_model": chain_config.JUDGE_MODEL,  # primary, kept for back-compat
        "judge_temperature": chain_config.JUDGE_TEMPERATURE,
        "judge_tie_band": chain_config.JUDGE_TIE_BAND,
        "judge_thinking_max_tokens": chain_config.JUDGE_THINKING_MAX_TOKENS,
        "gen_temperature": chain_config.DUEL_GEN_TEMPERATURE,
        "gen_max_tokens": chain_config.DUEL_GEN_MAX_TOKENS,
        "vllm_prompt_token_budget": VLLM_PROMPT_TOKEN_BUDGET,
        "dataset_repo": chain_config.DATASET_REPO,
        "dataset_shard_glob": chain_config.DATASET_SHARD_GLOB,
        "dataset_manifest_sha256": chain_config.DATASET_MANIFEST_SHA256,
        "chain_name": chain_config.NAME,
        "verdict_scale": judge_mod.VERDICT_SCORES,
    })

    try:
        async for chunk in _run_duel_inner(req, sink, flushed_ref,
                                            seed, n_samples, max_turns):
            yield chunk
    finally:
        # If the inner generator was interrupted (validator disconnect,
        # cancellation, exception) before reaching the verdict-yield path
        # the sink may not have been flushed yet. Make sure we always at
        # least try once.
        if not flushed_ref[0]:
            await _safe_flush_sink(sink, flushed_ref)
        try:
            king_ref = ModelRef(req.king["repo"], req.king["digest"])
            await _post_duel_cache_cleanup(king_ref, _all_keep_refs(req))
        except Exception:
            log.exception("post-duel cache cleanup failed (non-fatal)")


async def _run_duel_inner(req: EvalRequest, sink: "DatasetSink",
                           flushed_ref: list[bool], seed: bytes,
                           n_samples: int, max_turns: int) -> AsyncIterator[bytes]:
    eval_id = req.eval_id
    king_ref = ModelRef(req.king["repo"], req.king["digest"])
    yield _sse_event("phase", {"eval_id": eval_id, "phase": "materialize_challenger"})

    # 1. Materialize challenger (idempotent if already cached).
    try:
        chal_ref = ModelRef(req.challenger["repo"], req.challenger["digest"])
    except ValueError as exc:
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": f"invalid_challenger_ref: {exc}"})
        return
    try:
        await asyncio.to_thread(_ensure_disk_for_duel, king_ref, chal_ref, _all_keep_refs(req))
    except OSError as exc:
        log.error("disk check failed before materialize: %s", exc)
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": f"disk_full: {exc}"})
        return
    try:
        chal_dir = await asyncio.to_thread(
            materialize_model, chal_ref, None, 16
        )
    except Exception as exc:
        log.exception("challenger materialize failed")
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": f"materialize_failed: {exc}"})
        return

    yield _sse_event("phase", {"eval_id": eval_id, "phase": "start_challenger_vllm",
                                 "challenger": chal_ref.immutable_ref})

    # 2. Start challenger vLLM. King is assumed already running (managed by
    #    /set_king). If king happens to be down, fail loudly: this would
    #    indicate the validator misordered its calls.
    if not STATE.king_proc.is_alive():
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": "king_vllm_not_running"})
        return

    try:
        await STATE.chal_proc.start(chal_dir, chal_ref.immutable_ref)
    except Exception as exc:
        log.exception("challenger vllm failed to start")
        try:
            keep = _all_keep_refs(req)
            await asyncio.to_thread(prune_model_cache, *keep)
        except Exception:
            log.exception("cache prune after chal vllm fail (non-fatal)")
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": f"chal_vllm_start_failed: {exc}"})
        return

    # 3. Sample fixtures.
    yield _sse_event("phase", {"eval_id": eval_id, "phase": "sample_fixtures"})
    try:
        samples = trajectory_sampler.sample(
            seed,
            n_samples=n_samples,
            max_turns_per_sample=max_turns,
            dataset_dir=DATASET_DIR,
        )
    except Exception as exc:
        log.exception("sampling failed")
        await STATE.chal_proc.stop()
        yield _sse_event("verdict", {"eval_id": eval_id, "accepted": False,
                                     "error": f"sample_failed: {exc}"})
        return

    yield _sse_event("phase", {"eval_id": eval_id, "phase": "duel",
                                 "n_turns_total": len(samples)})

    # 4. Run all turns under bounded concurrency.
    judge_models = chain_config.JUDGE_MODELS
    sem = asyncio.Semaphore(MAX_PARALLEL_TURNS)
    out_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    hb_task = asyncio.create_task(_heartbeat_pump(out_queue, stop_event))

    # Ensemble (averaged-across-judges) accumulators.
    king_avg_sum = 0.0
    chal_avg_sum = 0.0
    n_done = 0
    n_valid = 0
    parse_failures = 0
    vllm_errors = 0
    per_turn_ensemble_deltas: list[float] = []  # paired-bootstrap input

    # Per-judge accumulators. One `Counter` per side per judge so the
    # dashboard can show one bar per judge with its own accept/weak/reject
    # breakdown — same shape as affine.io's per-environment bars.
    per_judge_acc: dict[str, dict] = {
        jm: {
            "n":              0,
            "king_sum":       0.0,
            "chal_sum":       0.0,
            "verdicts_king":  Counter(),
            "verdicts_chal":  Counter(),
            "deltas":         [],   # per-judge per-turn deltas
            "parse_failures": 0,
        }
        for jm in judge_models
    }
    per_turn_records: list[dict] = []

    def _judges_summary() -> list[dict]:
        out = []
        for jm in judge_models:
            acc = per_judge_acc[jm]
            n = max(acc["n"], 1)
            out.append({
                "model":           jm,
                "n":               acc["n"],
                "king_mean":       acc["king_sum"] / n,
                "chal_mean":       acc["chal_sum"] / n,
                "delta":           (acc["chal_sum"] - acc["king_sum"]) / n,
                "verdicts_king":   dict(acc["verdicts_king"]),
                "verdicts_chal":   dict(acc["verdicts_chal"]),
                "parse_failures":  acc["parse_failures"],
            })
        return out

    try:
        async with httpx.AsyncClient(timeout=300.0) as vllm_client, \
                judge_mod.ChutesJudge() as judge_client:

            async def runner(sample: trajectory_sampler.Sample) -> None:
                nonlocal king_avg_sum, chal_avg_sum, n_done, n_valid
                nonlocal parse_failures, vllm_errors
                rec = await _score_one_turn(
                    sample, vllm_client, judge_client, sem, judge_models,
                    hotkey=req.hotkey,
                    challenger=req.challenger,
                )
                n_done += 1
                is_vllm_error = bool(rec.get("error"))
                if is_vllm_error:
                    vllm_errors += 1
                else:
                    n_valid += 1
                    # Ensemble — only scored turns count toward means/deltas.
                    king_avg_sum += rec["king_score_avg"]
                    chal_avg_sum += rec["chal_score_avg"]
                    if not rec.get("parse_ok", True):
                        parse_failures += 1
                    per_turn_ensemble_deltas.append(rec["delta_avg"])

                    # Per-judge accumulation.
                    for pj in rec["per_judge"]:
                        acc = per_judge_acc[pj["model"]]
                        acc["n"] += 1
                        acc["king_sum"] += pj["king_score"]
                        acc["chal_sum"] += pj["chal_score"]
                        acc["verdicts_king"][pj["king_verdict"]] += 1
                        acc["verdicts_chal"][pj["chal_verdict"]] += 1
                        acc["deltas"].append(pj["chal_score"] - pj["king_score"])
                        if not pj["parse_ok"]:
                            acc["parse_failures"] += 1

                per_turn_records.append({
                    "sample_idx":    rec["sample_idx"],
                    "turn_idx":      rec["turn_idx"],
                    "instance_id":   rec["instance_id"],
                    "king_score":    rec["king_score_avg"],
                    "chal_score":    rec["chal_score_avg"],
                    "delta":         rec["delta_avg"],
                    "parse_ok":      rec.get("parse_ok", True),
                    "per_judge":     [
                        {
                            "model":         pj["model"],
                            "king_verdict":  pj["king_verdict"],
                            "chal_verdict":  pj["chal_verdict"],
                            "king_score":    pj["king_score"],
                            "chal_score":    pj["chal_score"],
                        }
                        for pj in rec["per_judge"]
                    ],
                    "error":         rec.get("error"),
                })

                # Persist the FULL turn (prompt + both replies + every
                # judge's verdict + rationale) for downstream distillation.
                sink.append({
                    "type":             "turn",
                    "schema_version":   EVAL_TRACE_SCHEMA_VERSION,
                    "eval_id":          eval_id,
                    "hotkey":           rec.get("hotkey"),
                    "challenger":       rec.get("challenger"),
                    "sample_idx":       rec["sample_idx"],
                    "turn_idx":         rec["turn_idx"],
                    "instance_id":      rec["instance_id"],
                    "repo":             rec.get("repo", ""),
                    "messages_prefix":  rec.get("messages_prefix", []),
                    "messages_prompt":  rec.get("messages_prompt", []),
                    "prompt_truncated": rec.get("prompt_truncated", False),
                    "prompt_truncation": rec.get("prompt_truncation", {}),
                    "original_reply":   rec.get("original_reply", ""),
                    "king": {
                        "reply":  rec.get("king_reply", ""),
                        "usage":  rec.get("king_usage", {}),
                    },
                    "chal": {
                        "reply":  rec.get("chal_reply", ""),
                        "usage":  rec.get("chal_usage", {}),
                    },
                    "judges":         rec["per_judge"],
                    "king_score_avg": rec["king_score_avg"],
                    "chal_score_avg": rec["chal_score_avg"],
                    "delta_avg":      rec["delta_avg"],
                    "parse_ok":       rec.get("parse_ok", True),
                    "error":          rec.get("error"),
                    "completed_at":   datetime.now(timezone.utc).isoformat(),
                })
                await out_queue.put(_sse_event("progress", {
                    "eval_id":         eval_id,
                    "n_done":          n_done,
                    "n_valid":         n_valid,
                    "n_total":         len(samples),
                    "king_mean":       king_avg_sum / max(n_valid, 1),
                    "chal_mean":       chal_avg_sum / max(n_valid, 1),
                    "mean_delta":      (chal_avg_sum - king_avg_sum) / max(n_valid, 1),
                    "parse_failures":  parse_failures,
                    "vllm_errors":     vllm_errors,
                    "judges":          _judges_summary(),
                    "last": {
                        "sample_idx":  rec["sample_idx"],
                        "turn_idx":    rec["turn_idx"],
                        "instance_id": rec["instance_id"],
                        "per_judge":   [
                            {"model": pj["model"],
                             "king_verdict": pj["king_verdict"],
                             "chal_verdict": pj["chal_verdict"]}
                            for pj in rec["per_judge"]
                        ],
                    },
                }))

            tasks = [asyncio.create_task(runner(s)) for s in samples]

            async def collector() -> None:
                # Drain `out_queue` into the SSE stream as tasks emit events.
                pending = len(tasks)
                while pending > 0:
                    item = await out_queue.get()
                    yield_buffer.append(item)
                    pending = sum(1 for t in tasks if not t.done())
                    if all(t.done() for t in tasks) and out_queue.empty():
                        break

            # Drain the queue while tasks complete. We can't yield from
            # inside `collector()` because we'd need a generator inside
            # a coroutine; instead, poll the queue + task statuses here.
            while True:
                done_count = sum(1 for t in tasks if t.done())
                if done_count >= len(tasks) and out_queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(out_queue.get(), timeout=SSE_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield _sse_event("heartbeat", {"ts": time.time(), "n_done": n_done})
                    continue
                yield item

            # Surface any task exceptions.
            for t in tasks:
                exc = t.exception()
                if exc is not None:
                    log.warning("turn task failed: %s", exc)
    finally:
        stop_event.set()
        with contextlib.suppress(Exception):
            await hb_task
        with contextlib.suppress(Exception):
            await STATE.chal_proc.stop()

    # 5. Verdict.
    # Dethrone rule: per-judge mean scores — beat the king on at least one
    # judge dimension, tie-or-better on the rest (no dimension where the king
    # clearly wins). Ensemble paired-bootstrap LCB is still reported for
    # diagnostics / history, but does not gate acceptance.
    min_turns = max(8, chain_config.DUEL_N_SAMPLES // 4)
    mean_delta, lcb, se = paired_bootstrap_lcb(
        per_turn_ensemble_deltas,
        resamples=chain_config.DUEL_BOOTSTRAP_RESAMPLES,
        alpha=chain_config.DUEL_ALPHA,
        rng_seed=seed,
    )

    judges_final: list[dict] = []
    tie_band = chain_config.JUDGE_TIE_BAND
    judge_outcomes: list[str] = []
    for jm in judge_models:
        acc = per_judge_acc[jm]
        n = max(acc["n"], 1)
        j_mean_delta, j_lcb, j_se = paired_bootstrap_lcb(
            acc["deltas"],
            resamples=chain_config.DUEL_BOOTSTRAP_RESAMPLES,
            alpha=chain_config.DUEL_ALPHA,
            rng_seed=seed + jm.encode(),
        )
        outcome = judge_dimension_outcome(j_mean_delta, tie_band=tie_band)
        judge_outcomes.append(outcome)
        judges_final.append({
            "model":           jm,
            "n":               acc["n"],
            "king_mean":       acc["king_sum"] / n,
            "chal_mean":       acc["chal_sum"] / n,
            "delta":           j_mean_delta,
            "lcb":             j_lcb,
            "se":              j_se,
            "outcome":         outcome,
            "verdicts_king":   dict(acc["verdicts_king"]),
            "verdicts_chal":   dict(acc["verdicts_chal"]),
            "parse_failures":  acc["parse_failures"],
        })

    accepted, dethrone_detail = dethrone_by_judge_dimensions(
        judge_outcomes, min_turns=min_turns, n_done=n_done, n_valid=n_valid,
    )

    verdict_record = {
        "type":      "verdict",
        "schema_version": EVAL_TRACE_SCHEMA_VERSION,
        "eval_id":   eval_id,
        "hotkey":    req.hotkey,
        "challenger": req.challenger,
        "accepted":  accepted,
        "dethrone":  dethrone_detail,
        "n_turns":   n_done,
        "n_valid_turns": n_valid,
        "n_vllm_errors": vllm_errors,
        "n_turns_total": len(samples),
        "king_mean": (king_avg_sum / n_valid) if n_valid else 0.0,
        "chal_mean": (chal_avg_sum / n_valid) if n_valid else 0.0,
        "mean_delta": mean_delta,
        "lcb_at_1_minus_alpha": lcb,
        "alpha": chain_config.DUEL_ALPHA,
        "eval_delta": chain_config.DUEL_EVAL_DELTA,
        "se": se,
        "parse_failures": parse_failures,
        "judges": judges_final,
        "judge_models": list(judge_models),
        "judge_model": chain_config.JUDGE_MODEL,  # primary, kept for back-compat
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    sink.append(verdict_record)

    # Best-effort upload — never blocks the verdict on Hippius being up.
    sink_info = await _safe_flush_sink(sink, flushed_ref)

    yield _sse_event("verdict", {
        **{k: v for k, v in verdict_record.items() if k != "type"},
        "per_turn":  per_turn_records,
        "evals":     sink_info,   # {url, key, bytes, uploaded, ...}
    })


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="albedo-eval")


@app.get("/health")
async def health() -> JSONResponse:
    dataset_dir = Path(DATASET_DIR)
    manifest = dataset_dir / trajectory_sampler.MANIFEST_NAME
    catalog_info: dict = {
        "dir": str(dataset_dir),
        "exists": dataset_dir.is_dir(),
        "manifest_exists": manifest.exists(),
        "pinned_manifest_sha256": chain_config.DATASET_MANIFEST_SHA256,
    }
    if dataset_dir.is_dir():
        try:
            catalog = trajectory_sampler.load_catalog(dataset_dir)
            catalog_info.update({
                "shards": len(catalog.shards),
                "total_rows": catalog.total_rows,
            })
        except Exception as exc:
            catalog_info["error"] = str(exc)
    return JSONResponse({
        "ok": True,
        "king": STATE.king_proc.health(),
        "challenger": STATE.chal_proc.health(),
        "eval_lock_held": STATE.eval_lock.locked(),
        "current_eval_id": STATE.current_eval_id,
        "disk": {
            "cache_dir": MODEL_CACHE_DIR,
            "free_bytes": disk_free_bytes(MODEL_CACHE_DIR),
            "tmp_dir": TMP_DIR,
            "tmp_free_bytes": disk_free_bytes(TMP_DIR),
            "min_required_bytes": MIN_DISK_BYTES,
        },
        "dataset": catalog_info,
        "chain": {
            "name": chain_config.NAME,
            "judge_models": list(chain_config.JUDGE_MODELS),
            "judge_model": chain_config.JUDGE_MODEL,  # primary, kept for back-compat
        },
    })


@app.post("/set_king")
async def set_king(req: SetKingRequest) -> JSONResponse:
    try:
        ref = ModelRef(req.king["repo"], req.king["digest"])
    except Exception as exc:
        raise HTTPException(400, f"bad king ref: {exc}")

    log.info("set_king: materializing %s", ref.immutable_ref)
    try:
        king_dir = await asyncio.to_thread(materialize_model, ref, None, 16)
    except Exception as exc:
        raise HTTPException(500, f"materialize_failed: {exc}")

    # materialize_model injects chat_template; restart vLLM if it was missing
    # (otherwise /set_king noop leaves a broken king running).
    try:
        await STATE.king_proc.start(king_dir, ref.immutable_ref)
    except Exception as exc:
        raise HTTPException(500, f"king_vllm_start_failed: {exc}")
    return JSONResponse({"status": "ok", "king": ref.immutable_ref})


class PruneCacheRequest(BaseModel):
    keep: list[dict] = []   # [{"repo": str, "digest": str}, ...]


@app.post("/prune_cache")
async def prune_cache_endpoint(req: PruneCacheRequest) -> JSONResponse:
    """Prune the model cache keeping only the listed model refs.
    Called by the validator on startup to remove stale weights from prior runs.
    Only fully-downloaded repos (with .safetensors) are deleted; config-only
    snapshots are left intact regardless of the keep list."""
    keep_refs: list[ModelRef] = []
    for entry in req.keep:
        try:
            keep_refs.append(ModelRef(entry["repo"], entry["digest"]))
        except Exception:
            pass
    freed = await asyncio.to_thread(prune_model_cache, *keep_refs)
    log.info(
        "startup cache prune via /prune_cache: freed %.2f GB, kept %d models",
        freed / 1e9, len(keep_refs),
    )
    return JSONResponse({
        "freed_bytes": freed,
        "freed_gb": round(freed / 1e9, 3),
        "kept": len(keep_refs),
    })


@app.post("/eval")
async def eval_endpoint(req: EvalRequest, request: Request) -> StreamingResponse:
    if STATE.eval_lock.locked():
        raise HTTPException(409, f"eval in progress: {STATE.current_eval_id}")

    async def stream() -> AsyncIterator[bytes]:
        async with STATE.eval_lock:
            STATE.current_eval_id = req.eval_id
            try:
                async for chunk in run_duel(req):
                    if await request.is_disconnected():
                        log.warning("validator disconnected mid-eval %s", req.eval_id)
                        break
                    yield chunk
            finally:
                STATE.current_eval_id = None

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.on_event("shutdown")
async def _shutdown() -> None:
    with contextlib.suppress(Exception):
        await STATE.chal_proc.stop()
    with contextlib.suppress(Exception):
        await STATE.king_proc.stop()


def main() -> int:
    import uvicorn
    uvicorn.run(
        "eval:app",
        host=os.environ.get("ALBEDO_EVAL_HOST", "0.0.0.0"),
        port=int(os.environ.get("ALBEDO_EVAL_PORT", "9000")),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
