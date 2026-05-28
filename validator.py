#!/usr/bin/env python3
"""Albedo validator — king-of-the-hill loop with LLM-as-judge duels.

Single async process; one outstanding duel at a time. Loop:

    poll chain reveals
        ↓ (filter by `seen` hotkeys and `completed_repos`)
    enqueue
        ↓ (burn hotkey at enqueue, not verdict — see §1-hotkey-1-eval)
    dispatch to eval.py /eval (SSE)
        ↓
    record verdict, maybe crown new king
        ↓ (on crown: POST /set_king to eval, set_weights on chain)

Ported from `teutonic-ref/validator.py`. Load-bearing patterns kept:
    - `_decode_commitment_pair` for v4 reveal decoding (bt 10.3 substrate
      returns raw bytes, not hex; one bad legacy row must not poison the
      scan).
    - `scan_reveals` 1-hotkey filter (intake gate).
    - `State.enqueue` 1-hotkey filter + completed_repos belt-and-suspenders.
    - `commit_reveal_enabled` startup assertion (refuse to run without CR
      so set_weights can't silently degrade to unencrypted form).
    - Async `set_weights` via run_in_executor with the "silent rate-limit
      no-op" bump on `success=False, message=""`.
    - Dashboard dual-write (Hippius first, R2 fallback, 60s cooldown).
    - Dashboard MUST NOT raise into the main loop.

Slimmed from teutonic:
    - Single king at 100% (no rolling N-king split in v1; add later).
    - No TaoMarketCap fetch in v1.
    - No re-eval / replenish (re-eval is permanently off in teutonic too).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import bittensor as bt
import boto3
import httpx
from botocore.config import Config as BotoConfig

import chain_config
from model_store import (
    ModelRef,
    list_remote_files,
    materialize_model,
    parse_reveal_v3,
    parse_reveal_v4,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NETUID  = int(os.environ.get("ALBEDO_NETUID", "0"))
NETWORK = os.environ.get("ALBEDO_NETWORK", "finney")
WALLET_NAME   = os.environ.get("BT_WALLET_NAME", "albedo")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")

EVAL_SERVER_URL = os.environ.get("ALBEDO_EVAL_SERVER", "http://127.0.0.1:9000")
SEED_REPO   = os.environ.get("ALBEDO_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("ALBEDO_SEED_DIGEST", chain_config.SEED_DIGEST)

R2_ENDPOINT   = os.environ.get("ALBEDO_R2_ENDPOINT", "")
R2_BUCKET     = os.environ.get("ALBEDO_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("ALBEDO_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("ALBEDO_R2_SECRET_KEY", "")

HIPPIUS_ENDPOINT   = os.environ.get("ALBEDO_DS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_BUCKET     = os.environ.get("ALBEDO_DS_BUCKET", "")
HIPPIUS_ACCESS_KEY = os.environ.get("ALBEDO_DS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY = os.environ.get("ALBEDO_DS_SECRET_KEY", "")

HIPPIUS_COOLDOWN_SECONDS = 60
DASHBOARD_FLUSH_MIN_INTERVAL = 5.0

POLL_INTERVAL    = int(os.environ.get("ALBEDO_POLL_INTERVAL", "30"))
WEIGHT_INTERVAL  = int(os.environ.get("ALBEDO_WEIGHT_INTERVAL", "300"))     # blocks
BURN_UID         = int(os.environ.get("ALBEDO_BURN_UID", "0"))
# Set ALBEDO_REQUIRE_COMMIT_REVEAL=0 in ecosystem.config.js to use plain
# set_weights (owner-operated subnets that haven't enabled CR yet).
REQUIRE_COMMIT_REVEAL = os.environ.get("ALBEDO_REQUIRE_COMMIT_REVEAL", "1").lower() not in (
    "0", "false", "no",
)

TICK_RESTART_AFTER = int(os.environ.get("ALBEDO_TICK_RESTART_AFTER", "2400"))
STREAM_IDLE_WARN_S = int(os.environ.get("ALBEDO_STREAM_IDLE_WARN_S", "600"))
STREAM_IDLE_KILL_S = int(os.environ.get("ALBEDO_STREAM_IDLE_KILL_S", "1800"))

MAX_CONSECUTIVE_TICK_ERRORS = int(os.environ.get("ALBEDO_MAX_CONSECUTIVE_TICK_ERRORS", "20"))

# How many hours back to scan history for our-side failures on startup, and
# re-queue those miners for another attempt.  Set to 0 to disable.
REEVAL_LOOKBACK_HOURS = float(os.environ.get("ALBEDO_REEVAL_LOOKBACK_HOURS", "24"))
# Maximum number of automatic re-evals granted per miner hotkey (across all
# paths: runtime maybe_retry AND startup lookback recovery combined).  Once a
# hotkey reaches this count it is skipped by both paths regardless of restarts.
MAX_REEVAL_PER_HOTKEY  = int(os.environ.get("ALBEDO_MAX_REEVAL_PER_HOTKEY", "1"))
# Initial eval-box backoff after an unreachable error (seconds).  Doubles on
# each consecutive failure up to EVAL_BOX_BACKOFF_MAX_S.
EVAL_BOX_BACKOFF_S     = int(os.environ.get("ALBEDO_EVAL_BOX_BACKOFF_S", "120"))
EVAL_BOX_BACKOFF_MAX_S = int(os.environ.get("ALBEDO_EVAL_BOX_BACKOFF_MAX_S", "1800"))

REPO_PATTERN_RE = re.compile(chain_config.REPO_PATTERN)

# Per-arch generic lock — preserved across all Albedo chains. The
# extra_lock_keys from chain.toml are appended on top.
_GENERIC_LOCK_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "model_type",
)

logging.basicConfig(
    level=os.environ.get("ALBEDO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("albedo.validator")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic_now() -> float:
    return time.monotonic()


def _ts(iso: str) -> float:
    """Parse an ISO-8601 timestamp string; return 0.0 on any error."""
    try:
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Object storage (R2 primary, Hippius for dashboard reads from miners)
# ---------------------------------------------------------------------------

class ObjectStore:
    """R2 (or any S3-shape) for validator-private state; Hippius for the
    public-facing dashboard. Hippius writes have a cooldown after failures
    so a Hippius outage cannot wedge the eval loop."""

    def __init__(self) -> None:
        cfg = dict(
            connect_timeout=15,
            read_timeout=45,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        if R2_ENDPOINT and R2_BUCKET and R2_ACCESS_KEY and R2_SECRET_KEY:
            self.client = boto3.client(
                "s3", endpoint_url=R2_ENDPOINT,
                aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
                region_name="auto",
                config=BotoConfig(**cfg),
            )
        else:
            log.warning("R2 not configured — state persistence disabled")
            self.client = None

        if HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY and HIPPIUS_BUCKET:
            self._hippius = boto3.client(
                "s3", endpoint_url=HIPPIUS_ENDPOINT,
                aws_access_key_id=HIPPIUS_ACCESS_KEY,
                aws_secret_access_key=HIPPIUS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    **cfg,
                ),
            )
            self._ensure_public_bucket()
        else:
            log.warning("Hippius not configured — public dashboard write disabled")
            self._hippius = None

        self._hippius_retry_after = 0.0

    def _ensure_public_bucket(self) -> None:
        """Idempotent: create the dashboard bucket if missing and grant
        anonymous s3:GetObject. Without this the dashboard JS gets 403s
        on a fresh deploy. Hippius S3 buckets are private by default."""
        try:
            self._hippius.head_bucket(Bucket=HIPPIUS_BUCKET)
        except Exception:
            try:
                self._hippius.create_bucket(Bucket=HIPPIUS_BUCKET)
                log.info("created Hippius bucket %s", HIPPIUS_BUCKET)
            except Exception as exc:
                log.warning("could not create Hippius bucket %s: %s", HIPPIUS_BUCKET, exc)
                return
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{HIPPIUS_BUCKET}/*",
            }],
        }
        try:
            self._hippius.put_bucket_policy(Bucket=HIPPIUS_BUCKET, Policy=json.dumps(policy))
        except Exception as exc:
            log.warning("put_bucket_policy on %s failed (non-fatal, may already be set): %s",
                        HIPPIUS_BUCKET, exc)

    def _hippius_available(self) -> bool:
        return self._hippius is not None and time.monotonic() >= self._hippius_retry_after

    def _mark_hippius_failure(self, key: str, exc: Exception) -> None:
        self._hippius_retry_after = time.monotonic() + HIPPIUS_COOLDOWN_SECONDS
        log.warning("Hippius write failed for %s; cooling %ss; falling back to R2: %s",
                    key, HIPPIUS_COOLDOWN_SECONDS, exc)

    def _put_dashboard_bytes(self, key: str, body: bytes, content_type: str,
                              cache_control: str | None = None) -> None:
        extra = {"CacheControl": cache_control} if cache_control else {}
        if self._hippius_available():
            try:
                self._hippius.put_object(
                    Bucket=HIPPIUS_BUCKET, Key=key, Body=body,
                    ContentType=content_type, **extra,
                )
                return
            except Exception as exc:
                self._mark_hippius_failure(key, exc)
        if self.client:
            try:
                self.client.put_object(
                    Bucket=R2_BUCKET, Key=key, Body=body,
                    ContentType=content_type, **extra,
                )
            except Exception:
                log.warning("R2 dashboard fallback put failed for %s (non-fatal)",
                            key, exc_info=True)

    def put_dashboard(self, key: str, data: dict) -> None:
        body = json.dumps(data, default=str).encode()
        self._put_dashboard_bytes(key, body, "application/json")

    def put_dashboard_raw(self, key: str, body: bytes, content_type: str,
                          cache_control: str | None = None) -> None:
        self._put_dashboard_bytes(key, body, content_type, cache_control=cache_control)

    def put(self, key: str, data: dict) -> None:
        if not self.client:
            return
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=json.dumps(data, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:
            log.warning("R2 put failed for %s (non-fatal)", key, exc_info=True)

    def get(self, key: str) -> dict | None:
        if not self.client:
            return None
        try:
            return json.loads(
                self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Chain reveal decoding + scan (ported from teutonic 555-616 verbatim shape)
# ---------------------------------------------------------------------------

def _decode_commitment_pair(pair: Any) -> tuple[str, list[tuple[int, str]]]:
    """Per-pair decoder. Avoids `decode_revealed_commitment_with_hotkey`
    which (a) raises on a single bad legacy row and poisons the whole scan
    and (b) assumes hex-encoded payloads in bt 10.3 while substrate returns
    raw bytes. Both bugs share a single fix — decode it ourselves."""
    hotkey_raw, entries_raw = pair
    if hasattr(hotkey_raw, "value"):
        hotkey_ss58 = hotkey_raw.value
    else:
        hotkey_ss58 = str(hotkey_raw)
    entries = entries_raw.value if hasattr(entries_raw, "value") else entries_raw
    out: list[tuple[int, str]] = []
    for entry in entries or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        text, block = entry
        if not isinstance(text, str):
            raise ValueError(f"unexpected commitment payload type {type(text).__name__}")
        if text.startswith(("0x", "0X")):
            raw = bytes.fromhex(text[2:])
        else:
            raw = text.encode("latin-1")
        if not raw:
            raise ValueError("empty commitment payload")
        mode = raw[0] & 0b11
        offset = 1 if mode == 0 else 2 if mode == 1 else 4
        out.append((block, raw[offset:].decode("utf-8", errors="ignore")))
    return hotkey_ss58, out


def scan_reveals(subtensor, netuid: int,
                  completed_repos: set[str], seen_hotkeys: set[str]) -> list[dict]:
    """Pull v4 reveals; return latest per hotkey not previously enqueued."""
    try:
        query = subtensor.query_map(module="Commitments", name="RevealedCommitments",
                                     params=[netuid])
    except Exception:
        log.exception("query_map RevealedCommitments failed")
        return []
    all_reveals: dict[str, list[tuple[int, str]]] = {}
    bad = 0
    for pair in query:
        try:
            hotkey_ss58, entries = _decode_commitment_pair(pair)
            all_reveals[hotkey_ss58] = entries
        except Exception:
            bad += 1
    if bad:
        log.warning("scan_reveals: skipped %d undecodable on-chain commitments", bad)
    if not all_reveals:
        return []

    new: list[dict] = []
    for hotkey, entries in all_reveals.items():
        if not entries or hotkey in seen_hotkeys:
            continue
        block, data = max(entries, key=lambda e: e[0])
        try:
            ref, author_hotkey = parse_reveal_v4(data)
        except ValueError:
            try:
                legacy_king, _legacy_ref, _legacy_author = parse_reveal_v3(data)
            except ValueError:
                continue
            log.warning("dropping legacy v3 reveal from %s at block %s (king_digest=%s)",
                        hotkey[:16], block, legacy_king[:19])
            continue
        if author_hotkey != hotkey:
            log.warning("v4 author_hotkey %s != chain key %s; trusting chain",
                        author_hotkey[:16], hotkey[:16])
        if not ref.digest.startswith("sha256:"):
            log.warning("dropping non-Hippius reveal from %s: digest=%s (sha256: required)",
                        hotkey[:16], ref.digest[:19])
            continue
        if ref.immutable_ref in completed_repos:
            continue
        new.append({
            "hotkey": hotkey,
            "block": block,
            "model_repo": ref.repo,
            "model_digest": ref.digest,
        })
    new.sort(key=lambda x: x["block"])
    return new


def _reeval_from_file(subtensor, netuid: int, state: "State") -> int:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "to_reeval.json")
    if not os.path.exists(path):
        return 0

    try:
        with open(path) as f:
            hotkeys = json.load(f)
    except Exception as exc:
        log.warning("_reeval_from_file: could not read %s: %s", path, exc)
        return 0

    if not isinstance(hotkeys, list):
        log.warning("_reeval_from_file: expected list in %s, got %s",
                    path, type(hotkeys).__name__)
        os.unlink(path)
        return 0

    target: set[str] = {h for h in hotkeys if isinstance(h, str) and h}
    if not target:
        log.info("_reeval_from_file: empty hotkey list — nothing to re-queue")
        os.unlink(path)
        return 0

    log.info("_reeval_from_file: looking up chain reveals for %d hotkey(s)", len(target))

    try:
        query = subtensor.query_map(module="Commitments", name="RevealedCommitments",
                                     params=[netuid])
    except Exception:
        log.exception("_reeval_from_file: query_map failed")
        os.unlink(path)
        return 0

    reveals: dict[str, dict] = {}
    bad = 0
    for pair in query:
        try:
            hotkey_ss58, entries = _decode_commitment_pair(pair)
        except Exception:
            bad += 1
            continue
        if hotkey_ss58 not in target:
            continue
        if not entries:
            continue
        block, data = max(entries, key=lambda e: e[0])
        try:
            ref, author_hotkey = parse_reveal_v4(data)
        except ValueError:
            continue
        if author_hotkey != hotkey_ss58:
            continue
        if not ref.digest.startswith("sha256:"):
            continue
        reveals[hotkey_ss58] = {
            "hotkey": hotkey_ss58,
            "block": block,
            "model_repo": ref.repo,
            "model_digest": ref.digest,
        }

    if bad:
        log.warning("_reeval_from_file: skipped %d undecodable chain entries", bad)

    missing = target - set(reveals)
    if missing:
        log.warning("_reeval_from_file: no chain reveal found for %d hotkey(s): %s",
                    len(missing), [h[:16] for h in sorted(missing)])

    n = 0
    for rev in reveals.values():
        cid = state.enqueue(rev, force=True)
        if cid:
            n += 1
            log.info("_reeval_from_file: re-queued %s for %s", cid, rev["hotkey"][:16])

    try:
        os.unlink(path)
        log.info("_reeval_from_file: deleted %s (%d queued)", path, n)
    except Exception as exc:
        log.warning("_reeval_from_file: could not delete %s: %s", path, exc)

    return n


_KING_CONFIG_CACHE: dict[str, dict] = {}


def _load_king_config(king_repo: str, king_digest: str) -> dict | None:
    key = f"{king_repo}@{king_digest}"
    if key in _KING_CONFIG_CACHE:
        return _KING_CONFIG_CACHE[key]
    try:
        ref = ModelRef(king_repo, king_digest)
        snap = materialize_model(ref, max_workers=4, config_only=True)
        with open(os.path.join(snap, "config.json")) as f:
            cfg = json.load(f)
        _KING_CONFIG_CACHE[key] = cfg
        return cfg
    except Exception:
        log.exception("could not load king config for %s@%s", king_repo, king_digest[:19])
        return None


def _exc_no_paths(exc: Exception) -> str:
    msg = str(exc)
    msg = re.sub(r""":\s+['"]/[^'"]*['"]""", "", msg)
    msg = re.sub(r"\s/(?:home|root|tmp|var|srv|opt|etc)/\S+", "", msg)
    return msg.strip(": ").strip()


def validate_challenger_config(model_repo: str, challenger_digest: str,
                                 king_repo: str, king_digest: str) -> str | None:
    """Architecture / shape lock + repo hygiene gate. Runs BEFORE we ship
    the challenger to the eval server so a malformed submission burns
    only a config-only fetch (~50 KB), not a vLLM bring-up cycle.

    Returns None on pass, or a human-readable rejection reason string.

    Defends against:
      - tokenizer/arch swaps (vocab_size / hidden_size / etc. mismatches)
      - custom modeling via `auto_map` (would let challenger run code
        during HF Auto-load if trust_remote_code ever gets flipped on)
      - `*.py` files shipped in the repo (same threat surface)
      - missing safetensors (forces vLLM startup failure)
      - oversized repos (disk-fill DOS via fp32/fp64 weights or duplicated
        shards beyond MAX_CHALLENGER_SAFETENSORS_GB)
      - repo name not matching the chain pattern
    """
    if not REPO_PATTERN_RE.fullmatch(model_repo):
        return f"repo name {model_repo!r} does not match required pattern {REPO_PATTERN_RE.pattern}"
    if not (challenger_digest or "").startswith("sha256:"):
        return f"challenger digest must be Hippius OCI sha256:… (got {challenger_digest!r})"

    king_cfg = _load_king_config(king_repo, king_digest)
    if not king_cfg:
        # If we can't load king config we can't enforce arch lock; rather
        # than block the queue, let the duel proceed and let vLLM startup
        # catch any catastrophic mismatch.
        log.warning("validate_challenger_config: king cfg unavailable; skipping lock check")
        return None

    # Phase 1: download / network — infra side.  Any failure here is
    # transient (Hippius outage, bad digest, partial download).
    try:
        ref = ModelRef(model_repo, challenger_digest)
        snap = materialize_model(ref, max_workers=4, config_only=True)
    except Exception as exc:
        return f"cannot materialize challenger config: {_exc_no_paths(exc)}"

    # Phase 2: parse config.json — miner's responsibility.
    # FileNotFoundError means the repo has no config.json at all.
    # JSON decode errors mean the config.json is malformed.
    # Neither is an infra failure — do NOT grant a retry.
    try:
        cfg_path = os.path.join(snap, "config.json")
        with open(cfg_path) as f:
            chall_cfg = json.load(f)
    except FileNotFoundError:
        return "config.json missing from challenger repo"
    except Exception as exc:
        return f"config.json unreadable in challenger repo: {_exc_no_paths(exc)}"

    # Phase 3: list repo files — infra side (S3 manifest fetch).
    try:
        repo_files = list_remote_files(ref)
    except Exception as exc:
        return f"cannot list challenger repo files: {_exc_no_paths(exc)}"

    _ARCH_SENTINEL = object()
    king_arch = king_cfg.get("architectures", _ARCH_SENTINEL)
    chall_arch = chall_cfg.get("architectures", _ARCH_SENTINEL)
    if king_arch != chall_arch:
        ka_str = king_arch if king_arch is not _ARCH_SENTINEL else "<absent>"
        ca_str = chall_arch if chall_arch is not _ARCH_SENTINEL else "<absent>"
        return f"architecture mismatch: king={ka_str!r} challenger={ca_str!r}"

    _SENTINEL = object()
    for key in _GENERIC_LOCK_KEYS + tuple(chain_config.EXTRA_LOCK_KEYS):
        k = king_cfg.get(key, _SENTINEL)
        c = chall_cfg.get(key, _SENTINEL)
        if k != c:
            k_str = k if k is not _SENTINEL else "<absent>"
            c_str = c if c is not _SENTINEL else "<absent>"
            return f"{key} mismatch: king={k_str} challenger={c_str}"

    if "auto_map" in chall_cfg:
        return "auto_map present in config.json (custom modeling code is not allowed)"

    py_files = [f for f in repo_files if f.endswith(".py")]
    if py_files:
        return f"repo ships *.py files (not allowed): {py_files[:3]}"

    st_files = [f for f in repo_files if f.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in challenger repo"
    if len(st_files) > 256:
        return f"too many safetensors shards ({len(st_files)}); refusing oversized layout"

    return None


def _is_infra_failure(verdict: dict) -> bool:
    """True when the eval server aborted before a meaningful duel ran."""
    if verdict.get("error"):
        return True
    n_turns = int(verdict.get("n_turns") or 0)
    n_valid = int(verdict.get("n_valid_turns") or 0)
    return n_turns == 0 and n_valid == 0


def _is_miner_fault(code: str, detail: str) -> bool:
    """True when the failure is the miner's own fault and no retry should
    be granted. False for transient infrastructure / network errors.

    Miner fault:
      - config_mismatch (wrong arch, forbidden files, etc.) — UNLESS the
        failure was a transient 404 / materialize fetch error.
      - chal_vllm_start_failed — the challenger model crashed on load;
        the weights themselves are broken.
      - no_king — validator internal state issue; not retriable.

    Infra/transient (retry OK):
      - eval_infra, eval_error, eval_http, no_verdict, hard_timeout
      - config_mismatch caused by a 404 or materialize_failed fetch error
    """
    if code == "no_king":
        return True
    if code == "config_mismatch":
        d = detail or ""
        if ("cannot materialize" in d or "cannot list challenger" in d
                or "materialize_failed" in d or "404" in d):
            return False
        return True
    if "chal_vllm_start_failed" in (detail or ""):
        return True
    return False


def _recover_from_lookback(state: "State") -> int:
    if REEVAL_LOOKBACK_HOURS <= 0:
        return 0

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - REEVAL_LOOKBACK_HOURS * 3600
    king_hotkey = (state.king or {}).get("hotkey", "")
    queued_hotkeys: set[str] = {e.get("hotkey") for e in state.queue}

    by_hotkey: dict[str, list[dict]] = {}
    for h in state.history:
        hk = h.get("hotkey", "")
        if hk:
            by_hotkey.setdefault(hk, []).append(h)

    requeued = 0
    for hotkey, entries in by_hotkey.items():
        if hotkey == king_hotkey or hotkey in queued_hotkeys:
            continue
        if state.retry_counts.get(hotkey, 0) >= MAX_REEVAL_PER_HOTKEY:
            continue  # budget already consumed by a prior runtime retry

        fail_entry: dict | None = None
        fail_ts = 0.0
        for h in reversed(entries):
            code = h.get("error_code", "")
            if not code:
                continue  # valid eval, not a failure
            if _is_miner_fault(code, h.get("error_detail", "")):
                continue  # miner caused this — no recovery
            cid = h.get("challenge_id", "")
            if cid in state.recovered_ids:
                continue  # already recovered or marked by maybe_retry
            ft = _ts(h.get("completed_at", ""))
            if ft < cutoff:
                continue  # outside lookback window
            fail_entry = h
            fail_ts = ft
            break  # newest-first — take the first candidate found

        if fail_entry is None:
            continue

        # Check if there's already a valid eval for this hotkey recorded AFTER
        # the failure — scan only this hotkey's entries, not all history.
        already_evaluated = any(
            not hh.get("error_code") and _ts(hh.get("completed_at", "")) > fail_ts
            for hh in entries
        )
        if already_evaluated:
            continue

        cid = fail_entry.get("challenge_id", "")
        requeue_entry = {
            "challenge_id": cid,
            "hotkey": hotkey,
            "model_repo": fail_entry.get("model_repo", ""),
            "model_digest": fail_entry.get("model_digest", ""),
            "block": fail_entry.get("block", 0),
            "queued_at": _now(),
        }
        state.retry_counts[hotkey] = state.retry_counts.get(hotkey, 0) + 1
        state.recovered_ids.add(cid)
        state.queue.append(requeue_entry)
        queued_hotkeys.add(hotkey)
        requeued += 1
        log.info("lookback recovery: re-queued %s for %s (code=%s, age=%.1fh)",
                 cid, hotkey[:16], fail_entry.get("error_code"),
                 (now_ts - fail_ts) / 3600)

    return requeued


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    def __init__(self, store: ObjectStore) -> None:
        self.store = store
        self.king: dict = {}
        # Past kings, most-recent-first. Combined with `king` for the
        # rolling N-king split in maybe_set_weights. Holds up to
        # (DUEL_KING_CHAIN_DEPTH - 1) entries.
        self.king_chain: list[dict] = []
        self.queue: list[dict] = []
        # 1-hotkey-1-eval: a miner gets exactly one shot per hotkey
        # registration, period. Hotkey is burned at enqueue, not at verdict
        # — see README/DESIGN for why.
        self.seen: set[str] = set()
        self.completed_repos: set[str] = set()
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self.counter = 0
        self.current_eval: dict | None = None
        self.history: list[dict] = []
        self.last_weight_block = 0
        self.uid_map: dict[str, int] = {}
        self.coldkey_for: dict[str, str] = {}
        self.watchdog = {
            "started_at": _now(),
            "phase": "startup",
            "phase_since": _now(),
            "last_state_flush_at": None,
            "last_dashboard_flush_at": None,
            "consecutive_tick_errors": 0,
        }
        self.infra_cooldown: dict[str, float] = {}
        self._last_disk_warn_monotonic = 0.0
        self._last_dashboard_flush_monotonic = 0.0
        # Tracks how many automatic retries have been granted per hotkey.
        # Capped at 1 so a miner can never loop forever on transient errors.
        self.retry_counts: dict[str, int] = {}
        # Set of challenge_ids that have been re-queued via the lookback
        # recovery scan.  Persisted so restarts don't double-recover the
        # same failures.
        self.recovered_ids: set[str] = set()
        # Eval-box back-off: monotonic deadline before which we skip the
        # queue, and consecutive-failure counter for exponential increase.
        # NOT persisted — a validator restart is itself a recovery event.
        self.eval_box_retry_after: float = 0.0
        self.eval_box_consecutive_fails: int = 0

    def load(self) -> None:
        k = self.store.get("king/current.json")
        if k:
            self.king = k
        kc = self.store.get("state/king_chain.json")
        if kc:
            self.king_chain = kc.get("chain", [])
        q = self.store.get("state/queue.json")
        if q:
            self.queue = q.get("pending", [])
        s = self.store.get("state/seen_hotkeys.json")
        if s:
            self.seen = set(s.get("hotkeys", []))
        cr = self.store.get("state/completed_repos.json")
        if cr:
            self.completed_repos = set(cr.get("repos", []))
        st = self.store.get("state/validator_state.json")
        if st:
            self.stats = st.get("stats", self.stats)
            self.counter = st.get("counter", 0)
            self.last_weight_block = st.get("last_weight_block", 0)
            self.retry_counts = st.get("retry_counts", {})
            self.recovered_ids = set(st.get("recovered_ids", []))
        h = self.store.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])

        log.info("loaded state: king=%s@%s past_kings=%d queue=%d seen=%d completed=%d",
                 self.king.get("model_repo", "none"),
                 (self.king.get("king_digest") or "")[:12],
                 len(self.king_chain),
                 len(self.queue), len(self.seen), len(self.completed_repos))

    def flush(self) -> None:
        now = _now()
        self.watchdog["last_state_flush_at"] = now
        self.store.put("state/validator_state.json", {
            "stats": self.stats, "counter": self.counter,
            "last_weight_block": self.last_weight_block,
            "retry_counts": self.retry_counts,
            "recovered_ids": sorted(self.recovered_ids),
            "updated_at": now,
        })
        self.store.put("state/queue.json", {"pending": self.queue, "updated_at": now})
        self.store.put("king/current.json", self.king)
        self.store.put("state/king_chain.json",
                       {"chain": self.king_chain, "updated_at": now})
        self.store.put("state/seen_hotkeys.json",
                       {"hotkeys": sorted(self.seen), "updated_at": now})
        self.store.put("state/completed_repos.json",
                       {"repos": sorted(self.completed_repos), "updated_at": now})
        self.store.put("state/dashboard_history.json",
                       {"history": self.history, "updated_at": now})

    def next_id(self) -> str:
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def enqueue(self, reveal: dict, *, force: bool = False) -> str | None:
        """The 1-hotkey-1-eval enforcement is HERE (plus scan_reveals).
        Both gates are required; scan_reveals filters intake but a
        validator restart can race with a re-scan, so this is the
        belt-and-suspenders.

        force=True bypasses the seen/completed_repos checks; used by
        _reeval_from_file to re-queue specific hotkeys on operator request.
        """
        repo = reveal.get("model_repo", "")
        digest = reveal.get("model_digest", "")
        model_key = f"{repo}@{digest}" if digest else repo
        hotkey = reveal.get("hotkey", "")
        king_hotkey = self.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("skipping enqueue: hotkey %s is the current king", hotkey[:16])
            return None
        if not force and hotkey and hotkey in self.seen:
            log.info("skipping enqueue: hotkey %s already used its 1-eval slot "
                     "(must re-register for another shot)", hotkey[:16])
            return None
        for existing in self.queue:
            if existing.get("hotkey") == hotkey:
                log.info("skipping enqueue: hotkey %s already has an entry in queue "
                         "(different model submitted while retry is pending?)", hotkey[:16])
                return None
            if existing.get("model_repo") == repo:
                log.info("skipping duplicate repo: %s already queued", repo)
                return None
        if not force and model_key in self.completed_repos:
            log.info("skipping %s: repo already evaluated", repo)
            return None
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now()}
        self.queue.append(entry)
        self.stats["queued"] += 1
        # Burn at enqueue, not at verdict. Crash between enqueue and verdict
        # loses this miner's shot — that's the intended policy.
        if hotkey:
            self.seen.add(hotkey)
        if repo:
            self.completed_repos.add(model_key)
        self.flush()
        self.flush_dashboard(force=True)
        return cid

    def unburn_challenge(self, entry: dict) -> None:
        """Return a miner's 1-eval slot after an infra failure (disk, vLLM, etc.).

        Only call for failures where the duel never meaningfully ran — not for
        completed duels that lost on merit.
        """
        hotkey = entry.get("hotkey", "")
        repo = entry.get("model_repo", "")
        digest = entry.get("model_digest", "")
        model_key = f"{repo}@{digest}" if digest else repo
        if hotkey and hotkey in self.seen:
            self.seen.discard(hotkey)
            log.info("unburned hotkey %s after infra failure", hotkey[:16])
        if model_key and model_key in self.completed_repos:
            self.completed_repos.discard(model_key)
            log.info("unburned repo %s after infra failure", model_key[:48])

    def maybe_retry(self, entry: dict, code: str, detail: str) -> bool:
        hotkey = entry.get("hotkey", "")
        cid = entry.get("challenge_id", "?")
        if _is_miner_fault(code, detail):
            log.info("%s: no retry — miner-fault error (code=%s hotkey=%s)",
                     cid, code, hotkey[:16] if hotkey else "?")
            return False
        used = self.retry_counts.get(hotkey, 0)
        if used >= MAX_REEVAL_PER_HOTKEY:
            log.info("%s: retry budget exhausted for %s (%d/%d, code=%s); marking final",
                     cid, hotkey[:16] if hotkey else "?", used, MAX_REEVAL_PER_HOTKEY, code)
            return False

        self.retry_counts[hotkey] = used + 1
        # Mark this challenge_id so _recover_from_lookback() on the next
        # restart won't see it as an unprocessed failure and re-queue again.
        if cid != "?":
            self.recovered_ids.add(cid)
        self.queue.insert(0, entry)
        log.info("%s: retry %d/%d granted for %s (code=%s); re-queued at front",
                 cid, used + 1, MAX_REEVAL_PER_HOTKEY, hotkey[:16] if hotkey else "?", code)
        return True

    def set_king(self, hotkey: str, model_repo: str, model_digest: str,
                  block: int, challenge_id: str = "seed",
                  *, dethrone_judges: list[dict] | None = None,
                  crown_judges: list[dict] | None = None) -> None:

        prev = self.king
        is_real_transition = (
            challenge_id != "seed"
            and prev
            and prev.get("hotkey")
            and prev.get("hotkey") != hotkey
        )
        if is_real_transition:
            prev = dict(prev)
            if dethrone_judges:
                prev["judges"] = dethrone_judges
            # Dedup: if `prev`'s hotkey is already in the chain, remove
            # the older entry before unshifting — a hotkey that reclaims
            # the throne shouldn't double-dip.
            self.king_chain = [
                e for e in self.king_chain if e.get("hotkey") != prev.get("hotkey")
            ]
            self.king_chain.insert(0, prev)
            # Keep at most (depth - 1) past kings; the current king is
            # tracked separately on `self.king`.
            max_past = max(0, chain_config.DUEL_KING_CHAIN_DEPTH - 1)
            self.king_chain = self.king_chain[:max_past]

        reign = prev.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        self.king = {
            "hotkey": hotkey,
            "model_repo": model_repo,
            "king_digest": model_digest,
            "crowned_at": _now(),
            "crowned_block": int(block),
            "reign_number": reign,
            "challenge_id": challenge_id,
            "judges": crown_judges or [],
        }
        log.info("crowned new king: hotkey=%s repo=%s digest=%s reign=#%d "
                 "(past_kings=%d, depth=%d)",
                 hotkey[:16] if hotkey else "?", model_repo,
                 (model_digest or "")[:19], reign,
                 len(self.king_chain), chain_config.DUEL_KING_CHAIN_DEPTH)
        self.flush()

    def refresh_uid_map(self, subtensor, netuid: int) -> None:
        try:
            meta = subtensor.metagraph(netuid)
        except Exception:
            log.exception("metagraph refresh failed (non-fatal)")
            return
        self.uid_map = {hk: i for i, hk in enumerate(meta.hotkeys)}
        coldkeys = getattr(meta, "coldkeys", None) or []
        self.coldkey_for = {hk: (coldkeys[i] if i < len(coldkeys) else "")
                             for i, hk in enumerate(meta.hotkeys)}

    def record_verdict(self, entry: dict, verdict: dict) -> None:
        evals = verdict.get("evals") or {}
        rec = {
            "challenge_id": entry.get("challenge_id"),
            "hotkey": entry.get("hotkey"),
            "uid": self.uid_map.get(entry.get("hotkey")),
            "model_repo": entry.get("model_repo"),
            "model_digest": entry.get("model_digest"),
            "accepted": verdict.get("accepted", False),
            "king_mean": verdict.get("king_mean", 0.0),
            "chal_mean": verdict.get("chal_mean", 0.0),
            "mean_delta": verdict.get("mean_delta", 0.0),
            "lcb": verdict.get("lcb_at_1_minus_alpha", 0.0),
            "n_turns": verdict.get("n_turns", 0),
            "n_valid_turns": verdict.get("n_valid_turns", 0),
            "n_vllm_errors": verdict.get("n_vllm_errors", 0),
            "parse_failures": verdict.get("parse_failures", 0),
            "verdicts_king": verdict.get("verdicts_king", {}),
            "verdicts_chal": verdict.get("verdicts_chal", {}),
            "judges": verdict.get("judges", []),
            "dethrone": verdict.get("dethrone", {}),
            "judge_model": verdict.get("judge_model", ""),
            "evals_url":   evals.get("url"),
            "evals_key":   evals.get("key"),
            "evals_bytes": evals.get("bytes"),
            "completed_at": _now(),
        }
        self.history.append(rec)
        # keep last 200 entries on R2; dashboard.json gets a slice anyway
        self.history = self.history[-200:]
        if verdict.get("accepted"):
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1

    def record_failure(self, entry: dict, code: str, detail: str) -> None:
        self.history.append({
            "challenge_id": entry.get("challenge_id"),
            "hotkey": entry.get("hotkey"),
            "uid": self.uid_map.get(entry.get("hotkey")),
            "model_repo": entry.get("model_repo"),
            "model_digest": entry.get("model_digest"),
            "error_code": code,
            "error_detail": detail,
            "completed_at": _now(),
        })
        self.history = self.history[-200:]
        self.stats["failed"] += 1

    def flush_dashboard(self, *, force: bool = False) -> bool:
        # MUST NOT raise into the main loop. A Hippius/R2 outage during
        # dashboard write must not break an in-flight eval.
        try:
            now_mon = _monotonic_now()
            if not force and (now_mon - self._last_dashboard_flush_monotonic) < DASHBOARD_FLUSH_MIN_INTERVAL:
                return False
            self._last_dashboard_flush_monotonic = now_mon
            self.watchdog["last_dashboard_flush_at"] = _now()
            king_hk = self.king.get("hotkey") if self.king else None


            eligible = _eligible_chain_hotkeys(self) if (king_hk or self.king_chain) else []
            equal_share = round(1.0 / len(eligible), 9) if eligible else 0.0

            def _chain_entry(e: dict) -> dict:
                hk = e.get("hotkey", "")
                registered = hk in self.uid_map
                return {
                    "hotkey":        hk,
                    "uid":           self.uid_map.get(hk),
                    "coldkey":       self.coldkey_for.get(hk, ""),
                    "model_repo":    e.get("model_repo", ""),
                    "king_digest":   e.get("king_digest", ""),
                    "reign_number":  e.get("reign_number"),
                    "crowned_at":    e.get("crowned_at"),
                    "crowned_block": e.get("crowned_block"),
                    "challenge_id":  e.get("challenge_id"),
                    # weight is None if hotkey is deregistered (not in
                    # uid_map) so the website can render it as a dimmed
                    # row instead of falsely promising emission.
                    "weight":        equal_share if registered else None,
                    "weight_share":  equal_share if registered else None,
                    "registered":    registered,
                    "judges":        e.get("judges") or [],
                }

            dashboard_king_chain: list[dict] = []
            if self.king:
                dashboard_king_chain.append(_chain_entry(self.king))
            for e in self.king_chain:
                dashboard_king_chain.append(_chain_entry(e))

            payload = {
                "updated_at": _now(),
                "chain": {
                    "name": chain_config.NAME,
                    "seed_repo": chain_config.SEED_REPO,
                    "seed_digest": chain_config.SEED_DIGEST,
                    "judge_models": list(chain_config.JUDGE_MODELS),
                    "judge_model": chain_config.JUDGE_MODEL,  # primary, kept for back-compat
                    "judge_tie_band": chain_config.JUDGE_TIE_BAND,
                    "dataset_repo": chain_config.DATASET_REPO,
                    "dataset_shard_glob": chain_config.DATASET_SHARD_GLOB,
                    "king_chain_depth": getattr(chain_config, "DUEL_KING_CHAIN_DEPTH", 1),
                },
                "king": {
                    **self.king,
                    "uid": self.uid_map.get(king_hk) if king_hk else None,
                    "coldkey": self.coldkey_for.get(king_hk, "") if king_hk else "",
                },
                "king_chain": dashboard_king_chain,
                "stats": self.stats,
                "current_eval": self.current_eval,
                "watchdog": self.watchdog,
                "queue": [
                    {"challenge_id": e.get("challenge_id"),
                     "hotkey": e.get("hotkey"),
                     "uid": self.uid_map.get(e.get("hotkey", "")),
                     "coldkey": self.coldkey_for.get(e.get("hotkey", ""), ""),
                     "model_repo": e.get("model_repo"),
                     "model_digest": e.get("model_digest"),
                     "queued_at": e.get("queued_at"),
                     "block": e.get("block")}
                    for e in self.queue
                ],
                "history": self.history,
            }
            self.store.put_dashboard("dashboard.json", payload)
            return True
        except Exception:
            log.warning("flush_dashboard failed (non-fatal)", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# set_weights
# ---------------------------------------------------------------------------

def _eligible_chain_hotkeys(state: State) -> list[str]:
    """Ordered list (current king first) of hotkeys that should receive
    emission this tick — registered on the metagraph and capped at
    DUEL_KING_CHAIN_DEPTH. Deduped by hotkey; deregistered hotkeys are
    silently dropped so the live splits renormalize."""
    out: list[str] = []
    cap = chain_config.DUEL_KING_CHAIN_DEPTH
    king_hk = (state.king or {}).get("hotkey", "")
    if king_hk:
        out.append(king_hk)
    for e in state.king_chain:
        hk = e.get("hotkey", "")
        if hk and hk not in out:
            out.append(hk)
        if len(out) >= cap:
            break
    return [hk for hk in out[:cap] if hk in state.uid_map]


async def maybe_set_weights(subtensor, wallet, state: State, *,
                             force: bool = False, reason: str = "") -> bool:
    try:
        current_block = subtensor.block
    except Exception:
        log.exception("failed to read current block for weight-set")
        return False
    if not force and current_block - state.last_weight_block < WEIGHT_INTERVAL:
        return False

    eligible = _eligible_chain_hotkeys(state)
    if eligible:
        target_uids = [int(state.uid_map[hk]) for hk in eligible]
        share = round(1.0 / len(eligible), 9)
        weights_list = [share] * len(eligible)
        log_target = (
            f"uids={target_uids} share={share:.4f} each "
            f"({len(eligible)} kings)"
        )
    else:
        target_uids = [BURN_UID]
        weights_list = [1.0]
        log_target = f"burn uid={BURN_UID} (no registered king)"

    log.info("set_weights at block %d (last=%d, %s) -> %s",
             current_block, state.last_weight_block,
             reason or ("forced" if force else "interval"), log_target)
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: subtensor.set_weights(
                wallet=wallet, netuid=NETUID, uids=target_uids, weights=weights_list,
            ),
        )
    except Exception:
        log.exception("set_weights threw")
        return False

    if not resp.success:
        if not resp.message:
            log.info("set_weights rate-limited (no-op); advancing last_weight_block")
            state.last_weight_block = current_block
        else:
            log.error("set_weights failed: %s", resp.message)
        return False
    state.last_weight_block = current_block
    state.flush()
    state.flush_dashboard(force=True)
    return True


# ---------------------------------------------------------------------------
# Duel dispatch (talks to eval.py SSE)
# ---------------------------------------------------------------------------

def _compute_seed(subtensor, hotkey: str) -> bytes:
    """seed = blake2b(block_hash_at_reveal_height || hotkey).

    bt's block_hash call can flake; fall back to a deterministic mix of
    current block + hotkey so the fixture set is still determined and
    miner-verifiable without depending on an exact historical hash."""
    try:
        block = subtensor.block
        block_hash = subtensor.substrate.get_block_hash(block)
        if isinstance(block_hash, str) and block_hash.startswith("0x"):
            block_hash_b = bytes.fromhex(block_hash[2:])
        elif isinstance(block_hash, (bytes, bytearray)):
            block_hash_b = bytes(block_hash)
        else:
            block_hash_b = str(block_hash).encode()
    except Exception:
        block_hash_b = str(int(time.time() // 600)).encode()
    return hashlib.blake2b(block_hash_b + hotkey.encode(), digest_size=32).digest()


async def _eval_disk_ok(http: httpx.AsyncClient, *, queue_len: int = 0) -> tuple[bool, int, int]:
    """Return (ok, free_bytes, min_bytes) from eval server /health.

    Connection failures return ok=False so the queue is deferred instead of
    burning through entries while the eval box / tunnel is down.
    """
    try:
        r = await http.get(f"{EVAL_SERVER_URL}/health", timeout=15.0)
        r.raise_for_status()
        disk = r.json().get("disk") or {}
        free_b = int(disk.get("free_bytes") or 0)
        min_b = int(disk.get("min_required_bytes") or (6 * 1024 ** 3))
        return free_b >= min_b, free_b, min_b
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
        log.warning(
            "eval server unreachable; deferring queue (%d pending): %s",
            queue_len, exc,
        )
        return False, 0, 0
    except httpx.HTTPStatusError as exc:
        log.warning("eval /health returned %s; deferring queue", exc.response.status_code)
        return False, 0, 0
    except Exception as exc:
        log.warning("eval disk health check failed (proceeding): %s", exc)
        return True, 0, 0


async def _startup_prune_cache(http: httpx.AsyncClient, state: "State") -> None:

    keep: list[dict] = []

    if state.king.get("model_repo") and state.king.get("king_digest"):
        keep.append({"repo": state.king["model_repo"], "digest": state.king["king_digest"]})

    for e in state.king_chain:
        if e.get("model_repo") and e.get("king_digest"):
            keep.append({"repo": e["model_repo"], "digest": e["king_digest"]})

    for h in state.history[-5:]:
        if h.get("model_repo") and h.get("model_digest"):
            keep.append({"repo": h["model_repo"], "digest": h["model_digest"]})

    for q in state.queue:
        if q.get("model_repo") and q.get("model_digest"):
            keep.append({"repo": q["model_repo"], "digest": q["model_digest"]})

    if not keep:
        log.info("startup cache prune: no models in state, skipping")
        return
    try:
        r = await http.post(
            f"{EVAL_SERVER_URL}/prune_cache",
            json={"keep": keep},
            timeout=120.0,
        )
        r.raise_for_status()
        result = r.json()
        log.info(
            "startup cache prune: freed %.3f GB, kept %d models",
            result.get("freed_gb", 0), result.get("kept", 0),
        )
    except Exception as exc:
        log.warning("startup cache prune failed (non-fatal): %s", exc)


async def _eval_set_king(http: httpx.AsyncClient, king: dict) -> None:
    r = await http.post(
        f"{EVAL_SERVER_URL}/set_king",
        json={"king": {"repo": king["model_repo"], "digest": king["king_digest"]}},
        timeout=600.0,
    )
    r.raise_for_status()
    log.info("eval /set_king ok: %s", r.json().get("king"))


async def process_challenge(state: State, http: httpx.AsyncClient,
                             entry: dict, subtensor, wallet) -> None:
    cid = entry["challenge_id"]
    challenger = {"repo": entry["model_repo"], "digest": entry["model_digest"]}
    king = state.king
    if not king:
        log.error("%s: no king set; skipping", cid)
        state.record_failure(entry, "no_king", "validator king is empty")
        state.current_eval = None
        state.flush()
        state.flush_dashboard(force=True)
        return

    # Cheap pre-eval gate: config-only fetch + arch/lock checks + repo
    # hygiene. Catches malformed submissions before we spend a vLLM
    # bring-up cycle on them.
    state.flush_dashboard(force=True)
    rejection = await asyncio.to_thread(
        validate_challenger_config,
        entry["model_repo"], entry["model_digest"],
        king["model_repo"], king["king_digest"],
    )
    if rejection:
        log.info("%s: rejected at config gate: %s", cid, rejection)
        state.record_failure(entry, "config_mismatch", rejection)
        state.maybe_retry(entry, "config_mismatch", rejection)
        state.current_eval = None
        state.flush()
        state.flush_dashboard(force=True)
        return

    seed = _compute_seed(subtensor, entry.get("hotkey", ""))
    req_body = {
        "king": {"repo": king["model_repo"], "digest": king["king_digest"]},
        "challenger": challenger,
        "seed_hex": seed.hex(),
        "eval_id": cid,
        "hotkey": entry.get("hotkey", ""),
        "n_samples": chain_config.DUEL_N_SAMPLES,
        "max_turns": chain_config.DUEL_MAX_TURNS_PER_SAMPLE,
        # Keep-list for post-duel cache pruning on the eval server.
        # Keeps kings + last 5 evaluated challengers + still-queued challengers.
        "king_chain": [
            {"repo": e["model_repo"], "digest": e["king_digest"]}
            for e in state.king_chain
            if e.get("model_repo") and e.get("king_digest")
        ],
        "recent_challengers": [
            {"repo": h["model_repo"], "digest": h["model_digest"]}
            for h in state.history[-5:]
            if h.get("model_repo") and h.get("model_digest")
        ],
        "queued_challengers": [
            {"repo": q["model_repo"], "digest": q["model_digest"]}
            for q in state.queue          # current entry already popped before this call
            if q.get("model_repo") and q.get("model_digest")
        ],
    }

    state.current_eval = {
        "challenge_id": cid,
        "challenger_repo": entry.get("model_repo", ""),
        "challenger_digest": entry.get("model_digest", ""),
        "hotkey": entry.get("hotkey", ""),
        "uid": state.uid_map.get(entry.get("hotkey", "")),
        "started_at": _now(),
        "phase": "dispatching",
        "n_done": 0,
        "n_total": chain_config.DUEL_N_SAMPLES * chain_config.DUEL_MAX_TURNS_PER_SAMPLE,
        "king_mean": 0.0,
        "chal_mean": 0.0,
        "mean_delta": 0.0,
        "verdicts_king": {},
        "verdicts_chal": {},
        "parse_failures": 0,
    }
    state.flush_dashboard(force=True)

    verdict: dict | None = None
    last_event_at = _monotonic_now()
    try:
        async with http.stream("POST", f"{EVAL_SERVER_URL}/eval", json=req_body,
                                timeout=httpx.Timeout(None, connect=30.0)) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                log.error("%s: eval server %s: %s", cid, resp.status_code, err[:300])
                _detail = f"{resp.status_code}: {err[:300].decode(errors='ignore')}"
                state.record_failure(entry, "eval_http", _detail)
                state.maybe_retry(entry, "eval_http", _detail)
                state.current_eval = None
                state.flush()
                state.flush_dashboard(force=True)
                return

            cur_event = ""
            async for raw_line in resp.aiter_lines():
                last_event_at = _monotonic_now()
                if raw_line.startswith("event:"):
                    cur_event = raw_line.split(":", 1)[1].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line.split(":", 1)[1].strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except Exception:
                    continue

                if cur_event == "phase":
                    state.current_eval["phase"] = data.get("phase", state.current_eval["phase"])
                    if "n_turns_total" in data:
                        state.current_eval["n_total"] = data["n_turns_total"]
                    state.flush_dashboard()
                elif cur_event == "progress":
                    state.current_eval.update({
                        "phase": "duel",
                        "n_done": data.get("n_done", 0),
                        "n_total": data.get("n_total", state.current_eval["n_total"]),
                        "king_mean": data.get("king_mean", 0.0),
                        "chal_mean": data.get("chal_mean", 0.0),
                        "mean_delta": data.get("mean_delta", 0.0),
                        "verdicts_king": data.get("verdicts_king", {}),
                        "verdicts_chal": data.get("verdicts_chal", {}),
                        "parse_failures": data.get("parse_failures", 0),
                        "last": data.get("last"),
                    })
                    state.flush_dashboard()
                elif cur_event == "heartbeat":
                    state.flush_dashboard()
                elif cur_event == "verdict":
                    verdict = data
                    break

                if _monotonic_now() - last_event_at > STREAM_IDLE_KILL_S:
                    raise TimeoutError(f"eval stream idle > {STREAM_IDLE_KILL_S}s")
                if _monotonic_now() - last_event_at > STREAM_IDLE_WARN_S:
                    log.warning("%s: eval stream idle > %ds", cid, STREAM_IDLE_WARN_S)

    except Exception as exc:
        err_str = str(exc).lower()
        conn_err = isinstance(
            exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)
        ) or "connection attempts failed" in err_str or "connecterror" in err_str
        if conn_err:
            state.eval_box_consecutive_fails += 1
            backoff = min(
                EVAL_BOX_BACKOFF_S * (2 ** (state.eval_box_consecutive_fails - 1)),
                EVAL_BOX_BACKOFF_MAX_S,
            )
            state.eval_box_retry_after = _monotonic_now() + backoff
            log.warning(
                "%s: eval unreachable (%s); re-queuing with %ds backoff "
                "(consecutive fails: %d)",
                cid, exc, backoff, state.eval_box_consecutive_fails,
            )
            state.record_failure(entry, "eval_infra", f"eval_unreachable: {exc}")
            state.queue.insert(0, entry)
            state.current_eval = None
            state.flush()
            state.flush_dashboard(force=True)
            return
        log.exception("%s: eval failed", cid)
        _detail = str(exc)
        state.record_failure(entry, "eval_error", _detail)
        state.maybe_retry(entry, "eval_error", _detail)
        state.current_eval = None
        state.flush()
        state.flush_dashboard(force=True)
        return

    # Reaching here means the SSE stream connected and ran — eval box is up.
    state.eval_box_consecutive_fails = 0
    state.eval_box_retry_after = 0.0

    if not verdict:
        log.error("%s: eval stream ended without verdict", cid)
        state.record_failure(entry, "no_verdict", "stream closed without verdict")
        state.maybe_retry(entry, "no_verdict", "stream closed without verdict")
        state.current_eval = None
        state.flush()
        state.flush_dashboard(force=True)
        return

    if _is_infra_failure(verdict):
        err = str(verdict.get("error") or "zero_turn_eval")
        log.warning("%s: infra failure (%s)", cid, err[:120])
        if "disk_full" in err and entry.get("hotkey"):
            state.infra_cooldown[entry["hotkey"]] = _monotonic_now() + 300.0
        state.record_failure(entry, "eval_infra", err)
        state.maybe_retry(entry, "eval_infra", err)
    else:
        state.record_verdict(entry, verdict)
        if verdict.get("accepted"):
            log.info("%s: ACCEPTED. crowning %s", cid, entry.get("hotkey", "?")[:16])
            try:
                block = subtensor.block
            except Exception:
                block = 0
            state.set_king(entry.get("hotkey", ""), entry.get("model_repo", ""),
                           entry.get("model_digest", ""), block, challenge_id=cid,
                           dethrone_judges=[
                               {"model": j["model"], "king_mean": j["king_mean"], "n": j.get("n", 0)}
                               for j in (verdict.get("judges") or [])
                           ],
                           crown_judges=[
                               {"model": j["model"], "king_mean": j["chal_mean"], "n": j.get("n", 0)}
                               for j in (verdict.get("judges") or [])
                           ])
            try:
                await _eval_set_king(http, state.king)
            except Exception:
                log.exception("post-dethrone /set_king failed; will retry on next tick")
            await maybe_set_weights(subtensor, wallet, state, force=True, reason="dethrone")
        else:
            deth = verdict.get("dethrone") or {}
            log.info("%s: REJECTED. dethrone=%s mean_delta=%.4f lcb=%.4f", cid,
                     deth, verdict.get("mean_delta", 0.0), verdict.get("lcb_at_1_minus_alpha", 0.0))
    state.current_eval = None
    state.flush()
    state.flush_dashboard(force=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> int:
    if NETUID == 0:
        log.error("set ALBEDO_NETUID to the actual subnet netuid before starting")
        return 1
    if not SEED_DIGEST:
        log.error("set ALBEDO_SEED_DIGEST (or fill chain.toml [seed].seed_digest)")
        return 1

    store = ObjectStore()
    state = State(store)
    state.load()

    wallet = bt.Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.Subtensor(network=NETWORK)

    # The load-bearing CR gate. Without commit-reveal, set_weights silently
    # downgrades to set_mechanism_weights and parallel validators can copy
    # weights. Refuse to start.
    if REQUIRE_COMMIT_REVEAL and not subtensor.commit_reveal_enabled(NETUID):
        log.error("commit-reveal NOT enabled on netuid %d. "
                  "Have the subnet owner enable CR before starting.", NETUID)
        return 2
    if not REQUIRE_COMMIT_REVEAL and not subtensor.commit_reveal_enabled(NETUID):
        log.warning("commit-reveal disabled on netuid %d; using plain set_weights "
                    "(ALBEDO_REQUIRE_COMMIT_REVEAL=0)", NETUID)

    state.refresh_uid_map(subtensor, NETUID)

    # Stamp + upload the website with the new build id so long-lived tabs
    # auto-reload after deploy.
    html_path = os.path.join(os.path.dirname(__file__) or ".", "website", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        build_id = hashlib.sha256(html_bytes).hexdigest()[:12]
        html_bytes = html_bytes.replace(b"__BUILD_ID__", build_id.encode())
        store.put_dashboard_raw(
            "index.html", html_bytes, "text/html; charset=utf-8",
            cache_control="no-cache, must-revalidate",
        )
        favicon_path = os.path.join(os.path.dirname(html_path), "favicon.svg")
        if os.path.exists(favicon_path):
            with open(favicon_path, "rb") as f:
                favicon_bytes = f.read()
            store.put_dashboard_raw(
                "favicon.svg", favicon_bytes, "image/svg+xml",
                cache_control="no-cache, must-revalidate",
            )
        log.info("uploaded website (build=%s)", build_id)
    state.flush_dashboard(force=True)

    if not state.king:
        try:
            seed_ref = ModelRef(SEED_REPO, SEED_DIGEST)
        except Exception as exc:
            log.error("invalid seed ref %s@%s: %s", SEED_REPO, SEED_DIGEST, exc)
            return 1
        # Sanity-fetch config so we fail early if the seed Hippius ref is bad.
        materialize_model(seed_ref, max_workers=4, config_only=True)
        state.set_king("", seed_ref.repo, seed_ref.digest,
                       subtensor.block, challenge_id="seed")

    # Bring up eval.py king once at startup (idempotent).
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as http:
        for attempt in range(3):
            try:
                await _eval_set_king(http, state.king)
                break
            except Exception as exc:
                log.warning("startup /set_king attempt %d failed: %s", attempt + 1, exc)
                await asyncio.sleep(10.0)
        else:
            log.error("eval server unreachable on startup; aborting")
            return 3

        await _startup_prune_cache(http, state)

        # Re-queue miners who hit infra-side failures within the lookback window.
        if REEVAL_LOOKBACK_HOURS > 0:
            n_recovered = _recover_from_lookback(state)
            if n_recovered:
                log.info("lookback recovery: %d miner(s) re-queued "
                         "(lookback=%.0fh)", n_recovered, REEVAL_LOOKBACK_HOURS)
                state.flush()
                state.flush_dashboard(force=True)

        # Re-queue hotkeys listed in to_reeval.json (operator-driven, one-shot).
        n_reeval = _reeval_from_file(subtensor, NETUID, state)
        if n_reeval:
            log.info("to_reeval: %d miner(s) re-queued", n_reeval)
            state.flush()
            state.flush_dashboard(force=True)

        await maybe_set_weights(subtensor, wallet, state, force=True, reason="startup")

        def _on_signal(sig, frame):
            log.info("signal %d -> shutdown", sig)
            sys.exit(0)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        log.info("validator running | king=%s@%s | netuid=%d | eval=%s",
                 state.king.get("model_repo", "?"),
                 (state.king.get("king_digest") or "")[:19],
                 NETUID, EVAL_SERVER_URL)

        while True:
            try:
                state.refresh_uid_map(subtensor, NETUID)
                reveals = scan_reveals(subtensor, NETUID, state.completed_repos, state.seen)
                now_mono = _monotonic_now()
                for rev in reveals:
                    hk = rev.get("hotkey", "")
                    if hk and now_mono < state.infra_cooldown.get(hk, 0):
                        continue
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s", cid, rev["hotkey"][:16])

                while state.queue:
                    _loop_mono = _monotonic_now()
                    # Honour eval-box backoff (set after eval_unreachable).
                    if _loop_mono < state.eval_box_retry_after:
                        remaining = int(state.eval_box_retry_after - _loop_mono)
                        if _loop_mono - state._last_disk_warn_monotonic > 60:
                            state._last_disk_warn_monotonic = _loop_mono
                            log.warning(
                                "eval box in backoff — %ds remaining "
                                "(consecutive fails: %d); deferring queue (%d pending)",
                                remaining, state.eval_box_consecutive_fails, len(state.queue),
                            )
                        break
                    disk_ok, free_b, min_b = await _eval_disk_ok(
                        http, queue_len=len(state.queue),
                    )
                    if not disk_ok:
                        if _loop_mono - state._last_disk_warn_monotonic > 60:
                            state._last_disk_warn_monotonic = _loop_mono
                            log.warning(
                                "eval disk low (%d bytes free, need %d); "
                                "deferring queue (%d pending)",
                                free_b, min_b, len(state.queue),
                            )
                        break
                    entry = state.queue.pop(0)
                    # Pre-populate current_eval before flush_dashboard so the
                    # queue section never shows a gap between "popped" and "in
                    # flight". process_challenge overwrites this with full details.
                    state.current_eval = {
                        "challenge_id":    entry.get("challenge_id"),
                        "challenger_repo": entry.get("model_repo", ""),
                        "challenger_digest": entry.get("model_digest", ""),
                        "hotkey":          entry.get("hotkey", ""),
                        "uid":             state.uid_map.get(entry.get("hotkey", "")),
                        "queued_at":       entry.get("queued_at"),
                        "started_at":      _now(),
                        "phase":           "starting",
                    }
                    state.flush()
                    state.flush_dashboard(force=True)

                    async def _bounded() -> None:
                        await process_challenge(state, http, entry, subtensor, wallet)

                    try:
                        await asyncio.wait_for(_bounded(), timeout=TICK_RESTART_AFTER)
                    except asyncio.TimeoutError:
                        log.error("%s: hard wall-clock timeout (%ds)",
                                  entry.get("challenge_id"), TICK_RESTART_AFTER)
                        _detail = f"exceeded {TICK_RESTART_AFTER}s"
                        state.record_failure(entry, "hard_timeout", _detail)
                        state.maybe_retry(entry, "hard_timeout", _detail)
                        state.current_eval = None
                        state.flush()
                        state.flush_dashboard(force=True)
                    except asyncio.CancelledError:
                        raise

                    try:
                        await maybe_set_weights(subtensor, wallet, state, reason="in-queue")
                    except Exception:
                        log.exception("in-queue set_weights failed")

                state.current_eval = None
                state.flush_dashboard(force=True)
                try:
                    await maybe_set_weights(subtensor, wallet, state, reason="periodic")
                except Exception:
                    log.exception("periodic set_weights failed")
                state.watchdog["consecutive_tick_errors"] = 0
            except KeyboardInterrupt:
                return 0
            except Exception:
                log.exception("tick error")
                state.watchdog["consecutive_tick_errors"] += 1
                if state.watchdog["consecutive_tick_errors"] >= MAX_CONSECUTIVE_TICK_ERRORS:
                    log.error("too many consecutive tick errors -> exit")
                    return 4
            await asyncio.sleep(POLL_INTERVAL)


def main_sync() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(main_sync())
