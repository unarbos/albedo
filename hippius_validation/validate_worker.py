"""Hippius validation worker — claim queued commits oldest-first and validate each model.

Startup checks (DB + OpenSearch) → ensure schema → enqueue from chain_commits → loop:
sweep expired leases → claim oldest queued → run checks (with a heartbeat) → finalize.
"""
from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger as log

from config_validation.fingerprint import compute_fingerprint

from hippius_validation import config, db
from hippius_validation.hippius import download_full, list_files, make_ref
from hippius_validation.opensearch import find_duplicate, health, index_fingerprint
from hippius_validation.uploads import put_fault, update_fingerprint_corpus
from hippius_validation.validate import check_architecture, check_repo

_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class Outcome:
    state: str                       # 'done' | 'failed'
    fault_class: str | None = None   # MINER_FAULT | INFRA_FAULT
    fault_code: str | None = None
    fault_message: str = ""
    retryable: bool = False
    result_summary: dict = field(default_factory=dict)
    # Full explanation written to fault.json (e.g. duplicate evidence incl. the fingerprint).
    fault_detail: dict = field(default_factory=dict)


def _miner(code: str, msg: str, summary: dict, fault_detail: dict | None = None) -> Outcome:
    return Outcome("failed", "MINER_FAULT", code, msg, False, summary, fault_detail or {})


def _infra(code: str, msg: str) -> Outcome:
    return Outcome("failed", "INFRA_FAULT", code, msg, True, {})


_NOT_FOUND_MARKERS = ("not found", "404", "no such", "does not exist", "nosuchkey",
                      "no revision", "not exist", "norepo")


def _is_not_found(exc: Exception) -> bool:
    """True if a Hippius error means the repo/revision simply doesn't exist (miner fault)."""
    return any(m in str(exc).lower() for m in _NOT_FOUND_MARKERS)


def process_model(model_uri: str, hotkey: str) -> Outcome:
    """Run the per-model check flow synchronously (blocking I/O). Returns an Outcome."""
    repo, _, digest = model_uri.partition("@")
    ref = make_ref(repo, digest)

    # 1 — file manifest
    try:
        files = list_files(ref)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        return _infra("list_files_failed", f"could not list repo files: {exc}")
    # An empty repo is the miner's fault, not infra.
    if not files:
        return _miner("empty_repo", "Hippius repo has no files", {})
    ok, msg = check_repo(files)
    if not ok:
        return _miner("file_manifest", msg, {"files": sorted(files)[:50]})

    # 2 — download
    try:
        model_dir = download_full(ref)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        return _infra("download_failed", f"model download failed: {exc}")
    # Repo that resolved but yielded no usable model content is a miner fault, not infra.
    mdir = Path(model_dir)
    if not (mdir / "config.json").exists() or not any(mdir.glob("*.safetensors")):
        return _miner("incomplete_repo",
                      "downloaded repo is missing config.json or *.safetensors", {})

    # 3 — universal, spec-driven architecture
    try:
        ok, msg = check_architecture(model_dir)
    except FileNotFoundError as exc:
        return _miner("architecture", f"config.json missing: {exc}", {})
    except Exception as exc:  # noqa: BLE001
        return _infra("architecture_read_failed", f"could not read config.json: {exc}")
    if not ok:
        return _miner("architecture", msg, {})

    # 4 — fingerprint + dedup
    try:
        fp = compute_fingerprint(model_dir)
    except Exception as exc:  # noqa: BLE001
        return _infra("fingerprint_failed", f"could not fingerprint model: {exc}")

    # Record this model's fingerprint into the two aggregate corpus files (all fingerprinted models).
    fp_uri, tensors_uri = update_fingerprint_corpus(model_uri, fp)

    try:
        dedup = find_duplicate(fp, hotkey)
    except Exception as exc:  # noqa: BLE001
        return _infra("opensearch_failed", f"dedup search failed: {exc}")

    if dedup["is_duplicate"]:
        reason = (f"duplicate of {dedup['matched_key']}: similarity "
                  f"{dedup['similarity']:.6f} >= {dedup['threshold']} threshold")
        summary = {"reason": reason, "similarity": dedup["similarity"], "threshold": dedup["threshold"],
                   "duplicate_of": dedup["matched_key"], "duplicate_of_hotkey": dedup["matched_hotkey"],
                   "candidates_checked": dedup["candidates_checked"]}
        # The full duplicate explanation + fingerprint evidence goes into fault.json.
        fault_detail = {**summary, "fingerprint": fp}
        return _miner("duplicate", reason, summary, fault_detail=fault_detail)

    # Not a duplicate → index it into the working corpus (OpenSearch).
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        index_fingerprint(model_uri, fp, hotkey=hotkey, repo=repo, digest=digest,
                          model_uri=model_uri, created_at=created_at)
    except Exception as exc:  # noqa: BLE001
        return _infra("opensearch_index_failed", f"could not index fingerprint: {exc}")

    return Outcome("done", result_summary={"similarity": dedup["similarity"], "threshold": dedup["threshold"],
                                           "fingerprint_file": fp_uri, "tensors_file": tensors_uri,
                                           "n_tensors": len(fp.get("layer_keys", []))})


async def _heartbeat_loop(pool, attempt_id) -> None:
    while True:
        await asyncio.sleep(config.HEARTBEAT_S)
        await db.heartbeat(pool, attempt_id, config.LEASE_SECONDS)


async def _finalize(pool, attempt, outcome: Outcome) -> None:
    if outcome.state == "done":
        await db.mark_done(pool, attempt["id"], outcome.result_summary)
        log.info("done — {}", attempt["model_uri"])
    elif outcome.retryable:
        new_state = await db.mark_retry(
            pool, attempt["id"], attempt_number=attempt["attempt_number"],
            max_attempts=config.MAX_ATTEMPTS, fault_class=outcome.fault_class,
            fault_code=outcome.fault_code, fault_message=outcome.fault_message)
        log.warning("infra fault [{}] {} → {}", outcome.fault_code, attempt["model_uri"], new_state)
    else:
        # Terminal miner fault: publish a full-explanation fault.json to Hippius (best-effort).
        # For a duplicate, fault_detail carries the matched model + similarity + fingerprint evidence.
        digest = attempt["model_uri"].partition("@")[2]
        fault_doc = {
            "model_uri": attempt["model_uri"], "hotkey": attempt["hotkey"],
            "block_number": attempt["block_number"], "fault_class": outcome.fault_class,
            "fault_code": outcome.fault_code, "fault_message": outcome.fault_message,
            **(outcome.fault_detail or {"details": outcome.result_summary}),
        }
        fault_uri = await asyncio.to_thread(put_fault, attempt["hotkey"], digest, fault_doc)
        summary = {**outcome.result_summary, "fault_uri": fault_uri}
        await db.mark_failed(pool, attempt["id"], fault_class=outcome.fault_class,
                             fault_code=outcome.fault_code, fault_message=outcome.fault_message,
                             result_summary=summary)
        log.warning("miner fault [{}] {} — {}", outcome.fault_code, attempt["model_uri"],
                    outcome.fault_message)


async def run() -> None:
    pool = await db.connect(config.DB_URL)
    if not health():
        raise RuntimeError(f"OpenSearch not healthy at {config.OPENSEARCH_URL}")

    log.info("hippius_validation started — worker={} netuid={}", _WORKER_ID, config.NETUID)
    n = await db.enqueue_from_commits(pool, config.NETUID)
    log.info("enqueued {} new commit(s)", n)

    try:
        while True:
            await db.sweep_expired(pool)
            attempt = await db.claim_next(pool, _WORKER_ID, config.LEASE_SECONDS)
            if attempt is None:
                await db.enqueue_from_commits(pool, config.NETUID)
                await asyncio.sleep(config.POLL_INTERVAL_S)
                continue

            log.info("claim — block={} hotkey={} {}", attempt["block_number"],
                     attempt["hotkey"][:10], attempt["model_uri"])

            # success_validated gate (read-only): one evaluation per hotkey.
            if await db.hotkey_validated(pool, attempt["hotkey"]):
                await db.mark_done(pool, attempt["id"],
                                   {"skipped": "hotkey_already_evaluated"})
                log.info("skip — hotkey already evaluated: {}", attempt["hotkey"][:10])
                continue

            hb = asyncio.create_task(_heartbeat_loop(pool, attempt["id"]))
            try:
                outcome = await asyncio.to_thread(
                    process_model, attempt["model_uri"], attempt["hotkey"])
            except Exception as exc:  # noqa: BLE001 — unexpected; treat as infra, retry
                outcome = _infra("unexpected", f"{type(exc).__name__}: {exc}")
            finally:
                hb.cancel()
            await _finalize(pool, attempt, outcome)
    finally:
        await pool.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("hippius_validation stopped")


if __name__ == "__main__":
    main()
