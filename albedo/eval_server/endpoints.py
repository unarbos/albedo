"""albedo.eval_server.endpoints — FastAPI application with all eval-server routes."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import hashlib

from albedo.config import DATASET_MANIFEST_SHA256, DUEL_MAX_TURNS, DUEL_N_SAMPLES
from albedo.duel import TrajectoryDataset, run_duel
from albedo.judge import ChutesJudge
from albedo.models import ModelRef, materialize_model, prune_model_cache
from albedo.preeval import add_fingerprint, check_fingerprint, probe_injection
from albedo.eval_server.server_state import STATE, CHAL_PORT, KING_PORT
from albedo.eval_server.vllm import reclaim_stray_on_ports
from albedo.eval_server.sink import DatasetSink
from albedo.eval_server.fingerprint_store import load_fingerprints, save_fingerprints
from albedo.eval_server.notifications import notify_problem
from albedo.eval_server.probe_logs import write_probe_artifacts
from albedo.eval_server.logging_setup import setup_logging

setup_logging()
log = logging.getLogger(__name__)

app = FastAPI(title="Albedo Eval Server")

_DATASET_DIR = os.environ.get("ALBEDO_DATASET_DIR", "/root/albedo/dataset")
_MIN_DISK_BYTES = int(os.environ.get("ALBEDO_MIN_DISK_BYTES", str(50 * 1024 ** 3)))  # 50 GB
# Concurrent judge calls in flight per duel (Chutes rate-limits beyond ~16).
_MAX_PARALLEL_TURNS = int(os.environ.get("ALBEDO_MAX_PARALLEL_TURNS", "8"))
# Hard ceiling on model download time — prevents a stalled Hippius hub from
# hanging the eval server indefinitely during materialize_model().
_MODEL_DOWNLOAD_TIMEOUT = float(os.environ.get("ALBEDO_MODEL_DOWNLOAD_TIMEOUT", "3600"))


def _short_ref(value: str, limit: int = 48) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit - 3] + "..."


def _eval_log(eval_id: str, phase: str, message: str, *args: object) -> None:
    log.info("[%s][%s] " + message, eval_id, phase, *args)


async def _add_and_persist(key: str, model_dir: str, hotkey: str = "") -> None:
    """Fingerprint a model into shared state and persist the state to S3 (best-effort).

    hotkey is recorded so a miner is never flagged as a duplicate of their own model.
    """
    try:
        await asyncio.to_thread(add_fingerprint, key, model_dir, STATE.fingerprints, hotkey)
        await asyncio.to_thread(save_fingerprints, STATE.fingerprints)
    except Exception:
        log.warning("fingerprint add/persist failed for %r", key, exc_info=True)


@app.on_event("startup")
async def _on_startup() -> None:
    """Reclaim orphaned vLLM from a hard restart, then load fingerprint state."""
    # If the previous eval server was SIGKILLed, its detached king/challenger vLLM
    # procs are still holding GPU memory — free them before we try to start ours.
    await asyncio.to_thread(reclaim_stray_on_ports, [KING_PORT, CHAL_PORT])

    loaded = await asyncio.to_thread(load_fingerprints)
    if loaded:
        STATE.fingerprints.update(loaded)
        log.info("startup: loaded %d persisted fingerprints", len(loaded))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Stop both vLLM subprocesses cleanly."""
    STATE.king_proc.stop()
    STATE.chal_proc.stop()
    log.info("shutdown: stopped king and challenger vLLM processes")


class EvalRequest(BaseModel):
    king:               dict
    challenger:         dict
    seed_hex:           str
    eval_id:            str
    hotkey:             str | None = None
    n_samples:          int | None = Field(None, ge=1, le=512)
    max_turns:          int | None = Field(None, ge=1, le=100)
    king_chain:         list[dict] = []
    recent_challengers: list[dict] = []
    queued_challengers: list[dict] = []

    @field_validator("seed_hex")
    @classmethod
    def _validate_hex(cls, v: str) -> str:
        bytes.fromhex(v)
        return v


def _dataset_info() -> dict[str, Any]:
    """Return basic dataset stats from manifest without loading parquet."""
    root = Path(_DATASET_DIR)
    manifest = root / "manifest.json"
    if not manifest.exists():
        return {"exists": False, "shards": 0, "total_rows": 0}
    try:
        import json
        data = json.loads(manifest.read_text())
        shards = data.get("shards", [])
        total_rows = data.get("total_rows", sum(s.get("rows", 0) for s in shards))
        return {"exists": True, "shards": len(shards), "total_rows": total_rows}
    except Exception:
        return {"exists": True, "shards": 0, "total_rows": 0}


def _disk_info() -> dict[str, Any]:
    usage = shutil.disk_usage("/")
    return {"free_bytes": usage.free, "min_required_bytes": _MIN_DISK_BYTES}


@app.get("/health")
async def health() -> JSONResponse:
    """Return server and subprocess health."""
    king = STATE.king_proc
    chal = STATE.chal_proc
    uptime = king.uptime_s
    return JSONResponse({
        "ok": True,
        "king": {
            "alive":      king.is_alive(),
            "model_name": king.model_name,
            "port":       KING_PORT,
            "uptime_s":   uptime if uptime is not None else 0.0,
        },
        "challenger": {
            "alive":      chal.is_alive(),
            "model_name": chal.model_name,
            "port":       CHAL_PORT,
        },
        "eval_lock_held":   STATE.eval_lock.locked(),
        "current_eval_id":  STATE.current_eval_id,
        "disk":             _disk_info(),
        "dataset":          _dataset_info(),
    })


@app.post("/set_king")
async def set_king(body: dict) -> JSONResponse:
    """Download and start a new king model; body: {"king": {"repo": str, "digest": str}}."""
    king_dict = body.get("king", {})
    try:
        ref = ModelRef(repo=king_dict["repo"], digest=king_dict["digest"])
    except (KeyError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)

    if STATE.king_proc.model_name == ref.immutable_ref and STATE.king_proc.is_alive():
        log.info("set_king: already running %r — no-op", ref.immutable_ref)
        return JSONResponse({"ok": True, "started": False, "model": ref.immutable_ref})

    log.info("[set_king][materialize] materializing king model %s", _short_ref(ref.immutable_ref))
    try:
        model_dir = await asyncio.wait_for(
            asyncio.to_thread(materialize_model, ref),
            timeout=_MODEL_DOWNLOAD_TIMEOUT,
        )
    except Exception as exc:
        await notify_problem(
            title="set_king materialize failed",
            message=f"Failed to materialize king model `{ref.immutable_ref}`.",
            dedupe_key=f"set_king:materialize:{ref.immutable_ref}:{type(exc).__name__}:{str(exc)[:120]}",
            details=str(exc),
        )
        raise
    log.info("[set_king][materialize] materialized king model %s at %s", _short_ref(ref.immutable_ref), model_dir)

    if STATE.eval_lock.locked():
        return JSONResponse(
            {"ok": False, "error": "eval in progress — try again after it completes"},
            status_code=409,
        )

    async with STATE.eval_lock:
        try:
            log.info("[set_king][startup] starting king vLLM for %s", _short_ref(ref.immutable_ref))
            await STATE.king_proc.start(model_dir, ref.immutable_ref)
            await STATE.king_proc.wait_healthy()
        except Exception as exc:
            await notify_problem(
                title="set_king startup failed",
                message=f"King vLLM failed to start for `{ref.immutable_ref}`.",
                dedupe_key=f"set_king:start:{ref.immutable_ref}:{type(exc).__name__}:{str(exc)[:120]}",
                details=str(exc),
            )
            raise
    log.info("[set_king][startup] king vLLM healthy for %s", _short_ref(ref.immutable_ref))

    # Fingerprint + persist in the background to avoid blocking the response
    asyncio.create_task(_add_and_persist(ref.immutable_ref, model_dir))

    return JSONResponse({"ok": True, "started": True, "model": ref.immutable_ref})


@app.post("/eval")
async def eval_endpoint(req: EvalRequest) -> StreamingResponse:
    """Run a full duel and stream SSE events; returns 409 if already in progress."""
    if STATE.eval_lock.locked():
        return JSONResponse(
            {"ok": False, "error": "eval already in progress", "eval_id": STATE.current_eval_id},
            status_code=409,
        )

    try:
        king_ref = ModelRef(repo=req.king["repo"], digest=req.king["digest"])
        chal_ref = ModelRef(repo=req.challenger["repo"], digest=req.challenger["digest"])
    except (KeyError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)

    seed = bytes.fromhex(req.seed_hex)
    n_samples = req.n_samples or DUEL_N_SAMPLES
    max_turns = req.max_turns or DUEL_MAX_TURNS
    hotkey = req.hotkey or ""
    _eval_log(
        req.eval_id,
        "request",
        "received hotkey=%s king=%s challenger=%s n_samples=%d max_turns=%d",
        hotkey or "unknown",
        _short_ref(king_ref.immutable_ref),
        _short_ref(chal_ref.immutable_ref),
        n_samples,
        max_turns,
    )

    async def _stream() -> Any:
        import json

        def _sse(event: str, data: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        async with STATE.eval_lock:
            STATE.current_eval_id = req.eval_id
            sink = DatasetSink(
                eval_id=req.eval_id,
                challenger_hotkey=hotkey,
                king_hotkey=req.king.get("hotkey", "") if req.king else "",
            )
            try:
                _eval_log(req.eval_id, "pre-eval", "lock acquired; starting pre-duel gates")
                # Keepalive: these pre-duel phases emit no 'start'/'turn' events, so feed the
                # validator's idle timer (it resets on any line) while we materialise + probe.
                yield _sse("phase", {"eval_id": req.eval_id, "phase": "materialize"})
                # Gate 1: materialise challenger and start vLLM
                try:
                    _eval_log(req.eval_id, "pre-eval:materialize", "materializing challenger %s", _short_ref(chal_ref.immutable_ref))
                    chal_dir = await asyncio.wait_for(
                        asyncio.to_thread(materialize_model, chal_ref),
                        timeout=_MODEL_DOWNLOAD_TIMEOUT,
                    )
                    _eval_log(req.eval_id, "pre-eval:materialize", "challenger materialized at %s; starting vLLM", chal_dir)
                    await STATE.chal_proc.start(chal_dir, chal_ref.immutable_ref)
                    await STATE.chal_proc.wait_healthy()
                    _eval_log(req.eval_id, "pre-eval:challenger", "challenger vLLM healthy")
                except Exception as exc:
                    log.error("challenger vLLM failed to start: %s", exc)
                    await notify_problem(
                        title="challenger startup failed",
                        message=f"Challenger vLLM failed to start for `{chal_ref.immutable_ref}`.",
                        dedupe_key=(
                            f"eval:chal_start:{chal_ref.immutable_ref}:{type(exc).__name__}:{str(exc)[:120]}"
                        ),
                        details=str(exc),
                    )
                    yield _sse("verdict", {
                        "eval_id": req.eval_id, "accepted": False,
                        "error": f"chal_vllm_start_failed: {exc}",
                    })
                    return

                # Gate 2: near-duplicate fingerprint check
                _eval_log(req.eval_id, "pre-eval:fingerprint", "running near-duplicate fingerprint check")
                is_dup, dup_key = await asyncio.to_thread(
                    check_fingerprint, chal_dir, STATE.fingerprints, None, hotkey
                )
                if is_dup:
                    log.warning("challenger %s is near-duplicate of %s",
                                chal_ref.immutable_ref[:40], dup_key[:40])
                    yield _sse("verdict", {
                        "eval_id": req.eval_id, "accepted": False,
                        "is_duplicate": True, "duplicate_of": dup_key,
                        "error": f"duplicate_model: too similar to {dup_key}",
                    })
                    return
                _eval_log(req.eval_id, "pre-eval:fingerprint", "fingerprint check passed")

                # Gate 3: injection probe — probe seed is distinct from duel seed
                yield _sse("phase", {"eval_id": req.eval_id, "phase": "probe"})
                _eval_log(req.eval_id, "pre-eval:probe", "starting injection probe")
                probe = await probe_injection(
                    challenger_url=f"http://localhost:{CHAL_PORT}",
                    eval_id=req.eval_id,
                    dataset_dir=_DATASET_DIR,
                )
                await write_probe_artifacts(
                    eval_id=req.eval_id,
                    challenger_ref=chal_ref.immutable_ref,
                    challenger_hotkey=hotkey,
                    probe_result=probe,
                )
                _eval_log(
                    req.eval_id,
                    "pre-eval:probe",
                    "finished clean=%s probes=%d injections=%d untested=%d",
                    probe.is_clean,
                    probe.n_probes,
                    probe.n_injections,
                    probe.n_untested,
                )
                if not probe.is_clean:
                    log.warning("injection detected in %s: triggered_judges=%s",
                                chal_ref.immutable_ref[:40], probe.triggered_judges)
                    yield _sse("verdict", {
                        "eval_id": req.eval_id, "accepted": False,
                        "error": (
                            f"chal_injection_detected: injection_finetuned: "
                            f"{probe.n_injections}/{probe.n_probes} probe turns "
                            f"triggered ({', '.join(probe.triggered_judges)})"
                        ),
                    })
                    return

                if not await STATE.king_proc.is_healthy():
                    log.warning(
                        "king vLLM not healthy at eval %s — restarting %s",
                        req.eval_id, _short_ref(king_ref.immutable_ref),
                    )
                    _eval_log(req.eval_id, "pre-eval:king", "king not healthy; restarting %s",
                              _short_ref(king_ref.immutable_ref))
                    yield _sse("phase", {"eval_id": req.eval_id, "phase": "king_restart"})
                    try:
                        king_dir = await asyncio.wait_for(
                            asyncio.to_thread(materialize_model, king_ref),
                            timeout=_MODEL_DOWNLOAD_TIMEOUT,
                        )
                        await STATE.king_proc.start(king_dir, king_ref.immutable_ref)
                        await STATE.king_proc.wait_healthy()
                        _eval_log(req.eval_id, "pre-eval:king", "king vLLM healthy after restart")
                    except Exception as exc:
                        log.error("king vLLM restart failed: %s", exc)
                        await notify_problem(
                            title="king restart failed",
                            message=(
                                f"King vLLM unavailable for eval `{req.eval_id}` "
                                f"(`{king_ref.immutable_ref}`)."
                            ),
                            dedupe_key=(
                                f"eval:king_restart:{king_ref.immutable_ref}:"
                                f"{type(exc).__name__}:{str(exc)[:120]}"
                            ),
                            details=str(exc),
                        )
                        yield _sse("verdict", {
                            "eval_id": req.eval_id, "accepted": False,
                            "error": f"king_unavailable: {exc}",
                        })
                        return

                # Duel
                manifest_path = Path(_DATASET_DIR) / "manifest.json"
                actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if actual_sha != DATASET_MANIFEST_SHA256:
                    raise RuntimeError(
                        f"dataset manifest mismatch: {actual_sha} != {DATASET_MANIFEST_SHA256}"
                    )
                _eval_log(req.eval_id, "pre-eval:dataset", "dataset manifest verified")
                dataset = await asyncio.to_thread(
                    lambda: TrajectoryDataset(_DATASET_DIR).sample(seed, n_samples, max_turns)
                )
                _eval_log(
                    req.eval_id,
                    "pre-eval:dataset",
                    "sampled dataset turns=%d (requested n_samples=%d max_turns=%d)",
                    len(dataset),
                    n_samples,
                    max_turns,
                )

                from albedo.config import JUDGE_MODELS
                judge = ChutesJudge()
                _eval_log(
                    req.eval_id,
                    "eval",
                    "EVAL START — pre-eval gates passed; duel begins king=%s challenger=%s "
                    "n_samples=%d max_turns=%d judges=%d (%s)",
                    _short_ref(king_ref.immutable_ref),
                    _short_ref(chal_ref.immutable_ref),
                    n_samples,
                    max_turns,
                    len(JUDGE_MODELS),
                    ", ".join(JUDGE_MODELS),
                )
                async for chunk in run_duel(
                    samples=dataset,
                    king_client=STATE.king_proc.client,
                    chal_client=STATE.chal_proc.client,
                    judge=judge,
                    judge_models=JUDGE_MODELS,
                    seed=seed,
                    eval_id=req.eval_id,
                    hotkey=hotkey,
                    max_parallel=_MAX_PARALLEL_TURNS,
                    sink=sink,
                ):
                    yield chunk
                _eval_log(req.eval_id, "duel", "duel stream completed")

                # Store + persist challenger fingerprint (with its hotkey) for future dup checks.
                # Awaited before yielding the verdict to ensure fingerprint is synced.
                _eval_log(req.eval_id, "persist", "persisting challenger fingerprint")
                await _add_and_persist(chal_ref.immutable_ref, chal_dir, hotkey)
                _eval_log(req.eval_id, "persist", "challenger fingerprint persisted")

            except Exception as exc:
                log.exception("eval %s failed", req.eval_id)
                await notify_problem(
                    title="eval request failed",
                    message=(
                        f"Eval `{req.eval_id}` failed for challenger `{chal_ref.immutable_ref}` "
                        f"and hotkey `{hotkey or 'unknown'}`."
                    ),
                    dedupe_key=f"eval:failure:{type(exc).__name__}:{str(exc)[:160]}",
                    details=str(exc),
                )
                raise
            finally:
                STATE.current_eval_id = None
                try:
                    flush_result = await sink.flush()
                    _eval_log(req.eval_id, "flush", "sink flush result=%s", flush_result)
                except Exception:
                    log.warning("DatasetSink.flush() failed for eval %r",
                                req.eval_id, exc_info=True)
                _eval_log(req.eval_id, "done", "finished")

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/prune_cache")
async def prune_cache(body: dict) -> JSONResponse:
    """Remove cached model dirs not in keep list; body: {"keep": [{"repo", "digest"}, ...]}."""
    if STATE.eval_lock.locked():
        return JSONResponse(
            {"ok": False, "error": "eval in progress — prune skipped"},
            status_code=409,
        )

    keep_list: list[dict] = body.get("keep", [])
    keep_refs: list[ModelRef] = []
    for item in keep_list:
        try:
            keep_refs.append(ModelRef(repo=item["repo"], digest=item["digest"]))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)

    async with STATE.eval_lock:
        freed = await asyncio.to_thread(prune_model_cache, *keep_refs)
    return JSONResponse({"ok": True, "freed_bytes": freed})


@app.post("/reset_fingerprints")
async def reset_fingerprints() -> JSONResponse:
    """Clear all near-duplicate fingerprint state (in-memory + persisted).

    Called by the validator's competition reset so a fresh replay re-fingerprints from
    scratch — otherwise every re-queued model false-matches its own prior fingerprint.
    """
    if STATE.eval_lock.locked():
        return JSONResponse({"ok": False, "error": "eval in progress"}, status_code=409)
    n = len(STATE.fingerprints)
    STATE.fingerprints.clear()
    try:
        await asyncio.to_thread(save_fingerprints, {})
    except Exception:
        log.warning("reset_fingerprints: persist clear failed", exc_info=True)
    log.warning("reset_fingerprints: cleared %d fingerprints", n)
    return JSONResponse({"ok": True, "cleared": n})


def main() -> None:
    """Launch the eval server with uvicorn."""
    import uvicorn
    host = os.environ.get("ALBEDO_EVAL_HOST", "0.0.0.0")
    port = int(os.environ.get("ALBEDO_EVAL_PORT", "9001"))  # matches validator + tunnel default
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
