"""Sanity service API - two endpoints: /health and /check."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from sanity_service import config, db
from sanity_service.runner import RUNNER, SANITY_PROMPTS, RunnerBusy


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Connect to Postgres on startup; close on shutdown.
    await db.init(config.DB_URL)
    yield
    await db.close()


app = FastAPI(title="Albedo Sanity Service", version="1.0", lifespan=_lifespan)


class CheckRequest(BaseModel):
    # Fields passed by the validator when requesting a sanity check.
    repo:            str
    digest:          str
    n_prompts:       int   = 3
    min_tokens:      int   = 5
    max_repetition:  float = 0.85
    min_vocab_ratio: float = 0.3


@app.get("/health")
async def health() -> JSONResponse:
    # Returns service liveness, whether a check is in progress, and which model is loaded.
    return JSONResponse({
        "ok":           True,
        "busy":         RUNNER.is_busy,
        "loaded_model": RUNNER.loaded_digest[:20] + "..." if RUNNER.loaded_digest else None,
        "current_job":  RUNNER.current_model or None,
        "db_connected": db.is_connected(),
        "ts_iso":       datetime.now(timezone.utc).isoformat(),
    })


@app.post("/check")
async def check(req: CheckRequest) -> JSONResponse:
    # Returns a cached result immediately if this digest was checked before.
    cached = await db.get_cached(req.digest)
    if cached:
        logger.info("[sanity] cache hit digest={:.16} passed={}", req.digest, cached["passed"])
        return JSONResponse({**cached, "prompts": SANITY_PROMPTS[:req.n_prompts]})

    logger.info("check requested repo={} digest={:.16}", req.repo, req.digest)
    try:
        result = await RUNNER.check(
            repo=req.repo,
            digest=req.digest,
            n_prompts=req.n_prompts,
            min_tokens=req.min_tokens,
            max_repetition=req.max_repetition,
            min_vocab_ratio=req.min_vocab_ratio,
        )
    except RunnerBusy as exc:
        # Another check slipped in between the cache lookup and the lock - reject as busy.
        raise HTTPException(status_code=409, detail=f"busy checking {exc}") from None

    # Skip caching transient infra failures so they stay retryable rather than permanently failed.
    if not result.infra_fault:
        await db.insert_result(result)

    return JSONResponse({
        "passed":       result.passed,
        "reason":       result.reason,
        "model_repo":   result.model_repo,
        "model_digest": result.model_digest,
        "checked_at":   result.checked_at,
        "responses":    result.responses,
        "prompts":      SANITY_PROMPTS[:req.n_prompts],
        "infra_fault":  result.infra_fault,
        "llm_gate":     result.llm_gate,
        "timing": {
            "total_s":      result.timing.total_s,
            "download_s":   result.timing.download_s,
            "vllm_s":       result.timing.vllm_s,
            "prompts_s":    result.timing.prompts_s,
            "model_cached": result.timing.model_cached,
            "vllm_reused":  result.timing.vllm_reused,
        },
        "cached": False,
    })
