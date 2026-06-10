"""Sanity service API - two endpoints: /health and /check."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from sanity_service.runner import RUNNER

app = FastAPI(title="Albedo Sanity Service", version="1.0")


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
        "ts_iso":       datetime.now(timezone.utc).isoformat(),
    })


@app.post("/check")
async def check(req: CheckRequest) -> JSONResponse:
    # Runs a full sanity check; blocks until complete and returns 409 if already busy.
    if RUNNER.is_busy:
        raise HTTPException(
            status_code=409,
            detail=f"busy checking {RUNNER.current_model}",
        )

    logger.info("check requested repo={} digest={:.16}", req.repo, req.digest)
    result = await RUNNER.check(
        repo=req.repo,
        digest=req.digest,
        n_prompts=req.n_prompts,
        min_tokens=req.min_tokens,
        max_repetition=req.max_repetition,
        min_vocab_ratio=req.min_vocab_ratio,
    )

    return JSONResponse({
        "passed":       result.passed,
        "reason":       result.reason,
        "model_repo":   result.model_repo,
        "model_digest": result.model_digest,
        "checked_at":   result.checked_at,
        "timing": {
            "total_s":      result.timing.total_s,
            "download_s":   result.timing.download_s,
            "vllm_s":       result.timing.vllm_s,
            "prompts_s":    result.timing.prompts_s,
            "model_cached": result.timing.model_cached,
            "vllm_reused":  result.timing.vllm_reused,
        },
        "n_responses":  len(result.responses),
    })
