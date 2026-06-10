"""Download model, start vLLM, run prompts, swap only when model changes."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

from sanity_service.checks import check_all

_SANITY_PORT      = int(os.environ.get("SANITY_VLLM_PORT", "9101"))
_SANITY_GPUS      = os.environ.get("SANITY_GPUS", "0")
_GPU_UTIL         = float(os.environ.get("SANITY_GPU_UTIL", "0.15"))
_VLLM_DTYPE       = os.environ.get("SANITY_VLLM_DTYPE", "bfloat16")
_DOWNLOAD_TIMEOUT = float(os.environ.get("SANITY_DOWNLOAD_TIMEOUT", "300"))
_VLLM_STARTUP_S   = float(os.environ.get("SANITY_VLLM_STARTUP_S", "120"))

# Prompts loaded from JSON so they can be edited without touching Python code.
_PROMPTS_FILE = Path(__file__).parent / "prompts.json"
SANITY_PROMPTS: list[str] = json.loads(_PROMPTS_FILE.read_text())


@dataclass
class TimingBreakdown:
    # Wall-clock seconds spent in each phase; 0 if the phase was skipped.
    download_s:   float = 0.0
    vllm_s:       float = 0.0
    prompts_s:    float = 0.0
    total_s:      float = 0.0
    model_cached: bool  = False
    vllm_reused:  bool  = False


@dataclass
class SanityResult:
    # Full outcome of one sanity check including timing and per-prompt responses.
    passed:        bool
    reason:        str             = ""
    responses:     list[str]       = field(default_factory=list)
    timing:        TimingBreakdown = field(default_factory=TimingBreakdown)
    model_repo:    str             = ""
    model_digest:  str             = ""
    checked_at:    str             = ""


class SanityRunner:
    # One vLLM process kept warm; swaps model only when the digest changes.

    def __init__(self) -> None:
        self._proc:          subprocess.Popen | None = None
        self._loaded_digest: str         = ""
        self._loaded_dir:    str         = ""
        self._lock:          asyncio.Lock = asyncio.Lock()
        self._current:       str         = ""

    @property
    def is_busy(self) -> bool:
        # True while a check is in progress.
        return self._lock.locked()

    @property
    def current_model(self) -> str:
        # Short label of the model currently being checked, empty when idle.
        return self._current

    @property
    def loaded_digest(self) -> str:
        # Digest of the model currently loaded in vLLM, empty if vLLM is not running.
        return self._loaded_digest

    async def _vllm_healthy(self) -> bool:
        # Returns True only if the process is running AND the HTTP health endpoint responds.
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (await c.get(f"http://localhost:{_SANITY_PORT}/health")).status_code == 200
        except Exception:
            return False

    async def check(
        self,
        repo: str,
        digest: str,
        n_prompts: int = 3,
        min_tokens: int = 5,
        max_repetition: float = 0.85,
        min_vocab_ratio: float = 0.3,
    ) -> SanityResult:
        # Runs the full check pipeline and returns a SanityResult regardless of outcome.
        async with self._lock:
            self._current = f"{repo}@{digest[:16]}"
            t_total = time.monotonic()
            timing  = TimingBreakdown()
            result  = SanityResult(
                passed=False,
                model_repo=repo,
                model_digest=digest,
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
            responses: list[str] = []
            try:
                same_model = (digest == self._loaded_digest
                              and await self._vllm_healthy())

                if same_model:
                    timing.model_cached = True
                    timing.vllm_reused  = True
                    model_dir = self._loaded_dir
                    logger.info("[sanity] same model loaded - skipping download+restart")
                else:
                    t0 = time.monotonic()
                    model_dir            = await self._materialize(repo, digest)
                    timing.download_s    = round(time.monotonic() - t0, 1)
                    timing.model_cached  = timing.download_s < 2.0  # near-zero = already on disk

                    t1 = time.monotonic()
                    await self._kill_vllm()
                    if self._loaded_dir and self._loaded_dir != model_dir:
                        await asyncio.to_thread(shutil.rmtree, self._loaded_dir, True)
                    await self._start_vllm(model_dir, digest)
                    self._loaded_digest = digest
                    self._loaded_dir    = model_dir
                    timing.vllm_s = round(time.monotonic() - t1, 1)

                t2 = time.monotonic()
                responses        = await self._run_prompts(digest, n_prompts)
                timing.prompts_s = round(time.monotonic() - t2, 1)
                result.responses = responses

                check = check_all(responses, min_tokens=min_tokens,
                                  max_repetition=max_repetition, min_vocab_ratio=min_vocab_ratio)
                result.passed = check.passed
                result.reason = check.reason

            except asyncio.TimeoutError:
                result.reason = "timed out"
                self._loaded_digest = ""
                self._loaded_dir    = ""
            except Exception as exc:
                result.reason = f"runner error: {exc}"
                logger.exception("[sanity] check failed for {}", repo)
                self._loaded_digest = ""
                self._loaded_dir    = ""
            finally:
                # Preserve any responses collected before the failure point.
                if not result.responses:
                    result.responses = responses

            timing.total_s = round(time.monotonic() - t_total, 1)
            result.timing  = timing
            self._current  = ""

            (logger.info if result.passed else logger.warning)(
                "[sanity] {} passed={} reason={!r} total={}s "
                "(download={}s vllm={}s prompts={}s cached={} reused={})",
                repo, result.passed, result.reason,
                timing.total_s, timing.download_s, timing.vllm_s,
                timing.prompts_s, timing.model_cached, timing.vllm_reused,
            )
            return result

    async def _materialize(self, repo: str, digest: str) -> str:
        # Downloads the model from Hippius and returns the local directory path.
        from albedo.models import ModelRef, materialize_model
        ref = ModelRef(repo=repo, digest=digest)
        return await asyncio.wait_for(
            asyncio.to_thread(materialize_model, ref),
            timeout=_DOWNLOAD_TIMEOUT,
        )

    async def _start_vllm(self, model_dir: str, model_name: str) -> None:
        # Launches a vLLM subprocess and waits until it reports healthy.
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  model_dir,
            "--served-model-name",      model_name,
            "--port",                   str(_SANITY_PORT),
            "--gpu-memory-utilization", str(_GPU_UTIL),
            "--dtype",                  _VLLM_DTYPE,
            "--max-model-len",          "4096",
            "--trust-remote-code",
        ]
        self._proc = subprocess.Popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": _SANITY_GPUS},
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(_VLLM_STARTUP_S)
        logger.info("[sanity] vLLM healthy on port {} model={:.40}", _SANITY_PORT, model_name)

    async def _wait_healthy(self, timeout: float) -> None:
        # Polls the vLLM health endpoint until it responds 200 or the timeout expires.
        url      = f"http://localhost:{_SANITY_PORT}/health"
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as c:
            while time.monotonic() < deadline:
                try:
                    if (await c.get(url)).status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")

    async def _run_prompts(self, model_name: str, n_prompts: int) -> list[str]:
        # Sends n_prompts test prompts in parallel and returns the responses.
        prompts = SANITY_PROMPTS[:n_prompts]
        url     = f"http://localhost:{_SANITY_PORT}/v1/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

        async def _one(prompt: str) -> str:
            try:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    r = await c.post(url, json={
                        "model":       model_name,
                        "messages":    [{"role": "user", "content": prompt}],
                        "max_tokens":  128,
                        "temperature": 0.0,
                    })
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                logger.warning("[sanity] prompt failed: {}", exc)
                return ""

        return list(await asyncio.gather(*[_one(p) for p in prompts]))

    async def _kill_vllm(self) -> None:
        # Kills the vLLM process group; retries if it doesn't exit within 5 seconds.
        if not self._proc:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("[sanity] vLLM did not exit after SIGKILL - retrying")
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass
        self._proc = None
        logger.info("[sanity] vLLM stopped")


# Module-level singleton shared by the FastAPI app.
RUNNER = SanityRunner()
