"""Sanity GPU worker - loads a challenger model, generates responses, runs heuristics."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from albedo_eval_service.remote_dataset import format_messages
from sanity_remote.config import SanityRemoteSettings, get_remote_settings
from sanity_remote.state import SanityRun
from sanity_service.checks import (
    check_code_present,
    check_collapsed,
    check_one,
    check_uniform_length,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FORBIDDEN_CONFIG_KEYS = frozenset({"auto_map", "quantization_config"})
_QWEN3_IM_END_TOKEN_ID = 151645


def _strip_thinking(text: str) -> str:
    # Removes <think>...</think> so heuristics evaluate the answer, not CoT reasoning.
    # Returns "" if thinking started but never closed - the model hit its token ceiling mid-thought.
    if "<think>" not in text:
        return text
    if "</think>" not in text:
        return ""
    return _THINK_RE.sub("", text).strip()


def _strip_model_config(model_dir: str) -> None:
    # Removes keys that can redirect model loading or force unexpected quantization modes.
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[sanity-remote] could not read config.json: {}", exc)
        return
    stripped = {k: v for k, v in config.items() if k not in _FORBIDDEN_CONFIG_KEYS}
    removed = set(config) - set(stripped)
    if not removed:
        return
    config_path.write_text(json.dumps(stripped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("[sanity-remote] stripped forbidden keys from config.json: {}", removed)


def _format_prompt_messages(
    tokenizer_path: str, prompt_messages: list[list[dict[str, str]]]
) -> list[str]:
    return [
        format_messages(messages, tokenizer_path=tokenizer_path, enable_thinking=False)
        for messages in prompt_messages
    ]


def _model_ref_parts(model_uri: str, digest: str) -> tuple[str, str]:
    repo, sep, uri_digest = model_uri.partition("@")
    return repo, uri_digest if sep else digest


class WorkerFault(Exception):
    # Carries a fault code + retryability for the run's failure event.
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class VllmEngine:
    # One warm vLLM process; swaps the model only when the digest changes (ported from runner.py).

    def __init__(self, settings: SanityRemoteSettings) -> None:
        self._s = settings
        self._proc: subprocess.Popen | None = None
        self._loaded_digest = ""
        self._loaded_dir = ""
        self._lock = asyncio.Lock()
        self._kill_port_squatter()

    async def run_job(
        self,
        model_uri: str,
        digest: str,
        prompts: list[str],
        max_tokens: int,
        prompt_messages: list[list[dict[str, str]]] | None = None,
    ) -> list[str]:
        # Serializes one generation job: ensure the model is loaded, then generate the prompts.
        async with self._lock:
            await self._ensure_model(model_uri, digest)
            if prompt_messages is not None:
                prompts = await asyncio.to_thread(
                    _format_prompt_messages, self._loaded_dir, prompt_messages
                )
            return await self._run_prompts(digest, prompts, max_tokens)

    def _kill_port_squatter(self) -> None:
        # On startup, kill any orphaned vLLM process that may still hold the configured port.
        # Without this, a restart of the worker process leaves _proc=None so _kill_vllm() is
        # a no-op, _wait_healthy() immediately returns True against the old server, and the new
        # digest is written to _loaded_digest while vLLM still serves the previous model → 404.
        import socket

        try:
            with socket.socket() as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", self._s.vllm_port)) != 0:
                    return  # port is free, nothing to kill
        except Exception:
            return
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self._s.vllm_port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for pid_str in result.stdout.split():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                    logger.info(
                        "[sanity-remote] killed orphan vLLM pid={} on port {}",
                        pid_str,
                        self._s.vllm_port,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def forget(self) -> None:
        # Forces a reload next time; keeps _loaded_dir so a stale model stays reclaimable.
        self._loaded_digest = ""

    async def _ensure_model(self, model_uri: str, digest: str) -> None:
        # Reuses a healthy warm model, otherwise downloads + swaps vLLM to the new digest.
        if digest == self._loaded_digest and await self._healthy():
            logger.info("[sanity-remote] reusing warm model {:.16}", digest)
            return
        try:
            model_dir = await asyncio.wait_for(
                self._materialize(model_uri, digest), timeout=self._s.download_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise WorkerFault(
                "download_timeout", f"download exceeded {self._s.download_timeout_s}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - download failures are retryable infra by default
            raise WorkerFault("download_failed", f"model download failed: {exc}") from exc

        await self._kill_vllm()
        old_dir = self._loaded_dir
        self._loaded_digest = ""
        self._loaded_dir = model_dir
        await asyncio.to_thread(_strip_model_config, model_dir)
        try:
            await self._start_vllm(model_dir, digest)
        except Exception as exc:  # noqa: BLE001 - boot failures are retryable infra
            raise WorkerFault("vllm_boot_failed", f"vLLM did not start: {exc}") from exc
        self._loaded_digest = digest
        if old_dir and old_dir != model_dir:
            await asyncio.to_thread(shutil.rmtree, old_dir, True)

    async def _materialize(self, model_uri: str, digest: str) -> str:
        # Downloads the model from Hippius and returns the local directory path.
        from hippius_validation.hippius import download_full, make_ref

        repo, ref_digest = _model_ref_parts(model_uri, digest)
        return await asyncio.to_thread(download_full, make_ref(repo, ref_digest))

    async def _start_vllm(self, model_dir: str, model_name: str) -> None:
        # Launches a vLLM subprocess (no --trust-remote-code) and waits until it reports healthy.
        cmd = [
            self._s.vllm_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_dir,
            "--served-model-name",
            model_name,
            "--port",
            str(self._s.vllm_port),
            "--gpu-memory-utilization",
            str(self._s.gpu_util),
            "--dtype",
            self._s.vllm_dtype,
            "--max-model-len",
            str(self._s.max_model_len),
            "--generation-config",
            "vllm",
        ]
        self._proc = subprocess.Popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": self._s.gpu_ids},
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(self._s.vllm_startup_s)
        logger.info(
            "[sanity-remote] vLLM healthy on {} model={:.40}", self._s.vllm_port, model_name
        )

    async def _healthy(self) -> bool:
        # True only if the process is alive AND its health endpoint responds 200.
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (
                    await c.get(f"http://localhost:{self._s.vllm_port}/health")
                ).status_code == 200
        except Exception:  # noqa: BLE001 - any probe failure means not healthy
            return False

    async def _wait_healthy(self, timeout: float) -> None:
        # Polls the vLLM health endpoint until 200 or the timeout expires.
        url = f"http://localhost:{self._s.vllm_port}/health"
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as c:
            while time.monotonic() < deadline:
                try:
                    if (await c.get(url)).status_code == 200:
                        return
                except Exception:  # noqa: BLE001 - keep polling until the deadline
                    pass
                await asyncio.sleep(2.0)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")

    async def _run_prompts(self, model_name: str, prompts: list[str], max_tokens: int) -> list[str]:
        # Per prompt: HTTP-error/malformed -> "" (model fault); transport error -> raise (infra).
        #
        # Keep this aligned with the full eval worker: SWE-ZERO prompts are already formatted
        # with the Qwen chat template, so they should be sent as raw completions.
        url = f"http://localhost:{self._s.vllm_port}/v1/completions"
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)

        async def _one(prompt: str) -> str:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    url,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": self._s.gen_temperature,
                        "top_p": self._s.gen_top_p,
                        "top_k": self._s.gen_top_k,
                        "min_p": self._s.gen_min_p,
                        "stop_token_ids": [_QWEN3_IM_END_TOKEN_ID],
                    },
                )
            if r.status_code >= 400:
                logger.warning(
                    "[sanity-remote] vLLM HTTP {} for prompt - model fault", r.status_code
                )
                return ""
            try:
                choice = r.json()["choices"][0]
                raw = choice["text"] or ""
                finish = choice.get("finish_reason", "unknown")
                answer = _strip_thinking(raw)
                logger.info(
                    "[sanity-remote] prompt finish={} thinking={} answer_words={}",
                    finish,
                    "<think>" in raw,
                    len(answer.split()),
                )
                return answer
            except (KeyError, IndexError, ValueError):
                logger.warning("[sanity-remote] malformed vLLM response body - model fault")
                return ""

        results = await asyncio.gather(*[_one(p) for p in prompts], return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                raise WorkerFault(
                    "generation_transport_error", f"vLLM request failed: {res}"
                ) from res
        return list(results)

    async def _kill_vllm(self) -> None:
        # Kills the vLLM process group; retries once if it doesn't exit within 5 seconds.
        if not self._proc:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001 - process may already be gone
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("[sanity-remote] vLLM did not exit after SIGKILL - retrying")
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - best-effort kill
                pass
        except Exception:  # noqa: BLE001 - best-effort reap
            pass
        self._proc = None


_ENGINE: VllmEngine | None = None


def _engine() -> VllmEngine:
    # Lazily builds the process-wide vLLM engine singleton.
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VllmEngine(get_remote_settings())
    return _ENGINE


def _heuristics(responses: list[str], req: Any, skip: bool = False) -> list[dict[str, Any]]:
    # Per-response heuristic verdicts; a set-level collapse signal fails all responses.
    if skip:
        return [{"passed": True, "reason": "heuristics disabled"} for _ in responses]

    out: list[dict[str, Any]] = []
    for resp in responses:
        r = check_one(
            resp,
            min_tokens=req.min_tokens,
            max_repetition=req.max_repetition,
            min_vocab_ratio=req.min_vocab_ratio,
        )
        out.append({"passed": r.passed, "reason": r.reason})
    if any(not item["passed"] for item in out):
        return out

    set_fail = next(
        (
            c
            for c in (
                check_collapsed(responses),
                check_uniform_length(responses),
                check_code_present(responses),
            )
            if not c.passed
        ),
        None,
    )
    if set_fail is not None:
        return [{"passed": False, "reason": set_fail.reason} for _ in responses]
    return out


async def generate(run: SanityRun, settings: SanityRemoteSettings | None = None) -> None:
    # Executes one run: ensure model -> generate -> heuristics -> emit result (or a fault).
    s = settings or get_remote_settings()
    req = run.request
    if s.mock_auto_result:
        run.succeed(
            responses=[f"mock response to: {p[:30]}" for p in req.prompts],
            heuristics=[{"passed": True, "reason": "mock"} for _ in req.prompts],
        )
        return

    engine = _engine()
    try:
        run.append_event({"type": "generation_started", "run_id": run.run_id})
        responses = await engine.run_job(
            req.model_uri,
            req.digest,
            req.prompts,
            req.gen_max_tokens,
            req.prompt_messages,
        )
        run.succeed(
            responses=responses,
            heuristics=_heuristics(responses, req, skip=s.skip_heuristics),
        )
    except WorkerFault as fault:
        engine.forget()
        logger.warning(
            "[sanity-remote] worker fault code={} digest={:.16} retryable={}: {}",
            fault.code,
            req.digest,
            fault.retryable,
            fault,
        )
        run.fail(fault_code=fault.code, fault_message=str(fault), retryable=fault.retryable)
    except Exception as exc:  # noqa: BLE001 - never let a worker crash strand the run
        engine.forget()
        logger.exception("[sanity-remote] generation failed for {}", req.digest)
        run.fail(fault_code="worker_error", fault_message=str(exc), retryable=True)
