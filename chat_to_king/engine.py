"""Warm vLLM engine that serves the current king and hot-swaps on coronation.

Differs from the preeval worker (sanity_remote/worker.py) in two ways:
  - it stays warm (no per-run teardown), and
  - the model is downloaded *before* the old one is killed, so the old king keeps serving through the
    (multi-minute) download; only the short kill->boot window is down, and the gateway covers it with an
    in-chat reload notice.

The vLLM process lifecycle (port-squatter reclaim, launch, health, kill) is copied from the preeval
worker and adapted for a stable served-model-name + persistent server. The stateless helpers
(download/seed-inject/config-strip/ref-parse) are imported from it so we don't diverge.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
import subprocess
import time

import httpx
from loguru import logger

from sanity_remote.worker import (
    _inject_seed_processor_files,
    _model_present,
    _model_ref_parts,
    _strip_model_config,
)
from albedo_eval_service.remote_dataset import _QWEN3_CHAT_TEMPLATE

from config import KingChatSettings
from king_source import King


class KingVllmEngine:
    def __init__(self, settings: KingChatSettings) -> None:
        self._s = settings
        self._proc: subprocess.Popen | None = None
        self._loaded_digest = ""
        self._loaded_dir = ""
        self._lock = asyncio.Lock()
        self.reloading = False
        self.serving_king: King | None = None
        self.incoming_king: King | None = None
        self._kill_port_squatter()


    @property
    def serving(self) -> bool:
        return bool(self._loaded_digest)

    async def healthy(self) -> bool:
        return await self._healthy()

    async def ensure_king(self, king: King) -> None:
        async with self._lock:
            if king.digest == self._loaded_digest and await self._healthy():
                return
            logger.info(
                "[king-chat] coronation: digest={:.16} uid={} v={} uri={}",
                king.digest, king.uid, king.king_version, king.model_uri,
            )
            self.incoming_king = king
            try:
                model_dir = await asyncio.wait_for(
                    self._materialize(king.model_uri, king.digest),
                    timeout=self._s.download_timeout_s,
                )
            except Exception as exc:
                self.incoming_king = None
                logger.error("[king-chat] download failed (keeping current king): {}", exc)
                return

            self.reloading = True
            try:
                await self._kill_vllm()
                old_dir = self._loaded_dir
                self._loaded_digest = ""
                self._loaded_dir = model_dir
                await asyncio.to_thread(_strip_model_config, model_dir)
                await self._start_vllm(model_dir)
                self._loaded_digest = king.digest
                self.serving_king = king
                if old_dir and old_dir != model_dir:
                    await asyncio.to_thread(shutil.rmtree, old_dir, True)
                    logger.info("[king-chat] deleted previous king at {}", old_dir)
            except Exception as exc:
                self._loaded_digest = ""
                logger.error("[king-chat] vLLM boot failed: {}", exc)
            finally:
                self.reloading = False
                self.incoming_king = None

    async def restart_loaded(self) -> None:
        async with self._lock:
            if not self._loaded_dir:
                return
            logger.warning("[king-chat] restarting vLLM from cache {}", self._loaded_dir)
            self.reloading = True
            try:
                await self._kill_vllm()
                await self._start_vllm(self._loaded_dir)
            except Exception as exc:
                self._loaded_digest = ""
                logger.error("[king-chat] restart from cache failed: {}", exc)
            finally:
                self.reloading = False


    async def _materialize(self, model_uri: str, digest: str) -> str:
        from hippius_validation.hippius import cache_dir, download_full, make_ref

        repo, ref_digest = _model_ref_parts(model_uri, digest)
        ref = make_ref(repo, ref_digest)
        dest = str(cache_dir(ref))
        if _model_present(dest):
            logger.info("[king-chat] reusing on-disk model at {} — skipping download", dest)
        else:
            logger.info("[king-chat] downloading {} digest={:.16} to {}", repo, ref_digest, dest)
            dest = await asyncio.to_thread(download_full, ref)
        await asyncio.to_thread(_inject_seed_processor_files, dest)
        return dest

    async def _start_vllm(self, model_dir: str) -> None:
        s = self._s
        logger.info("[king-chat] starting vLLM port={} model={}", s.vllm_port, s.served_model_name)
        template_path = os.path.join(s.models_dir, "albedo_chat_template.jinja")
        with open(template_path, "w", encoding="utf-8") as fh:
            fh.write(_QWEN3_CHAT_TEMPLATE)
        cmd = [
            s.vllm_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--served-model-name", s.served_model_name,
            "--host", s.vllm_host,
            "--port", str(s.vllm_port),
            "--gpu-memory-utilization", str(s.gpu_util),
            "--dtype", s.vllm_dtype,
            "--max-model-len", str(s.max_model_len),
            "--kv-cache-dtype", s.kv_cache_dtype,
            "--chat-template", template_path,
            "--generation-config", "vllm",
        ]
        if s.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(s.tensor_parallel_size)]
        if s.max_num_seqs > 0:
            cmd += ["--max-num-seqs", str(s.max_num_seqs)]
        if s.vllm_limit_mm:
            cmd += ["--limit-mm-per-prompt", s.vllm_limit_mm]
        if s.cpu_offload_gb > 0:
            cmd += ["--cpu-offload-gb", str(s.cpu_offload_gb)]
        if s.vllm_quantization:
            cmd += ["--quantization", s.vllm_quantization]
        if s.vllm_enforce_eager:
            cmd += ["--enforce-eager"]
        if s.vllm_moe_backend:
            cmd += ["--moe-backend", s.vllm_moe_backend]
        self._proc = subprocess.Popen(
            cmd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": s.gpu_ids,
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
            start_new_session=True,
        )
        await self._wait_healthy(s.vllm_startup_s)
        logger.info("[king-chat] vLLM healthy on {} model={}", s.vllm_port, s.served_model_name)

    async def _healthy(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (await c.get(f"http://localhost:{self._s.vllm_port}/health")).status_code == 200
        except Exception:
            return False

    async def _wait_healthy(self, timeout: float) -> None:
        url = f"http://localhost:{self._s.vllm_port}/health"
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

    async def _kill_vllm(self) -> None:
        if not self._proc:
            return
        logger.info("[king-chat] killing vLLM pid={}", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("[king-chat] vLLM did not exit after SIGKILL - retrying")
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass
        self._proc = None

    def _kill_port_squatter(self) -> None:
        try:
            with socket.socket() as sk:
                sk.settimeout(0.5)
                if sk.connect_ex(("127.0.0.1", self._s.vllm_port)) != 0:
                    return
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
                    logger.info("[king-chat] killed orphan vLLM pid={} on port {}", pid_str, self._s.vllm_port)
                except Exception:
                    pass
        except Exception:
            pass
