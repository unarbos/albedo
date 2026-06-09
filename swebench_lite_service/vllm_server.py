from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx

from .config import SETTINGS


class VLLMServer:
    def __init__(self, *, model_dir: str, served_model_name: str = "albedo-king") -> None:
        self.model_dir = model_dir
        self.served_model_name = served_model_name
        self.proc: subprocess.Popen | None = None

    async def __aenter__(self) -> "VLLMServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await asyncio.to_thread(self.stop)

    async def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            SETTINGS.vllm_host,
            "--port",
            str(SETTINGS.vllm_port),
            "--model",
            self.model_dir,
            "--served-model-name",
            self.served_model_name,
            f"hosted_vllm/{self.served_model_name}",
            f"openai/{self.served_model_name}",
            "--dtype",
            SETTINGS.vllm_dtype,
            "--max-model-len",
            str(SETTINGS.vllm_max_model_len),
            "--gpu-memory-utilization",
            str(SETTINGS.vllm_gpu_memory_utilization),
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
        ]
        attention_backend = os.environ.get("ALBEDO_SWEBENCH_VLLM_ATTENTION_BACKEND", "").strip()
        if attention_backend:
            cmd.extend(["--attention-backend", attention_backend])
        if SETTINGS.vllm_data_parallel_size > 1:
            cmd.extend([
                "--data-parallel-size",
                str(SETTINGS.vllm_data_parallel_size),
                "--data-parallel-size-local",
                str(SETTINGS.vllm_data_parallel_size),
            ])
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": SETTINGS.vllm_gpus,
            "ALBEDO_MODEL_CACHE_DIR": SETTINGS.model_cache_dir,
        }
        self.proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        await self.wait_healthy()

    async def wait_healthy(self, timeout_s: float = 600.0) -> None:
        url = f"http://{SETTINGS.vllm_host}:{SETTINGS.vllm_port}/health"
        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError("vLLM exited before becoming healthy")
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"vLLM did not become healthy within {timeout_s:.0f}s")
                await asyncio.sleep(2)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self.proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

