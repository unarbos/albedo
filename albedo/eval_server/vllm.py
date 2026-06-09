"""albedo.eval_server.vllm — Manage one vLLM subprocess (king or challenger)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import time

import httpx

from albedo.config import DUEL_GEN_MAX_LEN

log = logging.getLogger(__name__)


def reclaim_stray_on_ports(ports: list[int]) -> None:
    """SIGKILL any process group still listening on the given ports.

    vLLM children are launched detached (start_new_session=True), so if the eval
    server was hard-killed (PM2 SIGKILL / OOM / crash) without running its graceful
    shutdown, the old king/challenger vLLM processes survive and keep holding GPU
    memory — the next startup would then OOM. Called once on startup to free them.
    """
    want = {str(p) for p in ports}
    try:
        out = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception as exc:
        log.warning("startup reclaim: could not list listening ports: %s", exc)
        return

    killed_pgids: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]  # e.g. "0.0.0.0:8001"
        if local.rsplit(":", 1)[-1] not in want:
            continue
        for pid in {int(m) for m in re.findall(r"pid=(\d+)", line)}:
            try:
                pgid = os.getpgid(pid)
                if pgid in killed_pgids:
                    continue
                os.killpg(pgid, signal.SIGKILL)
                killed_pgids.add(pgid)
                log.warning("startup reclaim: SIGKILL stray pgid=%d holding %s", pgid, local)
            except (ProcessLookupError, PermissionError):
                pass

_GPU_MEMORY_UTILIZATION = os.environ.get("ALBEDO_GPU_MEMORY_UTILIZATION", "0.55")
_VLLM_DTYPE = os.environ.get("ALBEDO_VLLM_DTYPE", "bfloat16")


def _tensor_parallel_size(gpus: str) -> int:
    """vLLM tensor-parallel degree = number of GPUs assigned (e.g. '0,1,2,3' -> 4)."""
    return max(1, len([g for g in gpus.split(",") if g.strip()]))


class VLLMProcess:
    """Manages a single vLLM OpenAI-compatible server subprocess."""

    def __init__(self, *, role: str, gpus: str, port: int) -> None:
        self._role = role
        self._gpus = gpus
        self._port = port
        self._model_name: str = ""
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._client: httpx.AsyncClient | None = None
        self._log_fh = None  # per-role vLLM stdout/stderr sink

    async def start(self, model_dir: str, model_name: str) -> None:
        """Stop any running process and launch a fresh vLLM subprocess."""
        if self._model_name == model_name and self.is_alive():
            log.info("[%s] vLLM already running model %r — skipping restart", self._role, model_name)
            return

        await asyncio.to_thread(self.stop)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(self._port),
            "--max-model-len", str(DUEL_GEN_MAX_LEN),
            "--dtype", _VLLM_DTYPE,
            "--gpu-memory-utilization", _GPU_MEMORY_UTILIZATION,
            # Shard across every assigned GPU so larger models fit (e.g. "0,1,2,3" -> TP=4).
            "--tensor-parallel-size", str(_tensor_parallel_size(self._gpus)),
            "--generation-config", "vllm",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self._gpus}

        # Send this vLLM's verbose engine output + crash tracebacks to its own
        # per-role file so they don't pollute the eval-server log. Appended (not
        # truncated) so the tail before a crash/restart is preserved for debugging.
        log_dir = os.environ.get("ALBEDO_EVAL_LOG_DIR", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"vllm_{self._role}.log")
        self._log_fh = open(log_path, "a", buffering=1)
        self._log_fh.write(
            f"\n===== [{self._role}] vLLM start port={self._port} gpus={self._gpus} "
            f"model={model_name} =====\n"
        )
        self._log_fh.flush()

        log.info("[%s] starting vLLM on port %d — gpus=%s model=%r (output -> %s)",
                 self._role, self._port, self._gpus, model_name, log_path)
        self._proc = subprocess.Popen(
            cmd, env=env,
            start_new_session=True,  # isolate into its own process group
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        self._model_name = model_name
        self._started_at = time.monotonic()

        # Recycle client so it points at the fresh process
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def stop(self) -> None:
        """SIGTERM the process group; SIGKILL after 5 s if still alive."""
        if self._proc is None:
            return
        pgid = None
        try:
            pgid = os.getpgid(self._proc.pid)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self._proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("[%s] vLLM did not exit after SIGTERM — sending SIGKILL", self._role)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self._proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        log.info("[%s] vLLM process stopped", self._role)
        self._proc = None
        self._model_name = ""
        self._started_at = None

        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    async def wait_healthy(self, *, timeout: float = 180.0) -> None:
        """Poll GET /health until 200; raises TimeoutError or RuntimeError on early exit."""
        url = f"http://localhost:{self._port}/health"
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while True:
                if not self.is_alive():
                    raise RuntimeError(f"[{self._role}] vLLM process exited before becoming healthy")
                try:
                    resp = await probe.get(url)
                    if resp.status_code == 200:
                        log.info("[%s] vLLM healthy on port %d", self._role, self._port)
                        return
                except httpx.TransportError:
                    pass

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"[{self._role}] vLLM did not become healthy within {timeout}s"
                    )
                await asyncio.sleep(2.0)

    def is_alive(self) -> bool:
        """True if the subprocess is still running."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    async def is_healthy(self, *, timeout: float = 5.0) -> bool:
        if not self.is_alive():
            return False
        url = f"http://localhost:{self._port}/health"
        try:
            async with httpx.AsyncClient(timeout=timeout) as probe:
                resp = await probe.get(url)
                return resp.status_code == 200
        except httpx.TransportError:
            return False

    @property
    def model_name(self) -> str:
        """Currently loaded model name (immutable_ref string)."""
        return self._model_name

    @property
    def port(self) -> int:
        """Port this process listens on."""
        return self._port

    @property
    def uptime_s(self) -> float | None:
        """Seconds since start, or None if not running."""
        if self._started_at is None or not self.is_alive():
            return None
        return time.monotonic() - self._started_at

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy httpx.AsyncClient pre-pointed at this process's base URL."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"http://localhost:{self._port}",
                # Long read timeout: a 4B model generating 1024 tokens at ~50 tok/s
                # takes ~20s normally; allow 3× headroom for loaded GPU boxes.
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
            )
        return self._client
