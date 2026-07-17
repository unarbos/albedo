from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import time
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from .remote_dataset import EvalSample

_QWEN3_IM_END_TOKEN_ID = 248046  # <|im_end|> for Qwen3.6-35B-A3B (was 151645 for Qwen3-4B genesis)


@dataclass(frozen=True)
class GenerationResult:
    sample_id: str
    text: str
    error: str | None = None
    turns: list[dict[str, Any]] | None = None


class Generator(Protocol):
    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]: ...


def format_scored_trajectory(turns: list[dict[str, Any]]) -> str:
    target_count = sum(
        1 for turn in turns if turn.get("role") == "assistant" and turn.get("score_target")
    )
    target_label = (
        "CANDIDATE OUTPUT"
        if target_count == 1
        else f"CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT {target_count}"
    )
    assistant_index = 0
    parts = [
        "FULL CANDIDATE TRAJECTORY",
        f"Score ONLY {target_label}. "
        "The ENVIRONMENT OBSERVATION is context only.",
    ]
    for turn in turns:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").rstrip()
        if role == "assistant" and turn.get("score_target"):
            assistant_index += 1
            label = f"CANDIDATE OUTPUT {assistant_index}"
        elif role == "user" and turn.get("environment_observation"):
            label = "ENVIRONMENT OBSERVATION (context only, do not score)"
        else:
            label = f"CONTEXT {role.upper()} (do not score)" if role else "CONTEXT TURN (do not score)"
        parts.append(f"\n{label}:\n------\n{content}\n------")
    return "\n".join(parts).strip()


class VllmProcessGenerator:
    def __init__(
        self,
        *,
        model: str,
        gpu_ids: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        compile_cache_dir: str = "",
        gpu_memory_utilization: float = 0.95,
        kv_cache_dtype: str = "auto",
        result_timeout_seconds: float = 900.0,
    ):
        self.model = model
        self.gpu_ids = gpu_ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.compile_cache_dir = compile_cache_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_dtype = kv_cache_dtype
        self.result_timeout_seconds = result_timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._result_queue = None
        self._process = None
        self._request_id = 0

    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]:
        if not samples:
            return []

        self._start()
        self._request_id += 1
        request_id = str(self._request_id)
        self._request_queue.put(
            {
                "id": request_id,
                "prompts": [sample.prompt for sample in samples],
                "sample_ids": [sample.sample_id for sample in samples],
            }
        )
        payload = self._wait_for_payload(request_id, samples)
        if payload.get("error"):
            return [
                GenerationResult(sample_id=sample.sample_id, text="", error=payload["error"])
                for sample in samples
            ]
        return [GenerationResult(**item) for item in payload["results"]]

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive() and self._request_queue is not None:
            self._request_queue.put(None)
            self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._process = None
        self._request_queue = None
        self._result_queue = None

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._request_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_vllm_worker,
            kwargs={
                "model": self.model,
                "gpu_ids": self.gpu_ids,
                "prompts": None,
                "sample_ids": None,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_model_len": self.max_model_len,
                "enforce_eager": self.enforce_eager,
                "compile_cache_dir": self.compile_cache_dir,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "kv_cache_dtype": self.kv_cache_dtype,
                "queue": self._result_queue,
                "request_queue": self._request_queue,
            },
        )
        self._process.start()

    def _wait_for_payload(self, request_id: str, samples: list[EvalSample]) -> dict[str, Any]:
        payload = None
        deadline = time.monotonic() + max(1.0, self.result_timeout_seconds)
        while self._process is not None and self._process.is_alive():
            if time.monotonic() >= deadline:
                payload = {
                    "error": (
                        f"vLLM process produced no result payload after "
                        f"{self.result_timeout_seconds:g}s"
                    )
                }
                break
            try:
                candidate = self._result_queue.get(timeout=1)
                if candidate.get("id") == request_id or ("id" not in candidate and candidate.get("error")):
                    payload = candidate
                    break
            except queue_module.Empty:
                continue
        if payload is None:
            try:
                payload = self._result_queue.get_nowait()
            except queue_module.Empty:
                payload = {
                    "error": f"vLLM process exited {self._process.exitcode} without result payload"
                }
        if self._process is not None and self._process.exitcode not in (None, 0):
            payload["error"] = payload.get("error") or f"vLLM process exited {self._process.exitcode}"
        return payload


def _vllm_worker(
    *,
    model: str,
    gpu_ids: list[str],
    prompts: list[str] | None,
    sample_ids: list[str] | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_model_len: int | None,
    enforce_eager: bool,
    compile_cache_dir: str = "",
    gpu_memory_utilization: float = 0.95,
    kv_cache_dtype: str = "auto",
    queue=None,
    request_queue=None,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": len(gpu_ids),
            "trust_remote_code": True,
            # Match the benchmark eval server's `--generation-config vllm`:
            # do not let vLLM auto-import Hugging Face generation_config.json.
            "generation_config": "vllm",
            "reasoning_parser": "qwen3",
            "enable_prefix_caching": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "kv_cache_dtype": kv_cache_dtype,
            # Text-only eval: cap multimodal inputs to 0 so vLLM skips vision-encoder
            # profiling, which hangs for the multimodal Qwen3.6 genesis architecture.
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if compile_cache_dir:
            llm_kwargs["compilation_config"] = {"cache_dir": compile_cache_dir}
        llm = LLM(**llm_kwargs)
        params_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": [_QWEN3_IM_END_TOKEN_ID],
        }
        if top_k is not None:
            params_kwargs["top_k"] = top_k
        params = SamplingParams(**params_kwargs)

        if request_queue is None:
            queue.put(_generate_payload(llm, params, prompts or [], sample_ids or []))
            return

        while True:
            request = request_queue.get()
            if request is None:
                return
            try:
                payload = _generate_payload(llm, params, request["prompts"], request["sample_ids"])
            except Exception as exc:
                logger.exception(f"[remote-gen] vLLM request failed model={model} gpu_ids={gpu_ids}: {exc}")
                payload = {"error": f"{type(exc).__name__}: {exc}"}
            payload["id"] = request["id"]
            queue.put(payload)
    except Exception as exc:
        logger.exception(f"[remote-gen] vLLM worker failed model={model} gpu_ids={gpu_ids}: {exc}")
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def _generate_payload(llm: Any, params: Any, prompts: list[str], sample_ids: list[str]) -> dict[str, Any]:
    outputs = llm.generate(prompts, params)
    results = []
    for sample_id, output in zip(sample_ids, outputs, strict=True):
        text = output.outputs[0].text if output.outputs else ""
        results.append({"sample_id": sample_id, "text": text, "error": None})
    return {"results": results}
