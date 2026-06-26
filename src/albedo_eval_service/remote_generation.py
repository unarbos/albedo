from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
from dataclasses import dataclass
from typing import Protocol

from .remote_dataset import EvalSample

_QWEN3_IM_END_TOKEN_ID = 248046  # <|im_end|> for Qwen3.6-35B-A3B (was 151645 for Qwen3-4B genesis)


@dataclass(frozen=True)
class GenerationResult:
    sample_id: str
    text: str
    error: str | None = None


class Generator(Protocol):
    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]: ...


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
        gpu_memory_utilization: float = 0.95,
        kv_cache_dtype: str = "auto",
    ):
        self.model = model
        self.gpu_ids = gpu_ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_dtype = kv_cache_dtype

    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]:
        if not samples:
            return []

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_vllm_worker,
            kwargs={
                "model": self.model,
                "gpu_ids": self.gpu_ids,
                "prompts": [sample.prompt for sample in samples],
                "sample_ids": [sample.sample_id for sample in samples],
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_model_len": self.max_model_len,
                "enforce_eager": self.enforce_eager,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "kv_cache_dtype": self.kv_cache_dtype,
                "queue": result_queue,
            },
        )
        process.start()
        payload = None
        while process.is_alive():
            try:
                payload = result_queue.get(timeout=1)
                break
            except queue_module.Empty:
                continue
        if payload is None:
            try:
                payload = result_queue.get_nowait()
            except queue_module.Empty:
                payload = {
                    "error": f"vLLM process exited {process.exitcode} without result payload"
                }
        process.join()

        if process.exitcode != 0:
            error = payload.get("error") or f"vLLM process exited {process.exitcode}"
            return [
                GenerationResult(sample_id=sample.sample_id, text="", error=error)
                for sample in samples
            ]
        if payload.get("error"):
            return [
                GenerationResult(sample_id=sample.sample_id, text="", error=payload["error"])
                for sample in samples
            ]
        return [GenerationResult(**item) for item in payload["results"]]


def _vllm_worker(
    *,
    model: str,
    gpu_ids: list[str],
    prompts: list[str],
    sample_ids: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_model_len: int | None,
    enforce_eager: bool,
    gpu_memory_utilization: float = 0.95,
    kv_cache_dtype: str = "auto",
    queue=None,
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
        outputs = llm.generate(prompts, params)
        results = []
        for sample_id, output in zip(sample_ids, outputs, strict=True):
            text = output.outputs[0].text if output.outputs else ""
            results.append({"sample_id": sample_id, "text": text, "error": None})
        queue.put({"results": results})
    except Exception as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})
