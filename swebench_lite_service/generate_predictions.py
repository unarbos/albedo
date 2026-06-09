from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset

from .config import SETTINGS


SYSTEM_PROMPT = (
    "You are a senior Python maintainer. Produce a minimal unified diff patch that "
    "fixes the issue. Output only the patch. Do not use Markdown fences."
)


def load_instances(limit: int = 0) -> list[dict[str, Any]]:
    dataset = load_dataset(SETTINGS.dataset_name, split=SETTINGS.split)
    rows = [dict(row) for row in dataset]
    return rows[:limit] if limit and limit > 0 else rows


async def generate_predictions(*, out_path: Path, raw_path: Path, model_name: str = "albedo-king") -> dict[str, Any]:
    instances = load_instances(SETTINGS.limit_instances)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(SETTINGS.generation_concurrency)
    timeout = httpx.Timeout(
        connect=10.0,
        read=SETTINGS.generation_timeout_s,
        write=30.0,
        pool=30.0,
    )
    base_url = f"http://{SETTINGS.vllm_host}:{SETTINGS.vllm_port}/v1"
    predictions: list[dict[str, Any]] = []
    raw_generations: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        tasks = [_one_prediction(client, sem, instance, model_name) for instance in instances]
        for item in await asyncio.gather(*tasks):
            predictions.append(item["prediction"])
            raw_generations.append(item["raw"])

    with out_path.open("w") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")
    raw_path.write_text(json.dumps(raw_generations, indent=2, sort_keys=True) + "\n")

    empty = sum(1 for pred in predictions if not pred["model_patch"])
    return {
        "instances": len(instances),
        "predictions_path": str(out_path),
        "raw_generations_path": str(raw_path),
        "empty_patches": empty,
    }


async def _one_prediction(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    instance: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    async with sem:
        prompt = _prompt(instance)
        try:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": SETTINGS.generation_temperature,
                    "max_tokens": SETTINGS.generation_max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            patch = extract_patch(text)
            error = ""
        except Exception as exc:
            text = ""
            patch = ""
            error = repr(exc)

    instance_id = instance["instance_id"]
    return {
        "prediction": {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        },
        "raw": {
            "instance_id": instance_id,
            "output": text,
            "patch": patch,
            "error": error,
        },
    }


def _prompt(instance: dict[str, Any]) -> str:
    hints = instance.get("hints_text") or ""
    hints_block = f"\nHints:\n{hints}\n" if hints else ""
    return (
        f"Repository: {instance.get('repo', '')}\n"
        f"Base commit: {instance.get('base_commit', '')}\n"
        f"Instance ID: {instance.get('instance_id', '')}\n"
        f"{hints_block}\n"
        "Issue:\n"
        f"{instance.get('problem_statement', '')}\n\n"
        "Return a git-style unified diff patch beginning with diff --git whenever possible."
    )


def extract_patch(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for fence in ("```diff", "```patch", "```"):
        if fence in text:
            after = text.split(fence, 1)[1]
            return after.split("```", 1)[0].strip()
    marker = "diff --git "
    if marker in text:
        return marker + text.split(marker, 1)[1].strip()
    return text if text.startswith("--- ") or text.startswith("diff ") else ""

