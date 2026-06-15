"""Sanity dataset sampling - reuses the eval SWE-ZERO manifest sampler + loader (stable side)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_PROMPTS_FILE = Path(__file__).parent / "prompts.json"


@dataclass(frozen=True)
class SanitySample:
    # One sampled prompt to send to the GPU worker (decoupled from the eval EvalSample type).
    sample_id: str
    prompt: str
    target: str | None = None
    messages: list[dict[str, str]] | None = None


def sample_prompts(
    *,
    seed: str,
    n: int = 3,
    max_turns: int = 10,
    manifest_path: str = "",
    manifest_hash: str = "",
    dataset_root: str = "",
) -> list[SanitySample]:
    # Deterministically samples n SWE-ZERO prompts for a challenger; falls back to prompts.json.
    if manifest_path and dataset_root:
        # Heavy deps (pyarrow via remote_dataset) load only when a real manifest is configured.
        from albedo_eval_service.dataset_manifest import load_manifest_file
        from albedo_eval_service.remote_dataset import load_swe_zero_samples
        from albedo_eval_service.sampling import swe_zero_manifest_sample_ids

        manifest = load_manifest_file(manifest_path, expected_sha256=manifest_hash)
        sample_ids = swe_zero_manifest_sample_ids(
            manifest, block_hash=str(seed), sample_count=n, max_turns_per_sample=max_turns
        )
        loaded = load_swe_zero_samples(dataset_root=dataset_root, sample_ids=sample_ids)
        return [SanitySample(s.sample_id, s.prompt, s.target, s.messages) for s in loaded]
    return _fallback_prompts(n)


def _fallback_prompts(n: int) -> list[SanitySample]:
    # Static prompts.json fallback for local/dev when no SWE-ZERO manifest is configured.
    prompts: list[str] = json.loads(_PROMPTS_FILE.read_text())[:n]
    return [
        SanitySample(f"fallback:{i}", prompt, messages=[{"role": "user", "content": prompt}])
        for i, prompt in enumerate(prompts)
    ]
