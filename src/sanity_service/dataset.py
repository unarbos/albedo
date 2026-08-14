from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_PROMPTS_FILE = Path(__file__).parent / "prompts.json"
_FALLBACK_SYSTEM = (
    "You are a coding agent in a SWE-agent style environment. "
    "Reply with exactly one fenced bash command for the environment to execute. "
    "Do not explain. When the task is complete, run "
    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
)


@dataclass(frozen=True)
class SanitySample:
    prompt: str
    messages: list[dict[str, str]] | None = None
    sample_id: str = ""


def sample_prompts(
    *,
    seed: str,
    n: int = 3,
    manifest_path: str = "",
    manifest_hash: str = "",
    dataset_root: str = "",
) -> list[SanitySample]:
    if manifest_path and dataset_root:
        from albedo_eval_service.remote.dataset import load_manifest_samples
        from albedo_eval_service.shared.dataset_manifest import load_manifest_file
        from albedo_eval_service.shared.sampling import multi_source_manifest_sample_ids

        manifest = load_manifest_file(manifest_path, expected_sha256=manifest_hash)
        ids = multi_source_manifest_sample_ids(manifest, block_hash=str(seed))
        sample_ids = random.Random(str(seed)).sample(ids, min(n, len(ids)))
        loaded = load_manifest_samples(dataset_root=dataset_root, sample_ids=sample_ids)
        return [SanitySample(s.prompt, s.messages, s.sample_id) for s in loaded]
    return _fallback_prompts(n)


def _fallback_prompts(n: int) -> list[SanitySample]:
    prompts: list[str] = json.loads(_PROMPTS_FILE.read_text())[:n]
    return [
        SanitySample(
            prompt,
            messages=[
                {"role": "system", "content": _FALLBACK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            sample_id=f"sanity-fallback:{i}",
        )
        for i, prompt in enumerate(prompts)
    ]
