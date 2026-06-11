from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GENESIS_MODEL_CONFIG_REF = (
    "registry.hippius.com/teutonic/albedo-qwen3-4b-genesis@"
    "sha256:3368b0c79b619ed90dc5610c20073cf02c3a93275ebc0c5b94a9d332fea6f606"
)

GENESIS_ARCH_SPEC: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "expected": {
        "vocab_size": 151936,
        "model_type": "qwen3",
        "max_position_embeddings": 40960,
        "tie_word_embeddings": True,
        "rope_theta": 1000000,
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "head_dim": 128,
    },
    "forbidden_keys": ["auto_map", "quantization_config"],
}


def canonical_model_config() -> dict[str, Any]:
    """Return the genesis architecture as a Hugging Face config overlay."""
    expected = dict(GENESIS_ARCH_SPEC["expected"])
    return {"architectures": list(GENESIS_ARCH_SPEC["architectures"]), **expected}


def canonical_max_model_len() -> int:
    return int(GENESIS_ARCH_SPEC["expected"]["max_position_embeddings"])


def apply_canonical_model_config(model_dir: Path) -> bool:
    """Replace model-supplied architecture fields with the canonical genesis values.

    Contestant artifacts provide weights/tokenizers, but eval must not trust their
    bundled architecture config. Existing non-architecture keys are retained where
    vLLM/Hugging Face may need them, while explicitly forbidden keys are removed.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model config is not valid JSON: {config_path}") from exc
    if not isinstance(existing, dict):
        raise ValueError(f"model config must be a JSON object: {config_path}")

    forbidden = set(GENESIS_ARCH_SPEC["forbidden_keys"])
    merged = {key: value for key, value in existing.items() if key not in forbidden}
    merged.update(canonical_model_config())
    config_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True
