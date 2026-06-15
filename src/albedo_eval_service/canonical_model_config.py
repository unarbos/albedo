from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


GENESIS_MODEL_CONFIG_REF = (
    "registry.hippius.com/teutonic/albedo-qwen3-4b-genesis@"
    "sha256:3368b0c79b619ed90dc5610c20073cf02c3a93275ebc0c5b94a9d332fea6f606"
)

GENESIS_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

GENESIS_GENERATION_CONFIG: dict[str, Any] = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.51.0",
}

GENESIS_ARCH_SPEC: dict[str, Any] = {
    "architectures": GENESIS_MODEL_CONFIG["architectures"],
    "expected": GENESIS_MODEL_CONFIG,
    "forbidden_keys": ["auto_map", "quantization_config"],
}


def canonical_model_config() -> dict[str, Any]:
    """Return the pinned genesis Hugging Face model config."""
    return deepcopy(GENESIS_MODEL_CONFIG)


def canonical_generation_config() -> dict[str, Any]:
    """Return the pinned genesis Hugging Face generation config."""
    return deepcopy(GENESIS_GENERATION_CONFIG)


def canonical_max_model_len() -> int:
    return int(GENESIS_MODEL_CONFIG["max_position_embeddings"])


def apply_canonical_model_config(model_dir: Path) -> bool:
    """Replace model-supplied config files with the canonical genesis values.

    Contestant artifacts provide weights/tokenizers, but eval must not trust their
    bundled model config. Existing extra keys are retained where vLLM/Hugging Face
    may need them, while explicitly forbidden keys are removed. Generation config
    is fully pinned to genesis.
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

    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text(
        json.dumps(canonical_generation_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True
