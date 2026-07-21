from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


# Canonical model upgraded from the Qwen3-4B genesis to Qwen3.6-35B-A3B
# (qwen3_5_moe). Pinned to the teutonic/qwen3.6-35b-a3b-genesis `genesis` manifest.
GENESIS_MODEL_CONFIG_REF = (
    "registry.hippius.com/teutonic/qwen3.6-35b-a3b-genesis@"
    "sha256:efd5b8d0a1c1f472be56ff919419cdd0561bdecd9013d5c2a96dd0e23e89c165"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_TOKENIZER_DIR = _REPO_ROOT / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
_CANONICAL_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)
_MINER_TOKENIZER_SIDECARS = (
    "special_tokens_map.json",
    "added_tokens.json",
    "configuration.json",
)
_CANONICAL_JSON_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_CACHE_MARKER_FILES = (".albedo-model-cache.json",)

# Full Hugging Face config.json for Qwen/Qwen3.6-35B-A3B. Kept byte-for-byte
# faithful to the published config so apply_canonical_model_config is idempotent
# on the canonical artifact and pins the architecture for contestants.
GENESIS_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5_moe",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}

# Published generation_config.json for Qwen3.6-35B-A3B.
GENESIS_GENERATION_CONFIG: dict[str, Any] = {
    "bos_token_id": 248044,
    "do_sample": True,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}

GENESIS_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}

GENESIS_VIDEO_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor",
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
    # max_position_embeddings lives under text_config for the multimodal MoE
    # config; fall back to the top level for flat configs.
    config = GENESIS_MODEL_CONFIG
    if "max_position_embeddings" in config:
        return int(config["max_position_embeddings"])
    return int(config["text_config"]["max_position_embeddings"])


def apply_canonical_model_config(model_dir: Path) -> bool:
    """Replace model-supplied config files with the canonical genesis values.

    Eval should use miner artifacts only for weights and model.safetensors.index.json.
    All model/tokenizer/processor metadata is pinned to the local genesis copy before
    vLLM or Transformers load the directory.
    """
    config_path = model_dir / "config.json"
    existing: dict[str, Any] = {}
    if config_path.exists():
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
    # Pin the canonical image+video processor configs so vLLM can construct the
    # multimodal model. Text-only eval never uses them, but vLLM/HF refuse to load
    # the multimodal Qwen3.6 architecture without an image+video processor present.
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(GENESIS_PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model_dir / "video_preprocessor_config.json").write_text(
        json.dumps(GENESIS_VIDEO_PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in _CANONICAL_TOKENIZER_FILES:
        source = _CANONICAL_TOKENIZER_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"canonical tokenizer asset missing: {source}")
        shutil.copyfile(source, model_dir / name)

    # These optional files are loaded by Hugging Face tokenizers when present. Do
    # not let a miner-supplied sidecar alter the canonical tokenizer we just pinned.
    for name in _MINER_TOKENIZER_SIDECARS:
        (model_dir / name).unlink(missing_ok=True)

    allowed = set(_CANONICAL_JSON_FILES) | set(_CANONICAL_TOKENIZER_FILES) | set(_CACHE_MARKER_FILES)
    allowed.add("model.safetensors.index.json")
    for path in sorted(model_dir.rglob("*"), reverse=True):
        if path.is_file() and path.name not in allowed and not path.name.endswith(".safetensors"):
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    return True
