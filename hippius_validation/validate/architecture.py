"""Universal, spec-driven architecture check.

Loads an external architecture spec (path from config.ARCH_SPEC_PATH / ALBEDO_ARCH_SPEC)
that declares the currently-locked architecture, and compares a model's config.json against
it. No model family is hard-coded — swapping the spec file changes the locked architecture
with no code change.

Spec format:
{
  "architectures": ["Qwen3ForCausalLM"],
  "expected": { "model_type": "qwen3", "hidden_size": 2560, ... },
  "forbidden_keys": ["auto_map", "quantization_config"]
}
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from hippius_validation import config


@functools.lru_cache(maxsize=4)
def _load_spec(path: str) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    spec.setdefault("architectures", None)
    spec.setdefault("expected", {})
    spec.setdefault("forbidden_keys", [])
    return spec


def _load_config_json(model_dir: str) -> dict[str, Any]:
    p = Path(model_dir) / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    return json.loads(p.read_text())


def check(model_dir: str, spec_path: str | None = None) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok."""
    spec = _load_spec(spec_path or config.ARCH_SPEC_PATH)
    cfg = _load_config_json(model_dir)

    for key in spec["forbidden_keys"]:
        if key in cfg:
            return False, f"config.json must not contain {key!r}"

    if spec["architectures"] is not None and cfg.get("architectures") != spec["architectures"]:
        return False, (f"architectures mismatch: expected {spec['architectures']!r}, "
                       f"got {cfg.get('architectures')!r}")

    for key, want in spec["expected"].items():
        if cfg.get(key) != want:
            return False, f"arch key {key!r} mismatch: expected {want!r}, got {cfg.get(key)!r}"

    return True, ""
