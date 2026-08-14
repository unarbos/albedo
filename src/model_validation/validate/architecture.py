from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from albedo_config import get_model_validation_settings

config = get_model_validation_settings()


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
    spec = _load_spec(spec_path or config.ARCH_SPEC_PATH)
    cfg = _load_config_json(model_dir)

    for key in spec["forbidden_keys"]:
        if key in cfg:
            return False, f"config.json must not contain {key!r}"

    if spec["architectures"] is not None and cfg.get("architectures") != spec["architectures"]:
        return False, (
            f"architectures mismatch: expected {spec['architectures']!r}, "
            f"got {cfg.get('architectures')!r}"
        )

    text_cfg = cfg.get("text_config") or {}
    for key, want in spec["expected"].items():
        got = cfg[key] if key in cfg else text_cfg.get(key)
        if got != want:
            return False, f"arch key {key!r} mismatch: expected {want!r}, got {got!r}"

    return True, ""
