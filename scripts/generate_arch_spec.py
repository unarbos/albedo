#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from albedo_config.chain_spec import ALL_LOCK_KEYS
from config_validation.pipeline import load_seed_config

_DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "model_validation"
    / "validate"
    / "architecture_spec.json"
)
_FORBIDDEN = ["auto_map", "quantization_config"]


def _collect_expected(cfg: dict) -> dict:
    text_cfg = cfg.get("text_config") or {}
    expected = {}
    for key in ALL_LOCK_KEYS:
        if key in cfg:
            expected[key] = cfg[key]
        elif key in text_cfg:
            expected[key] = text_cfg[key]
            print(f"  note: '{key}' sourced from text_config")
    return expected


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    cfg = load_seed_config()
    spec = {
        "_comment": "Generated from the genesis seed config by scripts/generate_arch_spec.py.",
        "architectures": cfg.get("architectures"),
        "expected": _collect_expected(cfg),
        "forbidden_keys": _FORBIDDEN,
    }
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote arch spec → {out}")
    print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
