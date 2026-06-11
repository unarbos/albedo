#!/usr/bin/env python3
"""Regenerate the locked architecture spec from the genesis seed model config.

Downloads the genesis seed config.json (via config_validation) and writes the expected
`architectures` + lock-key values + forbidden keys to the architecture spec file. This makes
the spec authoritative (derived from the current genesis) rather than hand-maintained.

Usage:
    python scripts/generate_arch_spec.py [output_path]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config_validation.config import ALL_LOCK_KEYS
from config_validation.pipeline import load_seed_config

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "hippius_validation" / "validate" / "architecture_spec.json"
_FORBIDDEN = ["auto_map", "quantization_config"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    cfg = load_seed_config()
    spec = {
        "_comment": "Generated from the genesis seed config by scripts/generate_arch_spec.py.",
        "architectures": cfg.get("architectures"),
        "expected": {k: cfg[k] for k in ALL_LOCK_KEYS if k in cfg},
        "forbidden_keys": _FORBIDDEN,
    }
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote arch spec → {out}")
    print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
