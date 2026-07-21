"""Byte-identical check of the model's metadata files against the genesis model repo.

Every config/tokenizer file a challenger ships must be byte-for-byte identical to the
genesis model — a miner only fine-tunes weights, so nothing in the metadata should
change. Two files are handled elsewhere:
- ``chat_template.jinja`` — hashed by ``chat_template.py`` (its dedicated check).
- ``model.safetensors.index.json`` — its ``weight_map`` legitimately differs once a
  checkpoint is re-sharded, so it is validated structurally in ``safetensors_index.py``.

Checking only ``chat_template.jinja`` (and a few config keys) leaves room to smuggle a
tampered tokenizer or poisoned config past validation while every existing check still
passes; a full content hash against the genesis repo closes that gap.

Hashes are pinned to a specific genesis revision so they are reproducible and cannot drift
if the repo is re-published. The genesis metadata is byte-identical to the Qwen base repo
(Qwen/Qwen3.6-35B-A3B); regenerate if the subnet ever pins a new genesis model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Source of truth: https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-genesis
GENESIS_MODEL = "dendriteholdings/albedo-qwen3.6-35b-king-genesis"
GENESIS_REVISION = "d7934c55d650e3a73de3081c11ad2c864009f4b7"

# sha256 of every metadata file that must match the genesis repo byte-for-byte.
# chat_template.jinja (chat_template.py) and model.safetensors.index.json
# (safetensors_index.py, allowed custom) are deliberately absent — checked elsewhere.
GENESIS_SHA256: dict[str, str] = {
    "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "video_preprocessor_config.json":
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(model_dir: str, files: list[str]) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok.

    Every file in GENESIS_SHA256 must be present and hash-match the genesis repo.
    """
    root = Path(model_dir)
    present = set(files)
    problems: list[str] = []

    for name, expected in GENESIS_SHA256.items():
        if name not in present:
            problems.append(f"missing required genesis file {name}")
            continue
        try:
            got = _sha256(root / name)
        except OSError as exc:
            problems.append(f"could not read {name}: {exc}")
            continue
        if got != expected:
            problems.append(
                f"{name} sha256 {got} != {expected} "
                f"(does not match genesis {GENESIS_MODEL})"
            )

    return (not problems, "; ".join(problems))
