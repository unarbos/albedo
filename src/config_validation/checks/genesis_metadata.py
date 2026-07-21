"""Check candidate metadata against the pinned genesis metadata hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path

from config_validation.checks import CheckOutcome

NAME = "genesis_metadata"

GENESIS_MODEL = "dendriteholdings/albedo-qwen3.6-35b-king-genesis"
GENESIS_SHA256: dict[str, str] = {
    "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "video_preprocessor_config.json":
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(model_dir: str, files: list[str]) -> CheckOutcome:
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

    return CheckOutcome(
        name=NAME,
        ok=not problems,
        reason="; ".join(problems),
        details={"checked_files": sorted(GENESIS_SHA256)},
    )
