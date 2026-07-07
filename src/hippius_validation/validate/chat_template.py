from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_CHAT_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"


def check(model_dir: str, files: list[str]) -> tuple[bool, str]:
    root = Path(model_dir)
    problems: list[str] = []

    if "chat_template.jinja" not in files:
        problems.append("missing required chat_template.jinja")
    else:
        try:
            got = hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest()
        except OSError as exc:
            return False, f"could not read chat_template.jinja: {exc}"
        if got != EXPECTED_CHAT_TEMPLATE_SHA256:
            problems.append(
                f"chat_template.jinja sha256 {got} != {EXPECTED_CHAT_TEMPLATE_SHA256}"
            )

    try:
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    except OSError as exc:
        return False, f"could not read tokenizer_config.json: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"invalid tokenizer_config.json: {exc}"

    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template:
        problems.append("tokenizer_config.json missing string chat_template")
    else:
        got = hashlib.sha256(template.encode()).hexdigest()
        if got != EXPECTED_CHAT_TEMPLATE_SHA256:
            problems.append(
                "tokenizer_config.json chat_template sha256 "
                f"{got} != {EXPECTED_CHAT_TEMPLATE_SHA256}"
            )

    return (not problems, "; ".join(problems))
