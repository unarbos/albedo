from __future__ import annotations

import json
import re
from pathlib import Path

BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
TRAJ_MARKER_RE = re.compile(r"^(REFERENCE STEP \d+:|ENVIRONMENT OBSERVATION:)\s*$", re.M)


def parse_system_user(prompt: str) -> tuple[str, str]:
    blocks = BLOCK_RE.findall(prompt)
    if len(blocks) < 2 or blocks[0][0] != "system" or blocks[1][0] != "user":
        raise ValueError(f"unexpected prompt structure: roles={[r for r, _ in blocks[:3]]}")
    return blocks[0][1].strip(), blocks[1][1].strip()


def parse_trajectory(traj: str) -> list[dict]:
    parts = TRAJ_MARKER_RE.split(traj)
    turns: list[dict] = []
    role = "assistant"
    for i, part in enumerate(parts):
        if i % 2 == 1:
            role = "assistant" if part.startswith("REFERENCE STEP") else "user"
            continue
        content = part.strip()
        if not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def sanitize(model: str) -> str:
    return re.sub(r"\W+", "_", model.split("/", 1)[-1]).strip("_").lower()


def extract_rows(run_dir: Path) -> dict[str, list[dict]]:
    run_id = run_dir.name
    prompts = {}
    with open(run_dir / "generated-samples.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            prompts[rec["sample_id"]] = rec["prompt"]
    by_model: dict[str, list[dict]] = {}
    with open(run_dir / "scoring-results.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            qs = rec.get("question_source") or {}
            ref_model, ref_traj = qs.get("reference_model"), qs.get("reference_trajectory")
            if not ref_model or not ref_traj or rec["sample_id"] not in prompts:
                continue
            system, user = parse_system_user(prompts[rec["sample_id"]])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                *parse_trajectory(ref_traj),
            ]
            by_model.setdefault(sanitize(ref_model), []).append(
                {"sample_id": rec["sample_id"], "messages": messages, "_run_id": run_id}
            )
    return by_model
