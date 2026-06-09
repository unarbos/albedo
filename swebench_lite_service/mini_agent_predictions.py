from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import SETTINGS


def generate_predictions_with_mini_agent(
    *,
    out_path: Path,
    raw_path: Path,
    model_name: str = "albedo-king",
) -> dict[str, Any]:
    """Run mini-SWE-agent on SWE-bench Lite and convert preds.json to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    mini_dir = out_path.parent / "mini_swe_agent"
    mini_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_path.parent / "mini_swe_agent_config.yaml"
    registry_path = out_path.parent / "litellm_registry.json"
    stdout_path = out_path.parent / "mini_swe_agent_stdout.txt"
    stderr_path = out_path.parent / "mini_swe_agent_stderr.txt"

    config_path.write_text(_config_yaml(model_name))
    registry_path.write_text(json.dumps(_registry(model_name), indent=2, sort_keys=True) + "\n")

    mini_extra = Path(sys.executable).with_name("mini-extra")
    mini_extra_cmd = str(mini_extra) if mini_extra.exists() else "mini-extra"

    cmd = [
        mini_extra_cmd,
        "swebench",
        "--output",
        str(mini_dir),
        "--subset",
        "lite",
        "--split",
        SETTINGS.split,
        "--workers",
        str(SETTINGS.agent_workers),
        "--model",
        f"hosted_vllm/{model_name}",
        "--config",
        "swebench_backticks.yaml",
        "--config",
        str(config_path),
        "--environment-class",
        "docker",
    ]
    if SETTINGS.instance_filter:
        cmd.extend(["--filter", SETTINGS.instance_filter])
    if SETTINGS.limit_instances > 0:
        cmd.extend(["--slice", f"0:{SETTINGS.limit_instances}"])

    env = {
        **os.environ,
        "LITELLM_MODEL_REGISTRY_PATH": str(registry_path),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
        "MSWEA_GLOBAL_COST_LIMIT": os.environ.get("MSWEA_GLOBAL_COST_LIMIT", "0"),
        "MSWEA_COST_TRACKING": os.environ.get("MSWEA_COST_TRACKING", "ignore_errors"),
    }
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    combined_output = f"{proc.stdout}\n{proc.stderr}"

    preds_path = mini_dir / "preds.json"
    preds = _load_preds(preds_path)
    with out_path.open("w") as fh:
        for pred in preds:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")
    raw_path.write_text(json.dumps({
        "runner": "mini-swe-agent",
        "returncode": proc.returncode,
        "command": cmd,
        "mini_output_dir": str(mini_dir),
        "preds_json": str(preds_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "predictions": preds,
    }, indent=2, sort_keys=True) + "\n")

    empty = sum(1 for pred in preds if not pred.get("model_patch"))
    summary: dict[str, Any] = {
        "runner": "mini-swe-agent",
        "instances": len(preds),
        "predictions_path": str(out_path),
        "raw_generations_path": str(raw_path),
        "mini_output_dir": str(mini_dir),
        "mini_preds_path": str(preds_path),
        "mini_stdout_path": str(stdout_path),
        "mini_stderr_path": str(stderr_path),
        "empty_patches": empty,
        "mini_returncode": proc.returncode,
    }
    infra_error = _classify_infra_error(combined_output)
    if infra_error:
        summary["infra_error"] = infra_error
    if proc.returncode != 0:
        summary["mini_error"] = "mini-SWE-agent exited non-zero"
    if infra_error:
        raise RuntimeError(f"mini-SWE-agent infrastructure error: {infra_error}")
    if not preds:
        raise RuntimeError("mini-SWE-agent produced no predictions")
    if all(not pred.get("model_patch") for pred in preds):
        if "returned non-zero exit status" in combined_output or "toomanyrequests" in combined_output:
            raise RuntimeError("mini-SWE-agent environment failed before model call; check mini_swe_agent_stdout/stderr")
        if "tool choice requires" in combined_output:
            raise RuntimeError("mini-SWE-agent requested tool calling unsupported by vLLM; use a text-action config or enable vLLM tool parsing")
        raise RuntimeError("mini-SWE-agent produced only empty patches; check mini_swe_agent_stdout/stderr")
    return summary


def _classify_infra_error(output: str) -> str | None:
    lowered = output.lower()
    if "toomanyrequests" in lowered and "docker" in lowered:
        return "Docker Hub unauthenticated pull rate limit; run docker login on the pod or configure an image mirror/cache"
    # HF unauthenticated messages are warnings, not fatal benchmark failures.
    return None


def _config_yaml(model_name: str) -> str:
    api_base = f"http://{SETTINGS.vllm_host}:{SETTINGS.vllm_port}/v1"
    return f"""model:
  model_name: "hosted_vllm/{model_name}"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "{api_base}"
    drop_params: true
    temperature: {SETTINGS.generation_temperature}
agent:
  cost_limit: 0
"""


def _registry(model_name: str) -> dict[str, Any]:
    entry = {
        "max_tokens": SETTINGS.vllm_max_model_len,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "hosted_vllm",
        "mode": "chat",
    }
    return {
        model_name: entry,
        f"hosted_vllm/{model_name}": entry,
    }


def _load_preds(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        return [dict(item) for item in data.values()]
    raise TypeError(f"Unsupported mini-SWE-agent preds.json shape: {type(data).__name__}")
