from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import SETTINGS


def run_harness(*, predictions_path: Path, run_id: str, report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        SETTINGS.dataset_name,
        "--split",
        SETTINGS.split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(SETTINGS.harness_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(SETTINGS.harness_timeout_s),
        "--report_dir",
        str(report_dir),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (report_dir / "harness_stdout.txt").write_text(proc.stdout)
    (report_dir / "harness_stderr.txt").write_text(proc.stderr)

    # SWE-bench writes its top-level report to the process cwd as
    # <model_name>.<run_id>.json; keep a copy beside our other artifacts.
    for candidate in Path.cwd().glob(f"*.{run_id}.json"):
        shutil.copy2(candidate, report_dir / candidate.name)

    summary = collect_summary(report_dir)
    summary.update({
        "harness_returncode": proc.returncode,
        "report_dir": str(report_dir),
    })
    if proc.returncode != 0:
        summary["harness_error"] = "swebench harness exited non-zero"
    return summary


def collect_summary(report_dir: Path) -> dict[str, Any]:
    """Best-effort parser across SWE-bench report layouts."""
    candidates = sorted(report_dir.rglob("*.json"))
    parsed: list[dict[str, Any]] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            parsed.append({"path": str(path), "data": data})

    for item in parsed:
        data = item["data"]
        if "total_instances" in data and "resolved_instances" in data:
            total = int(data.get("total_instances") or 0)
            resolved = int(data.get("resolved_instances") or 0)
            return {
                "summary_path": item["path"],
                "resolved": resolved,
                "total": total,
                "score": (resolved / total) if total else 0.0,
                "submitted": int(data.get("submitted_instances") or total),
                "completed": int(data.get("completed_instances") or 0),
                "unresolved": int(data.get("unresolved_instances") or 0),
                "empty_patches": int(data.get("empty_patch_instances") or 0),
                "errors": int(data.get("error_instances") or 0),
            }
        if "resolved_ids" in data or "unresolved_ids" in data:
            resolved_ids = data.get("resolved_ids") or []
            unresolved_ids = data.get("unresolved_ids") or []
            total = len(resolved_ids) + len(unresolved_ids)
            score = (len(resolved_ids) / total) if total else 0.0
            return {
                "summary_path": item["path"],
                "resolved": len(resolved_ids),
                "total": total,
                "score": score,
            }
        if "resolved" in data and "total" in data:
            total = int(data.get("total") or 0)
            resolved = int(data.get("resolved") or 0)
            return {
                "summary_path": item["path"],
                "resolved": resolved,
                "total": total,
                "score": (resolved / total) if total else 0.0,
            }
        if "resolved" in data and isinstance(data.get("resolved"), list):
            resolved = len(data.get("resolved") or [])
            unresolved = len(data.get("unresolved") or [])
            errors = len(data.get("error") or [])
            total = resolved + unresolved + errors
            return {
                "summary_path": item["path"],
                "resolved": resolved,
                "total": total,
                "score": (resolved / total) if total else 0.0,
                "errors": errors,
            }

    stdout_summary = _parse_stdout_summary(report_dir / "harness_stdout.txt")
    if stdout_summary:
        return stdout_summary

    return {
        "summary_path": "",
        "resolved": None,
        "total": None,
        "score": None,
    }


def _parse_stdout_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    fields = {
        "total_instances": "Total instances",
        "submitted": "Instances submitted",
        "completed": "Instances completed",
        "resolved": "Instances resolved",
        "unresolved": "Instances unresolved",
        "empty_patches": "Instances with empty patches",
        "errors": "Instances with errors",
    }
    out: dict[str, Any] = {"summary_path": str(path)}
    for key, label in fields.items():
        match = re.search(rf"^{re.escape(label)}:\s*(\d+)\s*$", text, re.MULTILINE)
        if match:
            out[key] = int(match.group(1))
    if "resolved" not in out:
        return None
    total = int(out.get("total_instances") or out.get("submitted") or 0)
    out["total"] = total
    out["score"] = (int(out["resolved"]) / total) if total else 0.0
    return out

