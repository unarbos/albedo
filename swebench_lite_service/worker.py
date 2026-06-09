from __future__ import annotations

import argparse
import asyncio
import logging
import re
import traceback
from pathlib import Path

from albedo.models import ModelRef, materialize_model

from .config import SETTINGS
from .harness import run_harness
from .kings import King, fetch_dashboard, kings_from_dashboard
from .mini_agent_predictions import generate_predictions_with_mini_agent
from .s3_publish import publication_plan
from .state import mark_complete, mark_failed, mark_running, select_next_king
from .vllm_server import VLLMServer

log = logging.getLogger("swebench_lite_service.worker")


async def run_one(*, retry_failed: bool = False) -> bool:
    dashboard = await fetch_dashboard()
    kings = kings_from_dashboard(dashboard)
    king = select_next_king(kings, retry_failed=retry_failed)
    if king is None:
        log.info("no unbenchmarked kings found")
        return False
    await benchmark_king(king)
    return True


async def benchmark_king(king: King) -> None:
    SETTINGS.state_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.runs_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = _run_id(king)
    run_dir = SETTINGS.runs_dir / run_id
    report_dir = SETTINGS.reports_dir / run_id
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = run_dir / "raw_generations.json"

    mark_running(king, run_id=run_id)
    partial = {"run_id": run_id, "run_dir": str(run_dir), "report_dir": str(report_dir)}
    try:
        log.info("materializing king %s", king.key)
        model_dir = await asyncio.to_thread(
            materialize_model,
            ModelRef(repo=king.repo, digest=king.digest),
        )
        partial["model_dir"] = model_dir

        async with VLLMServer(model_dir=model_dir):
            log.info("generating SWE-bench Lite predictions for %s", king.key)
            gen_summary = await asyncio.to_thread(
                generate_predictions_with_mini_agent,
                out_path=predictions_path,
                raw_path=raw_path,
            )
            partial.update(gen_summary)

        log.info("running SWE-bench harness for %s", king.key)
        harness_summary = await asyncio.to_thread(
            run_harness,
            predictions_path=predictions_path,
            run_id=run_id,
            report_dir=report_dir,
        )
        result = {**partial, **harness_summary}
        result["s3"] = publication_plan(king=king, result=result)
        mark_complete(king, result)
        log.info("completed %s: %s/%s", king.key, result.get("resolved"), result.get("total"))
    except Exception as exc:
        log.error("benchmark failed for %s: %s", king.key, exc)
        partial["traceback"] = traceback.format_exc()
        mark_failed(king, repr(exc), partial=partial)
        raise


async def main_loop(*, one_king: bool, retry_failed: bool) -> None:
    ran = await run_one(retry_failed=retry_failed)
    if one_king:
        return
    while True:
        if not ran:
            await asyncio.sleep(SETTINGS.loop_sleep_s)
        ran = await run_one(retry_failed=retry_failed)


def _run_id(king: King) -> str:
    digest_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", king.digest.replace("sha256:", ""))[:16]
    repo_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", king.repo)[-48:]
    reign = king.reign_number if king.reign_number is not None else "unknown"
    return f"{SETTINGS.run_id_prefix}-reign-{reign}-{repo_slug}-{digest_slug}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Albedo kings on SWE-bench Lite")
    parser.add_argument("--one-king", action="store_true", help="benchmark only the next unbenchmarked king")
    parser.add_argument("--retry-failed", action="store_true", help="allow failed kings to be retried")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(main_loop(one_king=args.one_king, retry_failed=args.retry_failed))

