"""
Albedo king-of-the-hill GPU evaluation — self-drive entrypoint for KubeTEE.

One Armada job submission = one 8-GPU Kubernetes pod = one evaluation of a
single challenger model against the current king. The pod runs the eval,
uploads artifacts to S3, prints the verdict JSON to stdout, and exits.

The pod runs:
  - this entrypoint (main container): drives the whole evaluation lifecycle
  - judge-api sidecar container in the SAME pod on 127.0.0.1:8091 which calls
    https://llm.kubetee.ai (OpenAI-compatible /v1/chat/completions) with
    ALBEDO_JUDGE_OPENROUTER_API_KEY (a LiteLLM virtual key)

This file implements *zero* forked business logic. It only:
  1. builds a single EvalRequest from per-job env vars
  2. wires RemoteSettings toward the in-pod judge-api via an HTTP scorer
  3. waits for the judge-api sidecar to be ready
  4. calls upstream `albedo_eval_service.remote_worker.RemoteEvalWorker.execute()`
  5. writes the final verdict JSON to stdout (and exits 0 on success)
  6. AFTER the upstream verdict is built + uploaded, writes a companion
     `eval-summary.json` (total LiteLLM spend + eval wall-clock seconds,
     excluding dataset prep which ran in the init container) and uploads
     it to S3 next to the verdict. Does NOT modify `verdict.json` — that
     file is upstream-owned.

Everything heavy — model resolution (HF/Hippius/S3/local), two tensor-parallel
vLLM processes (king on GPUs 0-3, challenger on GPUs 4-7), multi-turn trajectory
rollout with simulated environment observations, S3 artifact upload via
`albedo_eval_service.remote_artifacts`, and the final verdict event — is
upstream code, untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from albedo_eval_service.models import (
    Challenger,
    DatasetConfig,
    EvalRequest,
    GpuRequest,
    PreviousKing,
    ScoringConfig,
)
from albedo_eval_service.remote_artifacts import build_artifact_uploader
from albedo_eval_service.remote_config import RemoteSettings
from albedo_eval_service.remote_scoring import MockScoringClient
from albedo_eval_service.remote_state import RemoteRun
from albedo_eval_service.remote_worker import RemoteEvalWorker

# Set SELFDRIVE_MOCK_SCORING=1 to skip both the in-pod judge-api and
# llm.kubetee.ai entirely (infra smoke testing only, not a real eval).
_MOCK_SCORING = os.environ.get("SELFDRIVE_MOCK_SCORING", "").lower() in ("1", "true", "yes")

# Default judge-api base URL — same pod, port 8091 (matches the Pod spec where
# the sidecar is wired to listen on 0.0.0.0 in the pod network namespace).
_DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8091"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _required(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


def _csv(name: str, default: str = "") -> list[str]:
    value = _env(name, default)
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# EvalRequest + RemoteSettings assembly
# ---------------------------------------------------------------------------


def _load_eval_request() -> EvalRequest:
    """Build the ONE EvalRequest this job will run from per-job env vars the
    dispatcher injects into the Armada job spec for each evaluation.

    Required per-job:
        KING_MODEL_URI / KING_MODEL_HASH      -> previous king to compare against
        CHALLENGER_MODEL_URI / CHALLENGER_MODEL_HASH  -> the miner model being evaluated

    Optional per-job (everything else has a KubeTEE-side default):
        EVAL_RUN_ID / SUBMISSION_ID / KING_VERSION
    """
    king_uri = _required("KING_MODEL_URI")
    king_hash = _required("KING_MODEL_HASH")
    challenger_uri = _required("CHALLENGER_MODEL_URI")
    challenger_hash = _required("CHALLENGER_MODEL_HASH")

    # Where artifacts for the eval go. The uploader expands `eval_run_id`
    # into per-run paths under this prefix.
    artifact_bucket = _env("ALBEDO_EVAL_ARTIFACT_BUCKET", "albedo-artifacts")
    artifact_prefix = _env("ALBEDO_EVAL_ARTIFACT_PREFIX", f"s3://{artifact_bucket}/kubetee-poc")

    # Stable per-run identity — Denrite should set EVAL_RUN_ID + SUBMISSION_ID
    # so artifacts and the final verdict are discoverable deterministically.
    # Fall back to per-pod random UUIDs only if not provided.
    eval_run_id = uuid.UUID(_env("EVAL_RUN_ID") or str(uuid.uuid4()))
    submission_id = uuid.UUID(_env("SUBMISSION_ID") or str(uuid.uuid4()))

    # Expand eval_run_id into the artifact prefix so concurrent jobs never
    # write to the same S3 path and evals from multiple submissions coexist.
    artifact_prefix = f"{artifact_prefix.rstrip('/')}/{eval_run_id}"

    request = EvalRequest(
        eval_run_id=eval_run_id,
        submission_id=submission_id,
        challenger=Challenger(model_uri=challenger_uri, model_hash=challenger_hash),
        previous_king=PreviousKing(
            model_uri=king_uri,
            model_hash=king_hash,
            king_version=int(_env("KING_VERSION", "0")),
        ),
        dataset=DatasetConfig(
            version=_env(
                "ALBEDO_EVAL_DATASET_VERSION", "AlienKevin/SWE-ZERO-12M-trajectories"
            ),
            manifest_uri=_env(
                "ALBEDO_EVAL_DATASET_MANIFEST_URI",
                "s3://albedo-artifacts/datasets/swe-zero/manifest.json",
            ),
            manifest_hash=_env(
                "ALBEDO_EVAL_DATASET_MANIFEST_HASH",
                "982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f",
            ),
            sample_count=int(_env("ALBEDO_EVAL_SAMPLE_COUNT", "3")),
            # max_turns_per_sample is NOT a DatasetConfig field (removed upstream);
            # the worker uses trajectory_assistant_turns from RemoteSettings instead.
            # The ConfigMap's ALBEDO_EVAL_MAX_TURNS_PER_SAMPLE is inert (not read).
            sample_seed=_env("ALBEDO_EVAL_SAMPLE_SEED", "kubetee-poc"),
            sampling_algo=_env("ALBEDO_EVAL_SAMPLING_ALGO", "swe-zero-multi-source-sample-v1"),
            generation_batch_size=_n_samples_per_batch("generation"),
            scoring_batch_size=_n_samples_per_batch("scoring"),
            sample_ids=_csv("ALBEDO_EVAL_SAMPLE_IDS"),
        ),
        scoring=ScoringConfig(
            judge_config_hash=_env("ALBEDO_EVAL_JUDGE_CONFIG_HASH", "sha256:replace-with-real-hash"),
            judge_count=int(_env("ALBEDO_EVAL_JUDGE_COUNT", "3")),
            allowed_scores=[0.0, 0.5, 1.0],
        ),
        gpu_request=GpuRequest(
            accelerator=_env("ALBEDO_REMOTE_ACCELERATOR_TYPE", "H200"),
            min_gpus=8,
            preferred_gpus=8,
            previous_king_gpu_count=4,
            challenger_gpu_count=4,
            tensor_parallel_size_per_model=4,
        ),
        artifact_prefix=artifact_prefix,
    )
    return request


def _n_samples_per_batch(kind: str) -> int:
    """Generation + scoring batch sizes used by DatasetConfig.

    Defaults each to the full per-run sample count (ALBEDO_EVAL_SAMPLE_COUNT)
    so a small PoC run doesn't accidentally parallelize across the default
    batch size of 32 that model.eval author's production code assumes.
    Overridable per-run via ALBEDO_EVAL_GENERATION_BATCH_SIZE and
    ALBEDO_EVAL_SCORING_BATCH_SIZE respectively.

    kind: "generation" or "scoring".
    Returns: int batch size (>= 1).
    """
    sample_count = int(_env("ALBEDO_EVAL_SAMPLE_COUNT", "3"))
    if kind == "generation":
        return int(_env("ALBEDO_EVAL_GENERATION_BATCH_SIZE", str(sample_count)))
    if kind == "scoring":
        return int(_env("ALBEDO_EVAL_SCORING_BATCH_SIZE", str(sample_count)))
    raise ValueError(f"unknown batch kind: {kind}")


def _configure_remote_settings(
    settings: RemoteSettings, *, judge_base_url: str, artifact_spool_dir: str
) -> None:
    """Wire runtime configuration into the upstream RemoteEvalWorker.

    RemoteSettings picks up `ALBEDO_REMOTE_DATASET_ROOT` if provided. We mount
    an empty dir there in the Job spec; albedo's worker only needs the root
    to exist so `load_manifest_file` doesn't fail.
    """

    # Local scratch dirs (transient — tied postgres to the pod lifetime).
    settings.artifact_spool_dir = artifact_spool_dir
    settings.remote_state_dir = _env("ALBEDO_REMOTE_REMOTE_STATE_DIR", "/app/artifacts/state")
    settings.model_cache_dir = _env("ALBEDO_REMOTE_MODEL_CACHE_DIR", "/cache/models")

    # HTTP scorer against the in-pod judge-api sidecar.
    settings.scoring_backend = "http"
    settings.scoring_base_url = judge_base_url
    settings.scoring_auth_token = _env("SELFDRIVE_SCORING_AUTH_TOKEN", "poc")
    settings.scoring_timeout_seconds = float(_env("SELFDRIVE_SCORING_TIMEOUT_SECONDS", "300"))

    # S3 upload config — read by build_artifact_uploader(settings) below. Any
    # S3-API-compatible object store works: Hippius, our R2, AWS S3, MinIO...
    settings.s3_endpoint_url = _env("ALBEDO_REMOTE_S3_ENDPOINT_URL")
    settings.s3_region = _env("ALBEDO_REMOTE_S3_REGION")
    settings.s3_access_key_id = _env("AWS_ACCESS_KEY_ID")
    settings.s3_secret_access_key = _env("AWS_SECRET_ACCESS_KEY")
    settings.s3_session_token = _env("AWS_SESSION_TOKEN")
    settings.upload_artifacts = True


# ---------------------------------------------------------------------------
# Sidecar readiness + main loop
# ---------------------------------------------------------------------------


def _wait_sidecar_ready(url: str, timeout_s: int = 600) -> bool:
    """The judge-api sidecar is started in the SAME pod as this process (see
    the Pod spec in kubetee/deploy/pod-template.yaml). Poll /ready until it
    serves before vLLM is started (tenshy the timing gap)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(url.rstrip("//") + "/ready", timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Post-eval: fetch accumulated cost from the judge-api sidecar
# ---------------------------------------------------------------------------


def _fetch_eval_cost(judge_base_url: str) -> dict[str, Any]:
    """Fetch the accumulated judge cost from the in-pod judge-api sidecar's
    `/eval-cost` endpoint. The sidecar accumulates the `x-litellm-response-cost`
    header value of every judge request it makes (in `judge_openrouter.py`'s
    module-level counter, exposed via this endpoint).

    Returns the JSON dict from the sidecar on success, or `{"error": ...}` on
    any failure. Never raises — the cost is a diagnostic add-on, not a gate.
    """
    url = judge_base_url.rstrip("/") + "/eval-cost"
    try:
        r = httpx.get(url, timeout=10.0)
        if r.status_code != 200:
            return {"error": f"sidecar /eval-cost returned HTTP {r.status_code}"}
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"sidecar /eval-cost fetch failed: {exc}"}


def main() -> int:
    settings = RemoteSettings()

    judge_base_url = _env("SELFDRIVE_JUDGE_BASE_URL", _DEFAULT_JUDGE_BASE_URL).rstrip("/")
    artifact_spool_dir = _env("ALBEDO_REMOTE_ARTIFACT_SPOOL_DIR", "/app/artifacts/spool")
    _configure_remote_settings(
        settings, judge_base_url=judge_base_url, artifact_spool_dir=artifact_spool_dir
    )

    request = _load_eval_request()

    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    logger.info(
        "kubetee-poc start: eval_run={} king={} challenger={} samples={} artifacts={}",
        request.eval_run_id,
        request.previous_king.model_uri,
        request.challenger.model_uri,
        request.dataset.sample_count,
        request.artifact_prefix,
    )

    artifact_uploader = build_artifact_uploader(settings)

    if _MOCK_SCORING:
        logger.warning(
            "SELFDRIVE_MOCK_SCORING is set — using MockScoringClient. "
            "No real judge calls will be made; infra smoke only."
        )
        worker = RemoteEvalWorker(
            settings,
            artifact_uploader=artifact_uploader,
            scorer=MockScoringClient(settings),
        )
    else:
        if not _wait_sidecar_ready(judge_base_url, timeout_s=600):
            logger.error(
                "in-pod judge-api at {} did not become ready in time. "
                "Check ALBEDO_JUDGE_OPENROUTER_API_KEY + llm.kubetee.ai connectivity.",
                judge_base_url,
            )
            print(
                json.dumps(
                    {
                        "eval_run_id": str(request.eval_run_id),
                        "state": "failed",
                        "fault_code": "judge_not_ready",
                    }
                )
            )
            return 1
        # With the sidecar ready, RemoteEvalWorker.__init__(settings) builds
        # the built-in HttpScoringClient pointed at <judge_base_url> via
        # settings.scoring_base_url set above.
        worker = RemoteEvalWorker(settings, artifact_uploader=artifact_uploader)

    # Eval wall-clock starts HERE — after dataset prep (init container) and
    # sidecar readiness, just before the upstream worker begins the real
    # generation + scoring lifecycle.
    eval_start_wall = time.monotonic()
    eval_start_clock = datetime.now(timezone.utc)

    worker.execute(run)

    eval_elapsed_s = round(time.monotonic() - eval_start_wall, 3)
    eval_end_clock = datetime.now(timezone.utc)

    # Extract the final verdict from run events.
    verdict: dict[str, Any] | None = next(
        (e for e in reversed(run.events) if e.get("type") == "verdict"), None
    )
    if verdict is None:
        logger.error("eval run completed without a verdict event — infra failure")
        print(
            json.dumps(
                {
                    "eval_run_id": str(request.eval_run_id),
                    "state": "failed",
                    "fault_code": "no_verdict",
                }
            )
        )
        return 1

    # Verdict always goes to stdout (Armada captures pod logs); artifacts list
    # also goes to stderr for convenience when inspecting locally.
    print(json.dumps(verdict, indent=2))
    artifacts = verdict.get("artifacts") or {}
    if artifacts:
        print("--artifacts--", file=sys.stderr)
        for name, uri in artifacts.items():
            print(f"  {name}: {uri}", file=sys.stderr)

    # Companion eval-summary.json — total LiteLLM spend + wall-clock elapsed.
    # Written to S3 next to the verdict; does NOT modify verdict.json (that
    # file is upstream-owned). Best-effort: any error here is logged and
    # skipped so the eval's exit code reflects only the upstream verdict.
    _write_and_upload_eval_summary(
        request=request,
        artifact_uploader=artifact_uploader,
        spool_dir=artifact_spool_dir,
        eval_start_clock=eval_start_clock,
        eval_end_clock=eval_end_clock,
        eval_elapsed_s=eval_elapsed_s,
        verdict_state=str(verdict.get("state")),
        judge_base_url=judge_base_url,
    )

    return 0 if verdict.get("state") == "succeeded" else 1


def _write_and_upload_eval_summary(
    *,
    request: EvalRequest,
    artifact_uploader: Any,
    spool_dir: str,
    eval_start_clock: datetime,
    eval_end_clock: datetime,
    eval_elapsed_s: float,
    verdict_state: str,
    judge_base_url: str,
) -> None:
    """Fetch the accumulated judge cost from the in-pod judge-api sidecar
    (`/eval-cost` endpoint, backed by `judge_openrouter.py`'s module-level
    cost counter) and upload `eval-summary.json` (cost + timing) next to the
    verdict. Best-effort — never raises."""
    cost = _fetch_eval_cost(judge_base_url)

    # GPU infrastructure cost for the eval window. The node has 8 GPUs at
    # $2.50/hr/Gpu = $20/hr/node (HGX H200 market rate). Priced by the eval
    # wall-clock (worker.execute only — excludes dataset prep + sidecar
    # readiness, which is the window the GPUs are actually doing eval work).
    # Override via env if the rate changes; defaults are the PoC rates.
    gpu_count = int(_env("ALBEDO_REMOTE_GPU_COUNT", "8"))
    gpu_rate_per_hour = float(_env("KUBETEE_GPU_RATE_PER_HOUR", "2.50"))
    elapsed_hours = eval_elapsed_s / 3600.0
    gpu_cost = round(gpu_count * gpu_rate_per_hour * elapsed_hours, 8)
    judge_cost_total = float(cost.get("total_cost") or 0.0) if isinstance(cost, dict) else 0.0

    summary = {
        "type": "eval_summary",
        "eval_run_id": str(request.eval_run_id),
        "submission_id": str(request.submission_id),
        "verdict_state": verdict_state,
        "eval_start_utc": eval_start_clock.isoformat(),
        "eval_end_utc": eval_end_clock.isoformat(),
        "eval_elapsed_seconds": eval_elapsed_s,
        # Excludes dataset prep (ran in the init container before this
        # process started) and sidecar readiness wait (preceded the timer).
        "timing_scope": "worker.execute only (generation + scoring + upload); "
        "excludes dataset prep + sidecar readiness",
        "judge_cost": cost,
        "gpu_cost": {
            "gpu_count": gpu_count,
            "rate_per_gpu_per_hour": gpu_rate_per_hour,
            "rate_per_node_per_hour": round(gpu_count * gpu_rate_per_hour, 2),
            "elapsed_hours": round(elapsed_hours, 6),
            "total_gpu_cost": gpu_cost,
        },
        "total_eval_cost": round(judge_cost_total + gpu_cost, 8),
    }
    try:
        # Write to the same spool dir the upstream artifacts use, then upload.
        import pathlib

        spool_path = pathlib.Path(spool_dir) / str(request.eval_run_id)
        spool_path.mkdir(parents=True, exist_ok=True)
        summary_file = spool_path / "eval-summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        files = {"eval_summary": summary_file}  # Path, not str
        # Reuse the upstream uploader — it already knows the bucket + prefix.
        artifact_uploader.upload_run_artifacts(
            eval_run_id=request.eval_run_id,
            artifact_prefix=request.artifact_prefix,
            files=files,
        )
        logger.info(
            "eval-summary uploaded: elapsed_s={} judge_cost={} gpu_cost={} total={}",
            eval_elapsed_s,
            cost.get("total_cost", "n/a"),
            summary.get("gpu_cost", {}).get("total_gpu_cost", "n/a"),
            summary.get("total_eval_cost", "n/a"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval-summary upload failed (non-fatal): {}", exc)
        # Still print the summary to stdout so it's captured in pod logs.
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
