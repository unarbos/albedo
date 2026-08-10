"""
Albedo king-of-the-hill GPU evaluation — self-drive entrypoint for KubeTEE.

One Job = one challenger pod (4 GPU / TP=4 by default). Previous-king
generation goes to the always-on `albedo-king` Service when
`ALBEDO_REMOTE_KING_BASE_URL` is set; otherwise falls back to co-located
8-GPU king+challenger (legacy PoC). The pod runs the eval, uploads artifacts
to S3, prints the verdict JSON to stdout, and exits.

The pod runs this entrypoint (main container): drives the whole evaluation
lifecycle and talks HTTP to the shared `albedo-judge-api` Service in
`albedo-poc` (`SELFDRIVE_JUDGE_BASE_URL`), which calls LiteLLM
(`litellm.litellm.svc:4000`) with `ALBEDO_JUDGE_OPENROUTER_API_KEY`.

This file implements *zero* forked business logic. It only:
  1. builds a single EvalRequest from per-job env vars
  2. wires RemoteSettings toward the shared judge-api via an HTTP scorer
  3. waits for the judge-api Service to be ready
  4. calls upstream `albedo_eval_service.remote_worker.RemoteEvalWorker.execute()`
  5. writes the final verdict JSON to stdout (and exits 0 on success)
  6. AFTER the upstream verdict is built + uploaded, writes a companion
     `eval-summary.json` (total LiteLLM spend for this eval_run_id + eval
     wall-clock seconds, excluding dataset prep which ran in the init
     container) and uploads it to S3 next to the verdict. Does NOT modify
     `verdict.json` — that file is upstream-owned. Skips summary upload when
     `fault_code=king_changed` (no registering verdict).

Heavy lifting — model resolution, local challenger vLLM, remote-king HTTP
generate, multi-turn trajectory rollout, S3 upload — lives in
`albedo_eval_service.remote_worker`.
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

# Set SELFDRIVE_MOCK_SCORING=1 to skip both the shared judge-api and
# LiteLLM entirely (infra smoke testing only, not a real eval).
_MOCK_SCORING = os.environ.get("SELFDRIVE_MOCK_SCORING", "").lower() in ("1", "true", "yes")

# Default judge-api base URL — shared Service in albedo-poc (not an in-pod sidecar).
_DEFAULT_JUDGE_BASE_URL = (
    "http://albedo-judge-api.albedo-poc.svc.cluster.local:8091"
)


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


def _load_sample_ids() -> list[str]:
    """Pinned sample IDs for apple-to-apple scoring (CSV env or JSON file).

    Prefer ``ALBEDO_EVAL_SAMPLE_IDS`` (comma-separated). Else if
    ``ALBEDO_EVAL_SAMPLE_IDS_FILE`` is set, load from that path:

    - JSON list of strings
    - JSON object with ``sample_ids``
    - full EvalRequest-shaped JSON with ``dataset.sample_ids`` (e.g. the
      Denrite reference ``compare/reference-ca530856-…/request.json``)

    When non-empty, ``RemoteEvalWorker`` uses these IDs and ignores
    ``sample_seed`` resampling.
    """
    from_csv = _csv("ALBEDO_EVAL_SAMPLE_IDS")
    if from_csv:
        return from_csv
    path = _env("ALBEDO_EVAL_SAMPLE_IDS_FILE")
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        ids = [str(item).strip() for item in payload if str(item).strip()]
    elif isinstance(payload, dict):
        if isinstance(payload.get("sample_ids"), list):
            ids = [str(item).strip() for item in payload["sample_ids"] if str(item).strip()]
        else:
            nested = payload.get("dataset") or {}
            ids = [
                str(item).strip()
                for item in (nested.get("sample_ids") or [])
                if str(item).strip()
            ]
    else:
        raise RuntimeError(
            f"ALBEDO_EVAL_SAMPLE_IDS_FILE={path}: expected JSON list or object"
        )
    if not ids:
        raise RuntimeError(f"ALBEDO_EVAL_SAMPLE_IDS_FILE={path}: empty sample_ids")
    return ids


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

    sample_ids = _load_sample_ids()
    sample_count = (
        len(sample_ids)
        if sample_ids
        else int(_env("ALBEDO_EVAL_SAMPLE_COUNT", "3"))
    )
    if sample_ids:
        logger.info(
            "pinned dataset sample_ids={} (seed resampling disabled; "
            "apple-to-apple vs reference request.json)",
            len(sample_ids),
        )

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
            sample_count=sample_count,
            # max_turns_per_sample is NOT a DatasetConfig field (removed upstream);
            # the worker uses trajectory_assistant_turns from RemoteSettings instead.
            # The ConfigMap's ALBEDO_EVAL_MAX_TURNS_PER_SAMPLE is inert (not read).
            sample_seed=_env("ALBEDO_EVAL_SAMPLE_SEED", "kubetee-poc"),
            sampling_algo=_env("ALBEDO_EVAL_SAMPLING_ALGO", "swe-zero-multi-source-sample-v1"),
            generation_batch_size=_n_samples_per_batch("generation"),
            scoring_batch_size=_n_samples_per_batch("scoring"),
            sample_ids=sample_ids,
        ),
        scoring=ScoringConfig(
            judge_config_hash=_env("ALBEDO_EVAL_JUDGE_CONFIG_HASH", "sha256:replace-with-real-hash"),
            judge_count=int(_env("ALBEDO_EVAL_JUDGE_COUNT", "3")),
            allowed_scores=[0.0, 0.5, 1.0],
        ),
        gpu_request=GpuRequest(
            accelerator=_env("ALBEDO_REMOTE_ACCELERATOR_TYPE", "H200"),
            min_gpus=int(_env("ALBEDO_REMOTE_MIN_GPUS", _env("ALBEDO_REMOTE_GPU_COUNT", "8"))),
            preferred_gpus=int(
                _env("ALBEDO_REMOTE_PREFERRED_GPUS", _env("ALBEDO_REMOTE_GPU_COUNT", "8"))
            ),
            previous_king_gpu_count=int(_env("ALBEDO_REMOTE_PREVIOUS_KING_GPU_COUNT", "4")),
            challenger_gpu_count=int(_env("ALBEDO_REMOTE_CHALLENGER_GPU_COUNT", "4")),
            tensor_parallel_size_per_model=int(
                _env("ALBEDO_REMOTE_TENSOR_PARALLEL_SIZE_PER_MODEL", "4")
            ),
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

    # HTTP scorer against the shared albedo-judge-api Service.
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
# Shared judge readiness + cost fetch
# ---------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    """Bearer header for judge-api when SELFDRIVE_SCORING_AUTH_TOKEN is set.
    Must match ALBEDO_JUDGE_API_AUTH_TOKEN on the judge Deployment."""
    token = _env("SELFDRIVE_SCORING_AUTH_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _wait_judge_ready(url: str, timeout_s: int = 600) -> bool:
    """Poll the shared judge-api Service /ready until it serves.
    /ready is unauthenticated for kubelet probes; still send Bearer when set."""
    deadline = time.time() + timeout_s
    ready_url = url.rstrip("/") + "/ready"
    headers = _auth_headers()
    while time.time() < deadline:
        try:
            r = httpx.get(ready_url, headers=headers, timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def _fetch_eval_cost(judge_base_url: str, eval_run_id: str) -> dict[str, Any]:
    """Fetch accumulated judge cost for this eval_run_id from the shared
    judge-api (`GET /eval-cost/{eval_run_id}`). Concurrent Jobs stay isolated.

    Returns the JSON dict on success, or `{"error": ...}` on any failure.
    Never raises — the cost is a diagnostic add-on, not a gate.
    """
    url = f"{judge_base_url.rstrip('/')}/eval-cost/{eval_run_id}"
    try:
        r = httpx.get(url, headers=_auth_headers(), timeout=10.0)
        if r.status_code != 200:
            return {"error": f"judge /eval-cost returned HTTP {r.status_code}"}
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"judge /eval-cost fetch failed: {exc}"}


def _make_artifacts_public(*, settings: RemoteSettings, artifacts: dict[str, str]) -> None:
    """Set per-object ACL to `public-read` on each uploaded S3 artifact so they
    are fetchable by URL without credentials (matching albedo's production S3
    behavior). The Hippius bucket-level ACL can't be changed by our app key,
    but per-object put-object-acl works.

    Best-effort: individual failures are logged and skipped so one bad object
    doesn't block the rest. Never raises.
    """
    import re

    # Parse s3://bucket/key URIs into (bucket, key) pairs.
    s3_uris: list[tuple[str, str]] = []
    for uri in artifacts.values():
        if not isinstance(uri, str):
            continue
        m = re.match(r"s3://([^/]+)/(.+)", uri)
        if not m:
            continue
        s3_uris.append((m.group(1), m.group(2)))
    if not s3_uris:
        return

    try:
        import boto3

        session_kwargs: dict[str, str] = {}
        if settings.s3_access_key_id:
            session_kwargs["aws_access_key_id"] = settings.s3_access_key_id
        if settings.s3_secret_access_key:
            session_kwargs["aws_secret_access_key"] = settings.s3_secret_access_key
        if settings.s3_session_token:
            session_kwargs["aws_session_token"] = settings.s3_session_token
        if settings.s3_region:
            session_kwargs["region_name"] = settings.s3_region
        client_kwargs: dict[str, str] = {}
        if settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = settings.s3_endpoint_url
        client = boto3.session.Session(**session_kwargs).client("s3", **client_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("make-artifacts-public: boto3 init failed (non-fatal): {}", exc)
        return

    made = 0
    for bucket, key in s3_uris:
        try:
            client.put_object_acl(Bucket=bucket, Key=key, ACL="public-read")
            made += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "make-artifacts-public: put-object-acl failed for s3://{}/{} (non-fatal): {}",
                bucket, key, exc,
            )
    if made:
        logger.info("make-artifacts-public: {} of {} artifacts set to public-read", made, len(s3_uris))


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
        if not _wait_judge_ready(judge_base_url, timeout_s=600):
            logger.error(
                "shared judge-api at {} did not become ready in time. "
                "Check albedo-judge-api Deployment + LiteLLM connectivity.",
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
        # With the shared judge ready, RemoteEvalWorker.__init__(settings) builds
        # the built-in HttpScoringClient pointed at <judge_base_url> via
        # settings.scoring_base_url set above.
        worker = RemoteEvalWorker(settings, artifact_uploader=artifact_uploader)

    # Eval wall-clock starts HERE — after dataset prep (init container) and
    # judge readiness, just before the upstream worker begins the real
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

    # Companion eval-summary.json — skip for king_changed (no registering verdict).
    if str(verdict.get("fault_code") or "") == "king_changed":
        logger.warning(
            "king_changed — skipping eval-summary upload (no registered scored verdict)"
        )
        return 1

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

    # Make all uploaded artifacts publicly readable (match albedo's production
    # behavior — their S3 is public so Denrite's dispatcher + the dashboard can
    # fetch verdicts by URL without creds). The bucket-level ACL on our Hippius
    # bucket can't be changed by our app key (AccessDenied on put-bucket-acl),
    # but per-object put-object-acl works, so we set each artifact to public-read
    # after upload. Best-effort: logged + skipped on failure.
    summary_uri = f"{request.artifact_prefix}/eval-summary.json"
    _make_artifacts_public(
        settings=settings,
        artifacts={**artifacts, "eval_summary": summary_uri},
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
    """Fetch accumulated judge cost for this eval_run_id from the shared
    judge-api (`GET /eval-cost/{eval_run_id}`) and upload `eval-summary.json`
    (cost + timing) next to the verdict. Best-effort — never raises."""
    cost = _fetch_eval_cost(judge_base_url, str(request.eval_run_id))

    # GPU infrastructure cost for the eval window. The node has 8 GPUs at
    # $2.50/hr/Gpu = $20/hr/node (HGX H200 market rate). Priced by the eval
    # wall-clock (worker.execute only — excludes dataset prep + judge
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
        # process started) and judge readiness wait (preceded the timer).
        "timing_scope": "worker.execute only (generation + scoring + upload); "
        "excludes dataset prep + judge readiness",
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
