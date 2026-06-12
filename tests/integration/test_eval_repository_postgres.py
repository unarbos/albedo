from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest

from albedo_eval_service.config import Settings
from albedo_eval_service.dispatcher import build_eval_request
from albedo_eval_service.repository import EvalRepository


pytestmark = pytest.mark.integration


def _database_url() -> str:
    database_url = os.environ.get("ALBEDO_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    return database_url


@pytest.fixture()
def db_url() -> str:
    database_url = _database_url()
    schema_path = Path(__file__).resolve().parents[2] / "schema.sql"
    with psycopg.connect(database_url) as conn:
        has_schema = conn.execute("SELECT to_regclass('public.model_submissions')").fetchone()[0]
        if has_schema is None:
            conn.execute(schema_path.read_text(encoding="utf-8"))
    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            for table in (
                "sanity_results",
                "weight_transactions",
                "weight_epochs",
                "events",
                "reign_members",
                "reigns",
                "king_versions",
                "eval_runs",
                "artifacts",
                "stage_attempts",
                "model_submissions",
                "chain_commits",
                "miners",
                "remote_gpu_hosts",
            ):
                conn.execute(f"DELETE FROM {table}")
    return database_url


def test_claim_next_eval_is_sequential_and_creates_attempt(db_url: str):
    repo = EvalRepository(db_url)
    submission_id = _seed_eval_ready_submission(db_url)

    first = repo.claim_next_eval(
        worker_id="worker-a",
        lease_seconds=60,
        request_builder=_request_builder,
    )
    second = repo.claim_next_eval(
        worker_id="worker-b",
        lease_seconds=60,
        request_builder=_request_builder,
    )

    assert first is not None
    assert first.submission_id == submission_id
    assert second is None

    with psycopg.connect(db_url) as conn:
        submission_state = conn.execute(
            "SELECT state FROM model_submissions WHERE id = %s",
            (submission_id,),
        ).fetchone()[0]
        attempt_count = conn.execute(
            "SELECT count(*) FROM stage_attempts WHERE submission_id = %s AND stage = 'EVAL'",
            (submission_id,),
        ).fetchone()[0]
        active_eval_count = conn.execute(
            "SELECT count(*) FROM eval_runs WHERE state = 'DISPATCHED'",
        ).fetchone()[0]

    assert submission_state == "EVAL_RUNNING"
    assert attempt_count == 1
    assert active_eval_count == 1


def test_sweep_abandoned_eval_attempts_returns_submission_to_retryable(db_url: str):
    repo = EvalRepository(db_url)
    submission_id = _seed_eval_ready_submission(db_url)
    claimed = repo.claim_next_eval(worker_id="worker-a", lease_seconds=60, request_builder=_request_builder)
    assert claimed is not None

    with psycopg.connect(db_url) as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE stage_attempts SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
                (claimed.attempt_id,),
            )

    abandoned = repo.sweep_abandoned_eval_attempts(worker_id="sweeper")

    assert abandoned == 1
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            """
            SELECT ms.state, sa.state, er.state, ms.fault_class, ms.fault_code
            FROM model_submissions ms
            JOIN stage_attempts sa ON sa.submission_id = ms.id
            JOIN eval_runs er ON er.stage_attempt_id = sa.id
            WHERE ms.id = %s
            """,
            (submission_id,),
        ).fetchone()

    assert row == (
        "EVAL_RETRYABLE",
        "ABANDONED",
        "FAILED_RETRYABLE",
        "REMOTE_EVAL_FAULT",
        "eval_attempt_lease_expired",
    )


def test_record_remote_progress_and_verdict_artifacts_update_eval_run(db_url: str):
    repo = EvalRepository(db_url)
    submission_id = _seed_eval_ready_submission(db_url)
    claimed = repo.claim_next_eval(worker_id="worker-a", lease_seconds=60, request_builder=_request_builder)
    assert claimed is not None

    repo.record_remote_event(
        submission_id=claimed.submission_id,
        attempt_id=claimed.attempt_id,
        event={
            "type": "generation_started",
            "eval_run_id": str(claimed.eval_run_id),
            "sample_count": 2,
            "gpu_topology": {
                "accelerator": "B200",
                "previous_king": ["0", "1", "2", "3"],
                "challenger": ["4", "5", "6", "7"],
                "tensor_parallel_size_per_model": 4,
            },
        },
    )
    repo.record_remote_event(
        submission_id=claimed.submission_id,
        attempt_id=claimed.attempt_id,
        event={"type": "generation_batch_done", "eval_run_id": str(claimed.eval_run_id), "generated_sample_count": 2},
    )
    repo.record_remote_event(
        submission_id=claimed.submission_id,
        attempt_id=claimed.attempt_id,
        event={"type": "scoring_started", "eval_run_id": str(claimed.eval_run_id)},
    )
    repo.record_remote_event(
        submission_id=claimed.submission_id,
        attempt_id=claimed.attempt_id,
        event={"type": "scoring_batch_done", "eval_run_id": str(claimed.eval_run_id), "scored_sample_count": 2},
    )

    repo.mark_eval_succeeded(
        submission_id=claimed.submission_id,
        attempt_id=claimed.attempt_id,
        eval_run_id=claimed.eval_run_id,
        verdict={
            "type": "verdict",
            "state": "succeeded",
            "challenger_won": True,
            "score_challenger": 0.75,
            "score_king": 0.25,
            "valid_turns": 2,
            "total_turns": 2,
            "generated_sample_count": 2,
            "scored_sample_count": 2,
            "king_vllm_errors": 0,
            "chal_vllm_errors": 0,
            "judge_errors": 0,
            "gpu_topology": {"previous_king": ["0", "1", "2", "3"], "challenger": ["4", "5", "6", "7"]},
            "artifacts": {
                "generated_samples": "s3://albedo-artifacts/submissions/1/eval/2/generated-samples.jsonl",
                "scoring_results": "s3://albedo-artifacts/submissions/1/eval/2/scoring-results.jsonl",
            },
            "artifact_metadata": {
                "generated_samples": {
                    "sha256": "sha256:" + "a" * 64,
                    "size_bytes": 321,
                    "content_type": "application/x-ndjson",
                },
                "scoring_results": {
                    "sha256": "sha256:" + "b" * 64,
                    "size_bytes": 123,
                    "content_type": "application/x-ndjson",
                },
            },
        },
    )

    with psycopg.connect(db_url) as conn:
        eval_row = conn.execute(
            """
            SELECT state, generated_sample_count, scored_sample_count, gpu_ids
            FROM eval_runs
            WHERE id = %s
            """,
            (claimed.eval_run_id,),
        ).fetchone()
        artifact_row = conn.execute(
            """
            SELECT storage_backend, bucket, object_key, sha256, size_bytes, content_type
            FROM artifacts
            WHERE stage_attempt_id = %s AND artifact_type = 'GENERATED_SAMPLES'
            """,
            (claimed.attempt_id,),
        ).fetchone()

    assert eval_row[0] == "SUCCEEDED"
    assert eval_row[1] == 2
    assert eval_row[2] == 2
    assert eval_row[3] == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert artifact_row == (
        "s3",
        "albedo-artifacts",
        "submissions/1/eval/2/generated-samples.jsonl",
        "sha256:" + "a" * 64,
        321,
        "application/x-ndjson",
    )


def _request_builder(submission, king, _remote_host, eval_run_id):
    settings = Settings(
        database_url="postgresql://unused",
        dataset_manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
        judge_config_hash="sha256:judge",
    )
    return build_eval_request(settings, submission, king, eval_run_id)


def _seed_eval_ready_submission(database_url: str) -> UUID:
    submission_id = uuid4()
    chain_commit_id = uuid4()
    miner_id = uuid4()
    king_submission_id = uuid4()
    king_miner_id = uuid4()
    king_chain_commit_id = uuid4()
    king_artifact_id = uuid4()
    king_version_id = uuid4()
    reign_id = uuid4()
    eval_run_id = uuid4()

    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO miners (id, hotkey, uid, netuid)
                VALUES (%s, 'miner-hotkey', 7, 1), (%s, 'king-hotkey', 1, 1)
                """,
                (miner_id, king_miner_id),
            )
            conn.execute(
                """
                INSERT INTO chain_commits (
                    id, netuid, block_number, block_hash, uid, hotkey,
                    commit_payload, model_uri, payload_hash
                )
                VALUES
                    (%s, 1, 100, '0xabc', 7, 'miner-hotkey', '{}'::jsonb, 's3://models/challenger', 'payload-a'),
                    (%s, 1, 99, '0xking', 1, 'king-hotkey', '{}'::jsonb, 's3://models/king', 'payload-king')
                """,
                (chain_commit_id, king_chain_commit_id),
            )
            conn.execute(
                """
                INSERT INTO model_submissions (
                    id, miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,
                    model_hash, state, idempotency_key
                )
                VALUES
                    (%s, %s, %s, 1, 7, 'miner-hotkey', 's3://models/challenger', 'sha256:challenger', 'EVAL_QUEUED', 'idem-a'),
                    (%s, %s, %s, 1, 1, 'king-hotkey', 's3://models/king', 'sha256:king', 'COMPLETE_CORONATED', 'idem-king')
                """,
                (submission_id, miner_id, chain_commit_id, king_submission_id, king_miner_id, king_chain_commit_id),
            )
            conn.execute(
                """
                INSERT INTO remote_gpu_hosts (
                    id, role, base_url, state, gpu_count, free_gpu_count,
                    accelerator_type, capabilities, last_heartbeat_at
                )
                VALUES ('eval-host-1', 'EVAL', 'http://127.0.0.1:8090', 'READY', 8, 8, 'B200', '{}'::jsonb, now())
                """
            )
            conn.execute(
                """
                INSERT INTO artifacts (id, submission_id, artifact_type, storage_backend, uri)
                VALUES (%s, %s, 'MODEL_MANIFEST', 's3', 's3://models/king/manifest.json')
                """,
                (king_artifact_id, king_submission_id),
            )
            conn.execute(
                """
                INSERT INTO stage_attempts (
                    id, submission_id, stage, attempt_number, state, input_snapshot
                )
                VALUES (%s, %s, 'EVAL', 1, 'SUCCEEDED', '{}'::jsonb)
                """,
                (uuid4(), king_submission_id),
            )
            conn.execute(
                """
                INSERT INTO eval_runs (
                    id, submission_id, stage_attempt_id, king_model_hash,
                    challenger_model_hash, state, dataset_version,
                    dataset_manifest_hash, dataset_sample_seed,
                    dataset_sampling_algo, judge_config_hash
                )
                SELECT %s, %s, id, 'sha256:previous', 'sha256:king',
                       'SUCCEEDED', 'dataset', 'manifest', 'seed', 'algo', 'judge'
                FROM stage_attempts
                WHERE submission_id = %s
                LIMIT 1
                """,
                (eval_run_id, king_submission_id, king_submission_id),
            )
            conn.execute(
                """
                INSERT INTO king_versions (
                    id, submission_id, model_hash, artifact_id, eval_run_id,
                    version, entered_slot, activated_by
                )
                VALUES (%s, %s, 'sha256:king', %s, %s, 1, 1, 'test')
                """,
                (king_version_id, king_submission_id, king_artifact_id, eval_run_id),
            )
            conn.execute(
                """
                INSERT INTO reigns (id, version, reason, state, activated_at)
                VALUES (%s, 1, 'GENESIS', 'ACTIVE', now())
                """,
                (reign_id,),
            )
            conn.execute(
                """
                INSERT INTO reign_members (
                    reign_id, slot, king_version_id, submission_id, hotkey, uid,
                    model_hash, weight_bps
                )
                VALUES (%s, 1, %s, %s, 'king-hotkey', 1, 'sha256:king', 2000)
                """,
                (reign_id, king_version_id, king_submission_id),
            )
    return submission_id
