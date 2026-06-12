#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


NETUID = 97
UID = 0
COLDKEY = "5EUXD91ADceyH7nRWXCqG1wbaCEhsqosT4rjGhwaZDRR4ib6"
HOTKEY = "5EvHrbHz8rT8DrWazxFhzfMsmscFtPE3qhRDeY4ggKZrBcxZ"
MODEL_URI = (
    "registry.hippius.com/teutonic/albedo-qwen3-4b-genesis"
    "@sha256:3368b0c79b619ed90dc5610c20073cf02c3a93275ebc0c5b94a9d332fea6f606"
)
MODEL_HASH = "sha256:3368b0c79b619ed90dc5610c20073cf02c3a93275ebc0c5b94a9d332fea6f606"
REPO = "teutonic/albedo-qwen3-4b-genesis"
DIGEST = MODEL_HASH
BURN_UID = 0


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key] = value
    return env


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def weight_hash(*, reign_version: int) -> str:
    payload = {
        "netuid": NETUID,
        "reign_version": reign_version,
        "uids": [UID],
        "weights": ["1"],
        "policy": {
            "policy": "genesis_bootstrap_v1",
            "burn_uid": BURN_UID,
            "member_count": 1,
            "slot_weight_bps": {"1": 10000},
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the initial Albedo genesis king in the eval Postgres database."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file containing ALBEDO_EVAL_DATABASE_URL.",
    )
    args = parser.parse_args()

    database_url = load_env(Path(args.env_file)).get("ALBEDO_EVAL_DATABASE_URL")
    if not database_url:
        raise SystemExit("ALBEDO_EVAL_DATABASE_URL is not set")

    commit_payload = {
        "version": "v5",
        "repo": REPO,
        "digest": DIGEST,
        "author_hotkey": HOTKEY,
        "bootstrap": "genesis",
    }
    reveal_payload = f"v5|{REPO}|{DIGEST}"
    payload_hash = sha256_text(reveal_payload)
    idempotency_key = f"genesis:{NETUID}:{HOTKEY}:{payload_hash}"
    chain_block_hash = "genesis-bootstrap:" + payload_hash

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('genesis_bootstrap'))")

            active_reign = conn.execute(
                """
                SELECT r.id, r.version, rm.hotkey, rm.uid, rm.model_hash
                FROM reigns r
                LEFT JOIN reign_members rm ON rm.reign_id = r.id AND rm.slot = 1
                WHERE r.state = 'ACTIVE'
                ORDER BY r.version DESC
                LIMIT 1
                """
            ).fetchone()
            if active_reign and (
                active_reign["hotkey"] != HOTKEY
                or active_reign["uid"] != UID
                or active_reign["model_hash"] != MODEL_HASH
            ):
                raise SystemExit(
                    "refusing to create genesis king: a different active reign already exists"
                )

            miner_id = conn.execute(
                """
                INSERT INTO miners (id, hotkey, coldkey, uid, netuid, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (hotkey) DO UPDATE SET
                    coldkey = EXCLUDED.coldkey,
                    uid = EXCLUDED.uid,
                    netuid = EXCLUDED.netuid,
                    updated_at = now()
                RETURNING id
                """,
                (uuid4(), HOTKEY, COLDKEY, UID, NETUID),
            ).fetchone()["id"]

            chain_commit = conn.execute(
                """
                SELECT id
                FROM chain_commits
                WHERE netuid = %s AND hotkey = %s AND payload_hash = %s
                """,
                (NETUID, HOTKEY, payload_hash),
            ).fetchone()
            if chain_commit:
                chain_commit_id = chain_commit["id"]
            else:
                chain_commit_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO chain_commits (
                        id, netuid, block_number, block_hash, uid, hotkey,
                        commit_payload, model_uri, payload_hash
                    )
                    VALUES (%s, %s, 0, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chain_commit_id,
                        NETUID,
                        chain_block_hash,
                        UID,
                        HOTKEY,
                        Jsonb(commit_payload),
                        MODEL_URI,
                        payload_hash,
                    ),
                )

            submission = conn.execute(
                """
                SELECT id
                FROM model_submissions
                WHERE idempotency_key = %s OR model_hash = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (idempotency_key, MODEL_HASH),
            ).fetchone()
            if submission:
                submission_id = submission["id"]
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET miner_id = %s,
                        chain_commit_id = %s,
                        netuid = %s,
                        uid = %s,
                        hotkey = %s,
                        model_uri = %s,
                        commit_hash = %s,
                        model_hash = %s,
                        state = 'COMPLETE_CORONATED',
                        fault_class = NULL,
                        fault_code = NULL,
                        fault_message = NULL,
                        updated_at = now(),
                        finished_at = COALESCE(finished_at, now())
                    WHERE id = %s
                    """,
                    (
                        miner_id,
                        chain_commit_id,
                        NETUID,
                        UID,
                        HOTKEY,
                        MODEL_URI,
                        DIGEST,
                        MODEL_HASH,
                        submission_id,
                    ),
                )
            else:
                submission_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO model_submissions (
                        id, miner_id, chain_commit_id, netuid, uid, hotkey,
                        model_uri, commit_hash, model_hash, state,
                        idempotency_key, finished_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, 'COMPLETE_CORONATED',
                        %s, now()
                    )
                    """,
                    (
                        submission_id,
                        miner_id,
                        chain_commit_id,
                        NETUID,
                        UID,
                        HOTKEY,
                        MODEL_URI,
                        DIGEST,
                        MODEL_HASH,
                        idempotency_key,
                    ),
                )

            conn.execute(
                "UPDATE chain_commits SET submission_id = %s WHERE id = %s",
                (submission_id, chain_commit_id),
            )

            artifact = conn.execute(
                """
                SELECT id
                FROM artifacts
                WHERE submission_id = %s
                  AND artifact_type = 'MODEL_MANIFEST'
                  AND uri = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (submission_id, MODEL_URI),
            ).fetchone()
            if artifact:
                artifact_id = artifact["id"]
            else:
                artifact_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO artifacts (
                        id, submission_id, artifact_type, storage_backend,
                        uri, sha256, content_type
                    )
                    VALUES (%s, %s, 'MODEL_MANIFEST', 'hippius', %s, %s, 'application/vnd.oci.image.manifest.v1+json')
                    """,
                    (artifact_id, submission_id, MODEL_URI, MODEL_HASH.removeprefix("sha256:")),
                )

            if active_reign:
                reign_id = active_reign["id"]
                reign_version = int(active_reign["version"])
                king_version_id = conn.execute(
                    """
                    SELECT king_version_id
                    FROM reign_members
                    WHERE reign_id = %s AND slot = 1
                    """,
                    (reign_id,),
                ).fetchone()["king_version_id"]
            else:
                reign_id = uuid4()
                reign_version = 1
                king_version_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO reigns (id, version, reason, trigger_submission_id, state, activated_at)
                    VALUES (%s, %s, 'GENESIS', %s, 'ACTIVE', now())
                    """,
                    (reign_id, reign_version, submission_id),
                )
                conn.execute(
                    """
                    INSERT INTO king_versions (
                        id, submission_id, model_hash, artifact_id, eval_run_id,
                        version, entered_reign_id, entered_slot, activated_by
                    )
                    VALUES (%s, %s, %s, %s, NULL, 1, %s, 1, 'scripts/create_genesis_king.py')
                    """,
                    (king_version_id, submission_id, MODEL_HASH, artifact_id, reign_id),
                )
                conn.execute(
                    """
                    INSERT INTO reign_members (
                        id, reign_id, slot, king_version_id, submission_id,
                        hotkey, uid, model_hash, weight_bps
                    )
                    VALUES (%s, %s, 1, %s, %s, %s, %s, %s, 10000)
                    """,
                    (
                        uuid4(),
                        reign_id,
                        king_version_id,
                        submission_id,
                        HOTKEY,
                        UID,
                        MODEL_HASH,
                    ),
                )

            policy = {
                "policy": "genesis_bootstrap_v1",
                "burn_uid": BURN_UID,
                "member_count": 1,
                "slot_weight_bps": {"1": 10000},
                "source": "scripts/create_genesis_king.py",
            }
            epoch_weight_hash = weight_hash(reign_version=reign_version)
            conn.execute(
                """
                INSERT INTO weight_epochs (
                    id, netuid, reason, reign_id, state, uids, weights,
                    weight_policy, weight_hash
                )
                VALUES (%s, %s, 'SERVICE_REPLAY', %s, 'PENDING', %s, %s, %s, %s)
                ON CONFLICT (netuid, weight_hash) DO NOTHING
                """,
                (
                    uuid4(),
                    NETUID,
                    reign_id,
                    [UID],
                    [1],
                    Jsonb(policy),
                    epoch_weight_hash,
                ),
            )
            conn.execute(
                """
                INSERT INTO events (id, submission_id, event_type, severity, message, data)
                VALUES (%s, %s, 'genesis_king_bootstrapped', 'INFO', %s, %s)
                """,
                (
                    uuid4(),
                    submission_id,
                    "Genesis king bootstrap completed",
                    Jsonb(
                        {
                            "netuid": NETUID,
                            "uid": UID,
                            "coldkey": COLDKEY,
                            "hotkey": HOTKEY,
                            "model_uri": MODEL_URI,
                            "model_hash": MODEL_HASH,
                            "reign_id": str(reign_id),
                            "reign_version": reign_version,
                            "king_version_id": str(king_version_id),
                            "weight_hash": epoch_weight_hash,
                        }
                    ),
                ),
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "netuid": NETUID,
                "uid": UID,
                "hotkey": HOTKEY,
                "model_uri": MODEL_URI,
                "submission_id": str(submission_id),
                "reign_id": str(reign_id),
                "reign_version": reign_version,
                "king_version_id": str(king_version_id),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
