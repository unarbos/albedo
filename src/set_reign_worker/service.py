from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import psycopg
from loguru import logger as log
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class ReignMember:
    king_version_id: UUID
    submission_id: UUID
    hotkey: str
    uid: int
    model_hash: str
    previous_slot: int | None = None


@dataclass(frozen=True)
class PlannedReignMember:
    member: ReignMember
    slot: int
    weight_bps: int
    is_challenger: bool = False


class SetReignSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_EVAL_",
        extra="ignore",
    )

    database_url: str = Field(..., description="Postgres DSN")
    lease_seconds: int = 1800
    dispatch_poll_seconds: float = 5.0


class SetReignRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def promote_next_winner(self, *, worker_id: str, lease_seconds: int) -> bool:
        lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)

        with self._connect() as conn:
            with conn.transaction():
                locked = conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtext('reign_promotion')) AS locked"
                ).fetchone()
                if not locked or not locked["locked"]:
                    return False

                submission = conn.execute(
                    """
                    SELECT id, netuid, uid, hotkey, model_hash, model_uri, priority, created_at
                    FROM model_submissions
                    WHERE (
                        state = 'EVAL_WIN'
                        OR (
                            state = 'SET_REIGN_RETRYABLE'
                            AND updated_at <= now()
                                - (LEAST(GREATEST(retry_count, 1), 60) * interval '60 seconds')
                        )
                    )
                    ORDER BY priority ASC, created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """
                ).fetchone()
                if not submission:
                    return False

                attempt_id = uuid4()
                attempt_number = _next_attempt_number(conn, submission["id"], "SET_REIGN")
                conn.execute(
                    """
                    INSERT INTO stage_attempts (
                        id, submission_id, stage, attempt_number, state, worker_id,
                        lease_expires_at, started_at, input_snapshot
                    )
                    VALUES (%s, %s, 'SET_REIGN', %s, 'RUNNING', %s, %s, now(), %s)
                    """,
                    (
                        attempt_id,
                        submission["id"],
                        attempt_number,
                        worker_id,
                        lease_expires_at,
                        Jsonb(
                            {
                                "submission_id": str(submission["id"]),
                                "model_hash": submission["model_hash"],
                                "hotkey": submission["hotkey"],
                                "uid": submission["uid"],
                            }
                        ),
                    ),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = 'SET_REIGN_RUNNING',
                        fault_class = NULL,
                        fault_code = NULL,
                        fault_message = NULL,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (submission["id"],),
                )
                _record_event(
                    conn,
                    submission_id=submission["id"],
                    stage_attempt_id=attempt_id,
                    event_type="set_reign_claimed",
                    severity="INFO",
                    message=f"Set reign claimed by {worker_id}",
                    data={"worker_id": worker_id},
                )

                eval_run = conn.execute(
                    """
                    SELECT id, king_submission_id, king_model_hash, challenger_model_hash,
                           challenger_won, state
                    FROM eval_runs
                    WHERE submission_id = %s
                      AND state = 'SUCCEEDED'
                      AND challenger_won IS TRUE
                    ORDER BY finished_at DESC NULLS LAST, started_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (submission["id"],),
                ).fetchone()
                if not eval_run:
                    log.warning(
                        f"[set-reign] bailing (retryable): no winning eval_run for "
                        f"submission={submission['id']} hotkey={submission['hotkey']}"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="missing_winning_eval_run",
                        fault_message="No successful winning eval run is available for reign promotion",
                    )
                    return True

                if eval_run["challenger_model_hash"] != submission["model_hash"]:
                    log.warning(
                        f"[set-reign] bailing (retryable): challenger model hash mismatch for "
                        f"submission={submission['id']} eval_run={eval_run['id']} "
                        f"eval_hash={eval_run['challenger_model_hash']} "
                        f"submission_hash={submission['model_hash']}"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="challenger_model_hash_mismatch",
                        fault_message="Winning eval challenger model hash does not match submission model hash",
                    )
                    return True

                active_reign = conn.execute(
                    """
                    SELECT id, version
                    FROM reigns
                    WHERE state = 'ACTIVE'
                    ORDER BY version DESC
                    LIMIT 1
                    FOR UPDATE
                    """
                ).fetchone()
                if not active_reign:
                    log.warning(
                        f"[set-reign] bailing (retryable): no ACTIVE reign for winner "
                        f"submission={submission['id']} hotkey={submission['hotkey']}"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="missing_active_reign",
                        fault_message="No active reign exists for winner promotion",
                    )
                    return True

                active_rows = conn.execute(
                    """
                    SELECT rm.slot, rm.king_version_id, rm.submission_id, rm.hotkey,
                           rm.uid, rm.model_hash
                    FROM reign_members rm
                    JOIN king_versions kv ON kv.id = rm.king_version_id
                    WHERE rm.reign_id = %s
                    ORDER BY rm.slot ASC
                    FOR UPDATE OF rm, kv
                    """,
                    (active_reign["id"],),
                ).fetchall()
                active_members = [
                    ReignMember(
                        previous_slot=row["slot"],
                        king_version_id=row["king_version_id"],
                        submission_id=row["submission_id"],
                        hotkey=row["hotkey"],
                        uid=row["uid"],
                        model_hash=row["model_hash"],
                    )
                    for row in active_rows
                ]

                lead = next(
                    (member for member in active_members if member.previous_slot == 1), None
                )
                if not lead:
                    log.warning(
                        f"[set-reign] bailing (retryable): active reign {active_reign['id']} "
                        f"has no slot 1 lead king for submission={submission['id']}"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="missing_active_lead_king",
                        fault_message="Active reign has no slot 1 lead king",
                    )
                    return True

                if (
                    eval_run["king_submission_id"] != lead.submission_id
                    or eval_run["king_model_hash"] != lead.model_hash
                ):
                    log.warning(
                        f"[set-reign] bailing (retryable): stale winning eval for "
                        f"submission={submission['id']} eval_run={eval_run['id']}; did not beat "
                        f"current lead king (lead_submission={lead.submission_id})"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="stale_winning_eval",
                        fault_message="Winning eval did not beat the current active lead king",
                    )
                    return True

                artifact = conn.execute(
                    """
                    SELECT id
                    FROM artifacts
                    WHERE submission_id = %s
                      AND artifact_type = 'MODEL_MANIFEST'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (submission["id"],),
                ).fetchone()
                if not artifact:
                    log.warning(
                        f"[set-reign] bailing (retryable): no MODEL_MANIFEST artifact for "
                        f"submission={submission['id']} hotkey={submission['hotkey']}"
                    )
                    _mark_retryable(
                        conn,
                        submission_id=submission["id"],
                        attempt_id=attempt_id,
                        fault_code="missing_challenger_model_artifact",
                        fault_message="Winning submission has no MODEL_MANIFEST artifact for king version",
                    )
                    return True

                king_version_id = uuid4()
                king_version = _next_version(conn, "king_versions")
                reign_id = uuid4()
                reign_version = _next_version(conn, "reigns")
                challenger_member = ReignMember(
                    king_version_id=king_version_id,
                    submission_id=submission["id"],
                    hotkey=submission["hotkey"],
                    uid=submission["uid"],
                    model_hash=submission["model_hash"],
                )
                planned_members = build_reign_plan(active_members, challenger_member)
                challenger_plan = next(member for member in planned_members if member.is_challenger)
                retired_members = _retired_members(active_members, planned_members)

                conn.execute(
                    """
                    INSERT INTO king_versions (
                        id, submission_id, model_hash, artifact_id, eval_run_id,
                        version, entered_reign_id, entered_slot, activated_by
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        king_version_id,
                        submission["id"],
                        submission["model_hash"],
                        artifact["id"],
                        eval_run["id"],
                        king_version,
                        reign_id,
                        challenger_plan.slot,
                        worker_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE reigns
                    SET state = 'SUPERSEDED'
                    WHERE id = %s
                    """,
                    (active_reign["id"],),
                )
                conn.execute(
                    """
                    INSERT INTO reigns (
                        id, version, reason, trigger_eval_run_id, trigger_submission_id,
                        previous_reign_id, state, activated_at
                    )
                    VALUES (%s, %s, 'CORONATION', %s, %s, %s, 'ACTIVE', now())
                    """,
                    (
                        reign_id,
                        reign_version,
                        eval_run["id"],
                        submission["id"],
                        active_reign["id"],
                    ),
                )
                for planned in planned_members:
                    conn.execute(
                        """
                        INSERT INTO reign_members (
                            id, reign_id, slot, king_version_id, submission_id,
                            hotkey, uid, model_hash, weight_bps
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            uuid4(),
                            reign_id,
                            planned.slot,
                            planned.member.king_version_id,
                            planned.member.submission_id,
                            planned.member.hotkey,
                            planned.member.uid,
                            planned.member.model_hash,
                            planned.weight_bps,
                        ),
                    )

                if retired_members:
                    conn.execute(
                        """
                        UPDATE king_versions
                        SET retired_at = now(),
                            retire_reason = CASE
                                WHEN submission_id = %s THEN 'REPLACED_DUPLICATE'
                                ELSE 'SHIFTED_OUT'
                            END
                        WHERE id = ANY(%s::uuid[])
                          AND retired_at IS NULL
                        """,
                        (
                            submission["id"],
                            [member.king_version_id for member in retired_members],
                        ),
                    )

                weight_policy = _weight_policy(reign_version, planned_members)
                weight_hash = _weight_hash(
                    netuid=submission["netuid"],
                    reign_version=reign_version,
                    planned_members=planned_members,
                    weight_policy=weight_policy,
                )
                weight_uids, weight_values = weight_epoch_payload(planned_members)
                conn.execute(
                    """
                    INSERT INTO weight_epochs (
                        id, netuid, reason, reign_id, state, uids, weights,
                        weight_policy, weight_hash
                    )
                    VALUES (%s, %s, 'CORONATION', %s, 'PENDING', %s, %s, %s, %s)
                    ON CONFLICT (netuid, weight_hash) DO NOTHING
                    """,
                    (
                        uuid4(),
                        submission["netuid"],
                        reign_id,
                        weight_uids,
                        weight_values,
                        Jsonb(weight_policy),
                        weight_hash,
                    ),
                )
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET state = 'SUCCEEDED',
                        finished_at = now(),
                        lease_expires_at = NULL,
                        result_summary = %s
                    WHERE id = %s
                    """,
                    (
                        Jsonb(
                            {
                                "reign_id": str(reign_id),
                                "reign_version": reign_version,
                                "king_version_id": str(king_version_id),
                                "king_version": king_version,
                                "members": _planned_members_json(planned_members),
                                "retired_king_version_ids": [
                                    str(member.king_version_id) for member in retired_members
                                ],
                                "weight_hash": weight_hash,
                            }
                        ),
                        attempt_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = 'REIGN_SET',
                        fault_class = NULL,
                        fault_code = NULL,
                        fault_message = NULL,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (submission["id"],),
                )
                _record_event(
                    conn,
                    submission_id=submission["id"],
                    stage_attempt_id=attempt_id,
                    event_type="reign_set",
                    severity="INFO",
                    message=f"Created active reign version {reign_version}",
                    data={
                        "reign_id": str(reign_id),
                        "reign_version": reign_version,
                        "king_version_id": str(king_version_id),
                        "king_version": king_version,
                        "members": _planned_members_json(planned_members),
                        "retired_king_version_ids": [
                            str(member.king_version_id) for member in retired_members
                        ],
                        "weight_hash": weight_hash,
                    },
                )
                return True


def build_reign_plan(
    active_members: list[ReignMember],
    challenger: ReignMember,
) -> list[PlannedReignMember]:
    active = sorted(active_members, key=lambda member: member.previous_slot or 99)
    filtered = [
        member
        for member in active
        if member.hotkey != challenger.hotkey
        and member.model_hash != challenger.model_hash
        and member.submission_id != challenger.submission_id
    ]
    ordered = [challenger, *filtered[:4]]

    weight_bps = weight_bps_for_member_count(len(ordered))
    planned: list[PlannedReignMember] = []
    for index, member in enumerate(ordered):
        planned.append(
            PlannedReignMember(
                member=member,
                slot=index + 1,
                weight_bps=weight_bps[index],
                is_challenger=member.king_version_id == challenger.king_version_id,
            )
        )
    return planned


def weight_bps_for_member_count(member_count: int) -> list[int]:
    if member_count < 0 or member_count > 5:
        raise ValueError("member_count must be between 0 and 5")
    if member_count == 0:
        return []
    base = 10000 // member_count
    remainder = 10000 % member_count
    return [base + (1 if index < remainder else 0) for index in range(member_count)]


def weight_epoch_payload(
    planned_members: list[PlannedReignMember],
) -> tuple[list[int], list[Decimal]]:
    if not planned_members:
        return [0], [Decimal("1")]
    return (
        [member.member.uid for member in planned_members],
        [Decimal(member.weight_bps) / Decimal(10000) for member in planned_members],
    )


def _retired_members(
    active_members: list[ReignMember],
    planned_members: list[PlannedReignMember],
) -> list[ReignMember]:
    planned_ids = {member.member.king_version_id for member in planned_members}
    return [member for member in active_members if member.king_version_id not in planned_ids]


def _weight_policy(reign_version: int, planned_members: list[PlannedReignMember]) -> dict[str, Any]:
    return {
        "policy": "five_king_genesis_split_v1",
        "reign_version": reign_version,
        "genesis_rule": "new_challenger_slot_1; shift_existing_down; split_weight_evenly",
        "max_slots": 5,
        "member_count": len(planned_members),
        "empty_reign_burn_uid": 0,
        "slot_weight_bps": {str(member.slot): member.weight_bps for member in planned_members},
    }


def _weight_hash(
    *,
    netuid: int,
    reign_version: int,
    planned_members: list[PlannedReignMember],
    weight_policy: dict[str, Any],
) -> str:
    payload = {
        "netuid": netuid,
        "reign_version": reign_version,
        "members": _planned_members_json(planned_members),
        "weight_policy": weight_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _planned_members_json(planned_members: list[PlannedReignMember]) -> list[dict[str, Any]]:
    return [
        {
            "slot": member.slot,
            "king_version_id": str(member.member.king_version_id),
            "submission_id": str(member.member.submission_id),
            "hotkey": member.member.hotkey,
            "uid": member.member.uid,
            "model_hash": member.member.model_hash,
            "weight_bps": member.weight_bps,
            "is_challenger": member.is_challenger,
        }
        for member in planned_members
    ]


def _next_attempt_number(conn: psycopg.Connection, submission_id: UUID, stage: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
        FROM stage_attempts
        WHERE submission_id = %s AND stage = %s
        """,
        (submission_id, stage),
    ).fetchone()
    return int(row["next_attempt"])


def _next_version(conn: psycopg.Connection, table_name: str) -> int:
    row = conn.execute(
        f"SELECT COALESCE(MAX(version), 0) + 1 AS next_version FROM {table_name}"
    ).fetchone()
    return int(row["next_version"])


def _mark_retryable(
    conn: psycopg.Connection,
    *,
    submission_id: UUID,
    attempt_id: UUID,
    fault_code: str,
    fault_message: str,
) -> None:
    conn.execute(
        """
        UPDATE stage_attempts
        SET state = 'FAILED_RETRYABLE',
            finished_at = now(),
            lease_expires_at = NULL,
            fault_class = 'INFRA_FAULT',
            fault_code = %s,
            fault_message = %s
        WHERE id = %s
        """,
        (fault_code, fault_message, attempt_id),
    )
    conn.execute(
        """
        UPDATE model_submissions
        SET state = 'SET_REIGN_RETRYABLE',
            fault_class = 'INFRA_FAULT',
            fault_code = %s,
            fault_message = %s,
            retry_count = retry_count + 1,
            updated_at = now()
        WHERE id = %s
        """,
        (fault_code, fault_message, submission_id),
    )
    _record_event(
        conn,
        submission_id=submission_id,
        stage_attempt_id=attempt_id,
        event_type="set_reign_failed_retryable",
        severity="ERROR",
        message=fault_message,
        data={"fault_class": "INFRA_FAULT", "fault_code": fault_code},
    )


def _record_event(
    conn: psycopg.Connection,
    *,
    submission_id: UUID,
    stage_attempt_id: UUID | None,
    event_type: str,
    severity: str,
    message: str,
    data: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO events (
            id, submission_id, stage_attempt_id, event_type, severity, message, data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (uuid4(), submission_id, stage_attempt_id, event_type, severity, message, Jsonb(data)),
    )


async def run_forever(
    repository: SetReignRepository,
    *,
    worker_id: str,
    lease_seconds: int,
    poll_seconds: float,
) -> None:
    while True:
        try:
            did_work = repository.promote_next_winner(
                worker_id=worker_id, lease_seconds=lease_seconds
            )
        except Exception as exc:
            log.exception(f"[set-reign] run_forever iteration failed, continuing: {exc}")
            did_work = False
        if not did_work:
            await asyncio.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Albedo set reign worker.")
    parser.add_argument("--once", action="store_true", help="Promote at most one winning eval.")
    args = parser.parse_args()

    settings = SetReignSettings()
    worker_id = os.environ.get("ALBEDO_SET_REIGN_WORKER_ID", "set-reign-worker")
    lease_seconds = int(os.environ.get("ALBEDO_SET_REIGN_LEASE_SECONDS", settings.lease_seconds))
    poll_seconds = float(
        os.environ.get("ALBEDO_SET_REIGN_POLL_SECONDS", settings.dispatch_poll_seconds)
    )
    repository = SetReignRepository(settings.database_url)

    if args.once:
        promoted = repository.promote_next_winner(worker_id=worker_id, lease_seconds=lease_seconds)
        print(f"set_reign_promoted={int(promoted)}")
    else:
        asyncio.run(
            run_forever(
                repository,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                poll_seconds=poll_seconds,
            )
        )


if __name__ == "__main__":
    main()
