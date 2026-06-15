from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID, uuid4

import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class WeightMember:
    slot: int
    uid: int
    hotkey: str
    weight_bps: int


@dataclass(frozen=True)
class WeightPayload:
    uids: list[int]
    weights: list[float]
    policy: dict[str, Any]


@dataclass(frozen=True)
class ClaimedWeightEpoch:
    epoch_id: UUID
    transaction_id: UUID
    stage_attempt_id: UUID | None
    reign_id: UUID | None
    trigger_submission_id: UUID | None
    netuid: int
    reason: str
    weight_hash: str
    stored_uids: list[int]
    stored_weights: list[Decimal]
    weight_policy: dict[str, Any]


@dataclass(frozen=True)
class SetWeightsResult:
    success: bool
    message: str = ""
    extrinsic_hash: str | None = None


class ChainClient(Protocol):
    @property
    def block(self) -> int: ...

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]: ...

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> SetWeightsResult: ...


class WeightSetterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    coldkey: str = ""
    hotkey: str = ""
    wallet_path: str = ""
    network: str = "finney"
    netuid: int = 97
    set_rate_blocks: int = 100
    poll_seconds: float = 12.0
    worker_id: str = "weight-setter"
    burn_uid: int = 0
    retry_backoff_seconds: int = 60

    @classmethod
    def from_env(cls) -> "WeightSetterSettings":
        return cls(
            database_url=os.environ.get("ALBEDO_EVAL_DATABASE_URL", ""),
            coldkey=os.environ.get("ALBEDO_WEIGHT_COLDKEY", ""),
            hotkey=os.environ.get("ALBEDO_WEIGHT_HOTKEY", ""),
            wallet_path=os.environ.get("ALBEDO_WEIGHT_WALLET_PATH", ""),
            network=os.environ.get("ALBEDO_WEIGHT_NETWORK")
            or os.environ.get("CHAIN_NETWORK", "finney"),
            netuid=int(
                os.environ.get("ALBEDO_WEIGHT_NETUID")
                or os.environ.get("CHAIN_NETUID")
                or "97"
            ),
            set_rate_blocks=int(os.environ.get("ALBEDO_WEIGHT_SET_RATE_BLOCKS", "100")),
            poll_seconds=float(os.environ.get("ALBEDO_WEIGHT_POLL_SECONDS", "12")),
            worker_id=os.environ.get("ALBEDO_WEIGHT_WORKER_ID", "weight-setter"),
            burn_uid=int(os.environ.get("ALBEDO_WEIGHT_BURN_UID", "0")),
            retry_backoff_seconds=int(
                os.environ.get("ALBEDO_WEIGHT_RETRY_BACKOFF_SECONDS", "60")
            ),
        )


class BittensorChainClient:
    def __init__(self, *, coldkey: str, hotkey: str, network: str, wallet_path: str = ""):
        import bittensor as bt

        wallet_kwargs = {"name": coldkey, "hotkey": hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        self.wallet = bt.Wallet(**wallet_kwargs)
        self.subtensor = bt.Subtensor(network=network)

    @property
    def block(self) -> int:
        return int(getattr(self.subtensor, "block", 0) or 0)

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]:
        metagraph = self.subtensor.metagraph(netuid)
        neurons = getattr(metagraph, "neurons", None)
        if neurons is not None:
            return {int(neuron.uid): str(neuron.hotkey) for neuron in neurons}
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        return {uid: str(hotkey) for uid, hotkey in enumerate(hotkeys)}

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> SetWeightsResult:
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if isinstance(result, tuple):
            success = bool(result[0])
            message = str(result[1] or "") if len(result) > 1 else ""
            return SetWeightsResult(success=success, message=message)
        success = bool(getattr(result, "success", result))
        message = str(getattr(result, "message", "") or "")
        extrinsic_hash = getattr(result, "extrinsic_hash", None) or getattr(result, "hash", None)
        return SetWeightsResult(
            success=success,
            message=message,
            extrinsic_hash=str(extrinsic_hash) if extrinsic_hash else None,
        )


class WeightSetterRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def last_successful_block(self, *, netuid: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT wt.block_number
                FROM weight_transactions wt
                JOIN weight_epochs we ON we.id = wt.weight_epoch_id
                WHERE we.netuid = %s
                  AND wt.state = 'SUCCESS'
                  AND wt.block_number IS NOT NULL
                ORDER BY wt.block_number DESC
                LIMIT 1
                """,
                (netuid,),
            ).fetchone()
        return int(row["block_number"]) if row else None

    def claim_next_epoch(
        self,
        *,
        worker_id: str,
        wallet_hotkey: str,
        subtensor_url: str,
        current_block: int,
        rate_limit_blocks: int,
        netuid: int,
        burn_uid: int,
    ) -> ClaimedWeightEpoch | None:
        with self._connect() as conn:
            with conn.transaction():
                locked = conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtext('weight_setter')) AS locked"
                ).fetchone()
                if not locked or not locked["locked"]:
                    return None

                recent_marker = conn.execute(
                    """
                    SELECT wt.block_number
                    FROM weight_transactions wt
                    JOIN weight_epochs we ON we.id = wt.weight_epoch_id
                    WHERE we.netuid = %s
                      AND (
                        wt.state = 'SUCCESS'
                        OR wt.fault_code = 'weight_set_rate_limited'
                    )
                      AND wt.block_number IS NOT NULL
                    ORDER BY wt.block_number DESC
                    LIMIT 1
                    """,
                    (netuid,),
                ).fetchone()
                if recent_marker:
                    blocks_since = current_block - int(recent_marker["block_number"])
                    if blocks_since < rate_limit_blocks:
                        return None

                epoch = conn.execute(
                    """
                    SELECT we.id, we.netuid, we.reason, we.reign_id, we.uids, we.weights,
                           we.weight_policy, we.weight_hash, r.trigger_submission_id
                    FROM weight_epochs we
                    LEFT JOIN reigns r ON r.id = we.reign_id
                    WHERE (
                        we.state = 'PENDING'
                        OR (
                            we.state = 'FAILED_RETRYABLE'
                            AND we.updated_at <= now() - interval '60 seconds'
                        )
                    )
                    ORDER BY we.created_at ASC
                    FOR UPDATE OF we SKIP LOCKED
                    LIMIT 1
                    """
                ).fetchone()
                if not epoch:
                    epoch = _create_periodic_refresh_epoch_inside_tx(
                        conn,
                        netuid=netuid,
                        current_block=current_block,
                        rate_limit_blocks=rate_limit_blocks,
                        burn_uid=burn_uid,
                    )
                if not epoch:
                    return None

                stage_attempt_id = None
                if epoch["trigger_submission_id"]:
                    stage_attempt_id = uuid4()
                    attempt_number = _next_attempt_number(
                        conn, epoch["trigger_submission_id"], "WEIGHT_SET"
                    )
                    conn.execute(
                        """
                        INSERT INTO stage_attempts (
                            id, submission_id, stage, attempt_number, state,
                            worker_id, started_at, input_snapshot
                        )
                        VALUES (%s, %s, 'WEIGHT_SET', %s, 'RUNNING', %s, now(), %s)
                        """,
                        (
                            stage_attempt_id,
                            epoch["trigger_submission_id"],
                            attempt_number,
                            worker_id,
                            Jsonb(
                                {
                                    "weight_epoch_id": str(epoch["id"]),
                                    "weight_hash": epoch["weight_hash"],
                                }
                            ),
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE model_submissions
                        SET state = 'WEIGHT_SET_RUNNING',
                            fault_class = NULL,
                            fault_code = NULL,
                            fault_message = NULL,
                            updated_at = now()
                        WHERE id = %s
                          AND state IN ('REIGN_SET', 'WEIGHT_SET_RETRYABLE')
                        """,
                        (epoch["trigger_submission_id"],),
                    )

                transaction_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO weight_transactions (
                        id, weight_epoch_id, stage_attempt_id, wallet_hotkey,
                        subtensor_url, state, block_number
                    )
                    VALUES (%s, %s, %s, %s, %s, 'CREATED', %s)
                    """,
                    (
                        transaction_id,
                        epoch["id"],
                        stage_attempt_id,
                        wallet_hotkey,
                        subtensor_url,
                        current_block,
                    ),
                )
                conn.execute(
                    """
                    UPDATE weight_epochs
                    SET state = 'RUNNING',
                        attempt_count = attempt_count + 1,
                        updated_at = now(),
                        last_fault_class = NULL,
                        last_fault_code = NULL
                    WHERE id = %s
                    """,
                    (epoch["id"],),
                )
                if epoch["trigger_submission_id"]:
                    _record_event(
                        conn,
                        submission_id=epoch["trigger_submission_id"],
                        stage_attempt_id=stage_attempt_id,
                        event_type="weight_set_claimed",
                        severity="INFO",
                        message=f"Weight epoch claimed by {worker_id}",
                        data={
                            "weight_epoch_id": str(epoch["id"]),
                            "weight_transaction_id": str(transaction_id),
                            "current_block": current_block,
                        },
                    )

                return ClaimedWeightEpoch(
                    epoch_id=epoch["id"],
                    transaction_id=transaction_id,
                    stage_attempt_id=stage_attempt_id,
                    reign_id=epoch["reign_id"],
                    trigger_submission_id=epoch["trigger_submission_id"],
                    netuid=epoch["netuid"],
                    reason=epoch["reason"],
                    weight_hash=epoch["weight_hash"],
                    stored_uids=list(epoch["uids"]),
                    stored_weights=list(epoch["weights"]),
                    weight_policy=epoch["weight_policy"] or {},
                )

    def active_reign_members(self, *, reign_id: UUID | None) -> list[WeightMember]:
        if reign_id is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT slot, uid, hotkey, weight_bps
                FROM reign_members
                WHERE reign_id = %s
                ORDER BY slot ASC
                """,
                (reign_id,),
            ).fetchall()
        return [
            WeightMember(
                slot=int(row["slot"]),
                uid=int(row["uid"]),
                hotkey=row["hotkey"],
                weight_bps=int(row["weight_bps"]),
            )
            for row in rows
        ]

    def mark_success(
        self,
        *,
        claimed: ClaimedWeightEpoch,
        payload: WeightPayload,
        result: SetWeightsResult,
        block_number: int,
    ) -> None:
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE weight_transactions
                    SET state = 'SUCCESS',
                        extrinsic_hash = %s,
                        block_number = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (result.extrinsic_hash, block_number, claimed.transaction_id),
                )
                conn.execute(
                    """
                    UPDATE weight_epochs
                    SET state = 'SUCCESS',
                        uids = %s,
                        weights = %s,
                        weight_policy = %s,
                        updated_at = now(),
                        succeeded_at = now(),
                        last_fault_class = NULL,
                        last_fault_code = NULL
                    WHERE id = %s
                    """,
                    (
                        payload.uids,
                        [Decimal(str(weight)) for weight in payload.weights],
                        Jsonb(payload.policy),
                        claimed.epoch_id,
                    ),
                )
                if claimed.stage_attempt_id:
                    conn.execute(
                        """
                        UPDATE stage_attempts
                        SET state = 'SUCCEEDED',
                            finished_at = now(),
                            result_summary = %s
                        WHERE id = %s
                        """,
                        (
                            Jsonb(
                                {
                                    "weight_epoch_id": str(claimed.epoch_id),
                                    "weight_transaction_id": str(claimed.transaction_id),
                                    "uids": payload.uids,
                                    "weights": payload.weights,
                                    "block_number": block_number,
                                    "extrinsic_hash": result.extrinsic_hash,
                                }
                            ),
                            claimed.stage_attempt_id,
                        ),
                    )
                if claimed.trigger_submission_id:
                    conn.execute(
                        """
                        UPDATE model_submissions
                        SET state = 'COMPLETE_CORONATED',
                            fault_class = NULL,
                            fault_code = NULL,
                            fault_message = NULL,
                            updated_at = now(),
                            finished_at = now()
                        WHERE id = %s
                        """,
                        (claimed.trigger_submission_id,),
                    )
                    _record_event(
                        conn,
                        submission_id=claimed.trigger_submission_id,
                        stage_attempt_id=claimed.stage_attempt_id,
                        event_type="weight_set_succeeded",
                        severity="INFO",
                        message="Weights submitted successfully",
                        data={
                            "weight_epoch_id": str(claimed.epoch_id),
                            "weight_transaction_id": str(claimed.transaction_id),
                            "uids": payload.uids,
                            "weights": payload.weights,
                            "block_number": block_number,
                            "extrinsic_hash": result.extrinsic_hash,
                        },
                    )

    def mark_failed(
        self,
        *,
        claimed: ClaimedWeightEpoch,
        fault_code: str,
        fault_message: str,
        block_number: int,
    ) -> None:
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE weight_transactions
                    SET state = 'FAILED_RETRYABLE',
                        block_number = %s,
                        fault_class = 'CHAIN_FAULT',
                        fault_code = %s,
                        fault_message = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (block_number, fault_code, fault_message, claimed.transaction_id),
                )
                conn.execute(
                    """
                    UPDATE weight_epochs
                    SET state = 'FAILED_RETRYABLE',
                        last_fault_class = 'CHAIN_FAULT',
                        last_fault_code = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (fault_code, claimed.epoch_id),
                )
                if claimed.stage_attempt_id:
                    conn.execute(
                        """
                        UPDATE stage_attempts
                        SET state = 'FAILED_RETRYABLE',
                            finished_at = now(),
                            fault_class = 'CHAIN_FAULT',
                            fault_code = %s,
                            fault_message = %s
                        WHERE id = %s
                        """,
                        (fault_code, fault_message, claimed.stage_attempt_id),
                    )
                if claimed.trigger_submission_id:
                    conn.execute(
                        """
                        UPDATE model_submissions
                        SET state = 'WEIGHT_SET_RETRYABLE',
                            fault_class = 'CHAIN_FAULT',
                            fault_code = %s,
                            fault_message = %s,
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (fault_code, fault_message, claimed.trigger_submission_id),
                    )
                    _record_event(
                        conn,
                        submission_id=claimed.trigger_submission_id,
                        stage_attempt_id=claimed.stage_attempt_id,
                        event_type="weight_set_failed_retryable",
                        severity="ERROR",
                        message=fault_message,
                        data={
                            "weight_epoch_id": str(claimed.epoch_id),
                            "weight_transaction_id": str(claimed.transaction_id),
                            "fault_class": "CHAIN_FAULT",
                            "fault_code": fault_code,
                            "block_number": block_number,
                        },
                    )


class WeightSetter:
    def __init__(
        self,
        *,
        settings: WeightSetterSettings,
        repository: WeightSetterRepository,
        chain: ChainClient,
    ):
        self.settings = settings
        self.repository = repository
        self.chain = chain

    def run_once(self) -> bool:
        current_block = self.chain.block
        claimed = self.repository.claim_next_epoch(
            worker_id=self.settings.worker_id,
            wallet_hotkey=self.settings.hotkey,
            subtensor_url=self.settings.network,
            current_block=current_block,
            rate_limit_blocks=self.settings.set_rate_blocks,
            netuid=self.settings.netuid,
            burn_uid=self.settings.burn_uid,
        )
        if not claimed:
            return False

        try:
            members = self.repository.active_reign_members(reign_id=claimed.reign_id)
            uid_hotkeys = self.chain.hotkey_by_uid(claimed.netuid)
            payload = build_weight_payload(
                members,
                uid_hotkeys=uid_hotkeys,
                burn_uid=self.settings.burn_uid,
                base_policy=claimed.weight_policy,
            )
            validate_weight_payload(payload)
            result = self.chain.set_weights(
                netuid=claimed.netuid,
                uids=payload.uids,
                weights=payload.weights,
            )
        except Exception as exc:
            self.repository.mark_failed(
                claimed=claimed,
                fault_code="weight_set_exception",
                fault_message=f"{type(exc).__name__}: {exc}",
                block_number=current_block,
            )
            return True

        if result.success:
            self.repository.mark_success(
                claimed=claimed,
                payload=payload,
                result=result,
                block_number=current_block,
            )
        else:
            fault_code = "weight_set_rate_limited" if not result.message else "weight_set_rejected"
            fault_message = result.message or "set_weights returned false without a message"
            self.repository.mark_failed(
                claimed=claimed,
                fault_code=fault_code,
                fault_message=fault_message,
                block_number=current_block,
            )
        return True

    async def run_forever(self) -> None:
        while True:
            did_work = self.run_once()
            if not did_work:
                await asyncio.sleep(self.settings.poll_seconds)


def build_weight_payload(
    members: list[WeightMember],
    *,
    uid_hotkeys: dict[int, str],
    burn_uid: int,
    base_policy: dict[str, Any] | None = None,
) -> WeightPayload:
    policy = dict(base_policy or {})
    policy["burn_uid"] = burn_uid
    policy["deregistered_slots"] = []
    policy["submitted_members"] = []

    if not members:
        policy["empty_reign_burned"] = True
        return WeightPayload(uids=[burn_uid], weights=[1.0], policy=policy)

    by_uid: dict[int, Decimal] = {}
    burned_weight = Decimal("0")
    for member in sorted(members, key=lambda item: item.slot):
        weight = Decimal(member.weight_bps) / Decimal(10000)
        if uid_hotkeys.get(member.uid) == member.hotkey:
            by_uid[member.uid] = by_uid.get(member.uid, Decimal("0")) + weight
            policy["submitted_members"].append(
                {
                    "slot": member.slot,
                    "uid": member.uid,
                    "hotkey": member.hotkey,
                    "weight_bps": member.weight_bps,
                }
            )
        else:
            burned_weight += weight
            policy["deregistered_slots"].append(
                {
                    "slot": member.slot,
                    "expected_uid": member.uid,
                    "expected_hotkey": member.hotkey,
                    "current_hotkey": uid_hotkeys.get(member.uid),
                    "weight_bps": member.weight_bps,
                }
            )

    if burned_weight:
        by_uid[burn_uid] = by_uid.get(burn_uid, Decimal("0")) + burned_weight

    if not by_uid:
        return WeightPayload(uids=[burn_uid], weights=[1.0], policy=policy)

    ordered = sorted(by_uid.items(), key=lambda item: (item[0] != burn_uid, item[0]))
    return WeightPayload(
        uids=[uid for uid, _weight in ordered],
        weights=[float(weight) for _uid, weight in ordered],
        policy=policy,
    )


def periodic_refresh_weight_hash(
    *, netuid: int, reign_id: UUID | None, current_block: int, rate_limit_blocks: int
) -> str:
    refresh_window = current_block // max(rate_limit_blocks, 1)
    payload = {
        "netuid": netuid,
        "reign_id": str(reign_id) if reign_id else None,
        "reason": "PERIODIC_REFRESH",
        "refresh_window": refresh_window,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_weight_payload(payload: WeightPayload) -> None:
    if not payload.uids:
        raise ValueError("weight payload has no uids")
    if len(payload.uids) != len(payload.weights):
        raise ValueError("weight payload uids and weights have different lengths")
    if any(weight <= 0 for weight in payload.weights):
        raise ValueError("weight payload includes a non-positive weight")
    total = sum(payload.weights)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weight payload must sum to 1.0, got {total}")



def _create_periodic_refresh_epoch_inside_tx(
    conn: psycopg.Connection,
    *,
    netuid: int,
    current_block: int,
    rate_limit_blocks: int,
    burn_uid: int,
):
    active_reign = conn.execute(
        """
        SELECT id, version
        FROM reigns
        WHERE state = 'ACTIVE'
        ORDER BY version DESC
        LIMIT 1
        """
    ).fetchone()
    reign_id = active_reign["id"] if active_reign else None
    reign_version = int(active_reign["version"]) if active_reign else None

    members = []
    if reign_id:
        members = conn.execute(
            """
            SELECT slot, uid, hotkey, weight_bps
            FROM reign_members
            WHERE reign_id = %s
            ORDER BY slot ASC
            """,
            (reign_id,),
        ).fetchall()

    if members:
        uids = [int(member["uid"]) for member in members]
        weights = [Decimal(int(member["weight_bps"])) / Decimal(10000) for member in members]
        slot_weight_bps = {str(member["slot"]): int(member["weight_bps"]) for member in members}
    else:
        uids = [burn_uid]
        weights = [Decimal(1)]
        slot_weight_bps = {}

    refresh_window = current_block // max(rate_limit_blocks, 1)
    policy = {
        "policy": "periodic_refresh_v1",
        "burn_uid": burn_uid,
        "current_block": current_block,
        "rate_limit_blocks": rate_limit_blocks,
        "refresh_window": refresh_window,
        "reign_version": reign_version,
        "member_count": len(members),
        "slot_weight_bps": slot_weight_bps,
        "empty_reign_burned": not bool(members),
    }
    weight_hash = periodic_refresh_weight_hash(
        netuid=netuid,
        reign_id=reign_id,
        current_block=current_block,
        rate_limit_blocks=rate_limit_blocks,
    )
    inserted = conn.execute(
        """
        INSERT INTO weight_epochs (
            id, netuid, reason, reign_id, state, uids, weights,
            weight_policy, weight_hash
        )
        VALUES (%s, %s, 'PERIODIC_REFRESH', %s, 'PENDING', %s, %s, %s, %s)
        ON CONFLICT (netuid, weight_hash) DO NOTHING
        RETURNING id, netuid, reason, reign_id, uids, weights, weight_policy, weight_hash,
                  NULL::uuid AS trigger_submission_id
        """,
        (uuid4(), netuid, reign_id, uids, weights, Jsonb(policy), weight_hash),
    ).fetchone()
    if inserted:
        return inserted
    return conn.execute(
        """
        SELECT we.id, we.netuid, we.reason, we.reign_id, we.uids, we.weights,
               we.weight_policy, we.weight_hash, NULL::uuid AS trigger_submission_id
        FROM weight_epochs we
        WHERE we.netuid = %s AND we.weight_hash = %s
        LIMIT 1
        """,
        (netuid, weight_hash),
    ).fetchone()


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
        (
            uuid4(),
            submission_id,
            stage_attempt_id,
            event_type,
            severity,
            message,
            Jsonb(data),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Albedo weight setter.")
    parser.add_argument("--once", action="store_true", help="Submit at most one pending weight epoch.")
    args = parser.parse_args()

    settings = WeightSetterSettings.from_env()
    if not settings.database_url:
        raise RuntimeError("set ALBEDO_EVAL_DATABASE_URL")
    if not settings.coldkey or not settings.hotkey:
        raise RuntimeError("set ALBEDO_WEIGHT_COLDKEY and ALBEDO_WEIGHT_HOTKEY")

    repository = WeightSetterRepository(settings.database_url)
    chain = BittensorChainClient(
        coldkey=settings.coldkey,
        hotkey=settings.hotkey,
        network=settings.network,
        wallet_path=settings.wallet_path,
    )
    setter = WeightSetter(settings=settings, repository=repository, chain=chain)
    if args.once:
        did_work = setter.run_once()
        print(f"weight_setter_did_work={int(did_work)}")
    else:
        asyncio.run(setter.run_forever())


if __name__ == "__main__":
    main()
