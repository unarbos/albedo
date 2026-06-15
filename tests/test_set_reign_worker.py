from uuid import uuid4

from set_reign_worker.service import (
    ReignMember,
    build_reign_plan,
    weight_bps_for_member_count,
    weight_epoch_payload,
)


def _member(slot: int, hotkey: str | None = None) -> ReignMember:
    return ReignMember(
        previous_slot=slot,
        king_version_id=uuid4(),
        submission_id=uuid4(),
        hotkey=hotkey or f"hotkey-{slot}",
        uid=slot,
        model_hash=f"sha256:king-{slot}",
    )


def _challenger(hotkey: str = "challenger-hotkey") -> ReignMember:
    return ReignMember(
        king_version_id=uuid4(),
        submission_id=uuid4(),
        hotkey=hotkey,
        uid=99,
        model_hash="sha256:challenger",
    )


def test_genesis_weights_split_missing_slots_evenly():
    assert weight_bps_for_member_count(1) == [10000]
    assert weight_bps_for_member_count(2) == [5000, 5000]
    assert weight_bps_for_member_count(3) == [3334, 3333, 3333]
    assert weight_bps_for_member_count(4) == [2500, 2500, 2500, 2500]
    assert weight_bps_for_member_count(5) == [2000, 2000, 2000, 2000, 2000]


def test_empty_reign_weight_payload_burns_to_uid_zero():
    uids, weights = weight_epoch_payload([])

    assert uids == [0]
    assert [str(weight) for weight in weights] == ["1"]


def test_promotion_puts_challenger_in_slot_one_and_shifts_existing_members():
    active = [_member(1)]
    challenger = _challenger()

    plan = build_reign_plan(active, challenger)

    assert [(member.member.hotkey, member.slot, member.weight_bps) for member in plan] == [
        ("challenger-hotkey", 1, 5000),
        ("hotkey-1", 2, 5000),
    ]
    assert plan[0].is_challenger


def test_full_reign_promotion_shifts_and_drops_old_slot_five():
    active = [_member(slot) for slot in range(1, 6)]
    challenger = _challenger()

    plan = build_reign_plan(active, challenger)

    assert [(member.member.hotkey, member.slot, member.weight_bps) for member in plan] == [
        ("challenger-hotkey", 1, 2000),
        ("hotkey-1", 2, 2000),
        ("hotkey-2", 3, 2000),
        ("hotkey-3", 4, 2000),
        ("hotkey-4", 5, 2000),
    ]
    assert all(member.member.hotkey != "hotkey-5" for member in plan)


def test_duplicate_hotkey_is_moved_instead_of_duplicated():
    active = [_member(1), _member(2, hotkey="challenger-hotkey"), _member(3)]
    challenger = _challenger()

    plan = build_reign_plan(active, challenger)

    assert [(member.member.hotkey, member.slot, member.weight_bps) for member in plan] == [
        ("challenger-hotkey", 1, 3334),
        ("hotkey-1", 2, 3333),
        ("hotkey-3", 3, 3333),
    ]
