import pytest

from weight_setter.service import (
    WeightMember,
    WeightPayload,
    build_weight_payload,
    periodic_refresh_weight_hash,
    validate_weight_payload,
)


def test_weight_payload_uses_registered_reign_members():
    payload = build_weight_payload(
        [
            WeightMember(slot=1, uid=10, hotkey="hk-10", weight_bps=5000),
            WeightMember(slot=2, uid=11, hotkey="hk-11", weight_bps=5000),
        ],
        uid_hotkeys={10: "hk-10", 11: "hk-11"},
        burn_uid=0,
        base_policy={"policy": "test"},
    )

    assert payload.uids == [10, 11]
    assert payload.weights == [0.5, 0.5]
    assert payload.policy["deregistered_slots"] == []


def test_weight_payload_burns_deregistered_member_share():
    payload = build_weight_payload(
        [
            WeightMember(slot=1, uid=10, hotkey="hk-10", weight_bps=5000),
            WeightMember(slot=2, uid=11, hotkey="hk-11", weight_bps=5000),
        ],
        uid_hotkeys={10: "hk-10", 11: "different-hotkey"},
        burn_uid=0,
    )

    assert payload.uids == [0, 10]
    assert payload.weights == [0.5, 0.5]
    assert payload.policy["deregistered_slots"] == [
        {
            "slot": 2,
            "expected_uid": 11,
            "expected_hotkey": "hk-11",
            "current_hotkey": "different-hotkey",
            "weight_bps": 5000,
        }
    ]


def test_weight_payload_burns_empty_reign_to_uid_zero():
    payload = build_weight_payload([], uid_hotkeys={}, burn_uid=0)

    assert payload.uids == [0]
    assert payload.weights == [1.0]
    assert payload.policy["empty_reign_burned"] is True


def test_validate_weight_payload_rejects_bad_sum():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_weight_payload(WeightPayload(uids=[1], weights=[0.5], policy={}))



def test_periodic_refresh_hash_is_stable_within_rate_window():
    first = periodic_refresh_weight_hash(
        netuid=97, reign_id=None, current_block=199, rate_limit_blocks=100
    )
    same_window = periodic_refresh_weight_hash(
        netuid=97, reign_id=None, current_block=150, rate_limit_blocks=100
    )
    next_window = periodic_refresh_weight_hash(
        netuid=97, reign_id=None, current_block=200, rate_limit_blocks=100
    )

    assert first == same_window
    assert first != next_window
