from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwapEvent:
    uid: int
    old_hotkey: str
    new_hotkey: str
    registration_block: int

    def detail(self) -> dict:
        return {
            "uid": self.uid,
            "old_hotkey": self.old_hotkey,
            "new_hotkey": self.new_hotkey,
            "registration_block": self.registration_block,
        }


def find_swaps(
    prior: dict[int, tuple[str, int | None]],
    current: list[tuple[int, str, int]],
) -> list[SwapEvent]:
    swaps: list[SwapEvent] = []
    for uid, hotkey, reg_block in current:
        prev = prior.get(uid)
        if prev is None:
            continue
        prev_hotkey, prev_reg_block = prev
        if prev_hotkey == hotkey or prev_reg_block is None:
            continue
        if reg_block == prev_reg_block:
            swaps.append(SwapEvent(uid, prev_hotkey, hotkey, reg_block))
    return swaps


def describe(detail: dict) -> str:
    return (
        f"hotkey swap detected: uid {detail['uid']} "
        f"BEFORE hotkey={detail['old_hotkey']} BlockAtRegistration={detail['registration_block']} | "  # noqa: E501
        f"AFTER hotkey={detail['new_hotkey']} BlockAtRegistration={detail['registration_block']} "
        f"<- identical, no re-registration -> swap_hotkey"
    )
