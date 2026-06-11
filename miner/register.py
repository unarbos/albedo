"""Register a hotkey on the subnet (recycle / burned registration), miner side.

Mirrors what `btcli subnet register` does, but in-process with logging: shows the recycle
cost, confirms, submits `burned_register`, then reports the assigned UID.
"""
from __future__ import annotations

from loguru import logger


def _neuron_uid(st, ss58: str, netuid: int) -> int | None:
    """UID of ``ss58`` on ``netuid``, or None if not registered."""
    neuron = st.get_neuron_for_pubkey_and_subnet(ss58, netuid)
    if neuron is None or neuron.is_null:
        return None
    return neuron.uid


def register(coldkey: str, hotkey: str, netuid: int, network: str,
             *, assume_yes: bool = False, confirm=None):
    """Register ``hotkey`` on ``netuid`` via recycle. Returns the UID (existing or new), or None
    if aborted. ``confirm(text)->bool`` overrides the CLI [y/N] prompt (the TUI supplies its own).
    """
    import bittensor as bt

    from miner.commit import build_wallet

    wallet = build_wallet(coldkey, hotkey)
    ss58 = wallet.hotkey.ss58_address
    logger.info(f"connecting to {network}…")
    st = bt.Subtensor(network=network)

    uid = _neuron_uid(st, ss58, netuid)
    if uid is not None:
        logger.info(f"hotkey {ss58} already registered on netuid {netuid} (uid {uid}) — nothing to do")
        return uid

    cost = st.recycle(netuid)
    balance = st.get_balance(wallet.coldkeypub.ss58_address)
    logger.info(f"recycle (registration) cost on netuid {netuid}: {cost}")
    logger.info(f"coldkey {coldkey} balance: {balance}")
    if balance is not None and cost is not None and balance < cost:
        raise SystemExit(f"insufficient balance: have {balance}, need {cost} to register on netuid {netuid}")

    text = (
        f"about to register on-chain (recycle):\n"
        f"  coldkey : {coldkey}\n"
        f"  hotkey  : {hotkey} ({ss58})\n"
        f"  netuid  : {netuid}   network: {network}\n"
        f"  cost    : {cost}   (recycled from {coldkey})"
    )
    if not assume_yes:
        ok = confirm(text) if confirm else _cli_confirm(text)
        if not ok:
            logger.info("aborted — not registered")
            return None

    logger.info(f"submitting burned_register for {ss58} on netuid {netuid}…")
    resp = st.burned_register(wallet=wallet, netuid=netuid)
    if not getattr(resp, "success", False):
        raise SystemExit(f"registration failed: {getattr(resp, 'message', None) or getattr(resp, 'error', resp)}")

    uid = _neuron_uid(st, ss58, netuid)
    logger.info(f"registered ✓ hotkey {ss58} on netuid {netuid} — uid {uid}")
    return uid


def _cli_confirm(text: str) -> bool:
    print(text)
    return input("Proceed with registration? [y/N] ").strip().lower() in ("y", "yes")
