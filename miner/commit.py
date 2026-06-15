"""Commit the v6 reveal on-chain (miner side) with a preview + Y/N + registration check."""
from __future__ import annotations

import os

from loguru import logger

from config_validation.models import ModelRef


def build_reveal(ref: ModelRef) -> str:
    """The on-chain payload: ``v6|<repo>|<digest>`` (matches chain_reader's v6 parser)."""
    return f"v6|{ref.repo}|{ref.digest}"


def build_wallet(coldkey: str, hotkey: str):
    """Construct a bittensor wallet, honoring ALBEDO_WALLET_PATH if the miner set it."""
    import bittensor as bt

    kwargs = {"name": coldkey, "hotkey": hotkey}
    path = os.environ.get("ALBEDO_WALLET_PATH")
    if path:
        kwargs["path"] = path
    return bt.Wallet(**kwargs)


def registration_check(coldkey: str, hotkey: str, netuid: int, network: str) -> tuple[str, bool]:
    """Return (hotkey_ss58, is_registered_on_netuid)."""
    import bittensor as bt

    wallet = build_wallet(coldkey, hotkey)
    ss58 = wallet.hotkey.ss58_address
    logger.info(f"connecting to {network}…")
    st = bt.Subtensor(network=network)
    logger.info(f"fetching netuid {netuid} metagraph…")
    hotkeys = set(st.metagraph(netuid).hotkeys)
    logger.info(f"metagraph has {len(hotkeys)} hotkeys")
    return ss58, ss58 in hotkeys


def preview(ref: ModelRef, *, ss58: str, coldkey: str, hotkey: str, netuid: int, network: str) -> str:
    """Human-readable summary of exactly what will be committed."""
    return (
        f"about to commit on-chain:\n"
        f"  payload : {build_reveal(ref)}\n"
        f"  coldkey : {coldkey}\n"
        f"  hotkey  : {hotkey} ({ss58})\n"
        f"  netuid  : {netuid}   network: {network}"
    )


def submit(ref: ModelRef, *, coldkey: str, hotkey: str, netuid: int, network: str):
    """Write the reveal on-chain. Caller must have done the registration check + confirm."""
    import bittensor as bt

    wallet = build_wallet(coldkey, hotkey)
    logger.info(f"connecting to {network} to submit reveal…")
    st = bt.Subtensor(network=network)
    logger.info(f"submitting reveal on netuid {netuid}: {build_reveal(ref)}")
    result = st.set_reveal_commitment(
        wallet=wallet, netuid=netuid, data=build_reveal(ref),
        blocks_until_reveal=1,
    )
    logger.info("reveal submitted")
    return result


def commit_reveal(ref: ModelRef, *, coldkey: str, hotkey: str, netuid: int, network: str,
                  assume_yes: bool = False):
    """Full CLI path: registration check → preview → Y/N → submit. Returns the result or None."""
    ss58, registered = registration_check(coldkey, hotkey, netuid, network)
    if not registered:
        raise SystemExit(f"hotkey {ss58} is NOT registered on netuid {netuid} ({network}); "
                         f"register it before committing")
    logger.info(f"hotkey {ss58} is registered on netuid {netuid}")
    print(preview(ref, ss58=ss58, coldkey=coldkey, hotkey=hotkey, netuid=netuid, network=network))
    if not assume_yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            logger.info("aborted — nothing committed")
            return None
    result = submit(ref, coldkey=coldkey, hotkey=hotkey, netuid=netuid, network=network)
    ok = getattr(result, "success", True)
    if ok:
        logger.info("committed")
    else:
        logger.error(f"commit failed: {getattr(result, 'message', result)}")
    return result
