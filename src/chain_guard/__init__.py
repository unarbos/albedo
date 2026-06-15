"""chain_guard — hotkey-reuse guard for the chain pipeline.

Library used by chain_reader (not a standalone process). It maintains the ``used_hotkeys``
ledger: hotkeys that may not enter eval. The ledger is seeded at chain_reader startup with every
hotkey that committed before ``CHAIN_START_BLOCK`` (raw, all reveal versions), and added to after
a submission finishes eval. chain_reader checks the ledger before admitting a commit; a reuse is
recorded as a rejected submission and uploaded to S3.
"""
