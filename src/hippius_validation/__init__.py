"""hippius_validation — validate miner models discovered on-chain (Albedo backend).

Reads commits from chain_commits, queues them oldest-block-first, and validates each model
(file manifest → download → architecture → OpenSearch weight-dedup) with durable per-attempt
state in hippius_stage_attempts.
"""
