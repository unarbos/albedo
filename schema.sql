-- Current Albedo eval database schema.
-- Use this file to initialize a fresh Postgres database for the eval stack.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS chain_commits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    netuid INT NOT NULL,
    block_number BIGINT NOT NULL,
    block_hash TEXT NOT NULL,
    extrinsic_hash TEXT,
    uid INT NOT NULL,
    hotkey TEXT NOT NULL,
    commit_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_uri TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    submission_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS miners (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotkey TEXT NOT NULL UNIQUE,
    coldkey TEXT,
    uid INT,
    netuid INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    miner_id UUID REFERENCES miners(id),
    chain_commit_id UUID NOT NULL REFERENCES chain_commits(id),
    netuid INT NOT NULL,
    uid INT NOT NULL,
    hotkey TEXT NOT NULL,
    model_uri TEXT NOT NULL,
    commit_hash TEXT,
    model_hash TEXT,
    architecture TEXT,
    parameter_count BIGINT,
    state TEXT NOT NULL,
    fault_class TEXT,
    fault_code TEXT,
    fault_message TEXT,
    retry_count INT NOT NULL DEFAULT 0,
    priority INT NOT NULL DEFAULT 100,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT model_submissions_state_ck CHECK (state IN (
        'SUBMITTED',
        'HIPPIUS_RUNNING',
        'HIPPIUS_RETRYABLE',
        'HIPPIUS_VALIDATED',
        'PRE_EVAL_QUEUED',
        'PRE_EVAL_RUNNING',
        'PRE_EVAL_RETRYABLE',
        'PRE_EVAL_PASSED',
        'EVAL_QUEUED',
        'EVAL_RUNNING',
        'EVAL_RETRYABLE',
        'EVAL_WIN',
        'SET_REIGN_RUNNING',
        'SET_REIGN_RETRYABLE',
        'REIGN_SET',
        'WEIGHT_SET_RUNNING',
        'WEIGHT_SET_RETRYABLE',
        'COMPLETE_LOSS',
        'COMPLETE_CORONATED',
        'TERMINAL_INVALID',
        'TERMINAL_INFRA_FAILED'
    )),
    CONSTRAINT model_submissions_fault_class_ck CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    )
);

ALTER TABLE chain_commits
    ADD CONSTRAINT chain_commits_submission_id_fkey
    FOREIGN KEY (submission_id) REFERENCES model_submissions(id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE UNIQUE INDEX IF NOT EXISTS chain_commits_netuid_extrinsic_hash_uidx
    ON chain_commits (netuid, extrinsic_hash)
    WHERE extrinsic_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS chain_commits_netuid_hotkey_payload_hash_uidx
    ON chain_commits (netuid, hotkey, payload_hash);
CREATE INDEX IF NOT EXISTS chain_commits_netuid_block_idx
    ON chain_commits (netuid, block_number);
CREATE INDEX IF NOT EXISTS model_submissions_state_priority_created_idx
    ON model_submissions (state, priority, created_at);
CREATE INDEX IF NOT EXISTS model_submissions_miner_created_idx
    ON model_submissions (miner_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS model_submissions_model_hash_uidx
    ON model_submissions (model_hash)
    WHERE model_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS model_submissions_chain_commit_id_uidx
    ON model_submissions (chain_commit_id);
CREATE INDEX IF NOT EXISTS model_submissions_hotkey_state_idx
    ON model_submissions (hotkey, state, created_at DESC);

CREATE TABLE IF NOT EXISTS stage_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES model_submissions(id),
    stage TEXT NOT NULL,
    attempt_number INT NOT NULL,
    state TEXT NOT NULL,
    worker_id TEXT,
    lease_expires_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    input_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    fault_class TEXT,
    fault_code TEXT,
    fault_message TEXT,
    UNIQUE (submission_id, stage, attempt_number),
    CONSTRAINT stage_attempts_stage_ck CHECK (stage IN ('HIPPIUS', 'PRE_EVAL', 'EVAL', 'SET_REIGN', 'WEIGHT_SET')),
    CONSTRAINT stage_attempts_state_ck CHECK (state IN (
        'PENDING',
        'CLAIMED',
        'RUNNING',
        'SUCCEEDED',
        'FAILED_RETRYABLE',
        'FAILED_TERMINAL',
        'ABANDONED'
    )),
    CONSTRAINT stage_attempts_fault_class_ck CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS stage_attempts_one_active_eval_uidx
    ON stage_attempts (submission_id, stage)
    WHERE stage = 'EVAL' AND state IN ('CLAIMED', 'RUNNING');
CREATE UNIQUE INDEX IF NOT EXISTS stage_attempts_one_active_per_stage_uidx
    ON stage_attempts (submission_id, stage)
    WHERE state IN ('CLAIMED', 'RUNNING');

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES model_submissions(id),
    stage_attempt_id UUID REFERENCES stage_attempts(id),
    artifact_type TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    uri TEXT NOT NULL,
    bucket TEXT,
    object_key TEXT,
    sha256 TEXT,
    size_bytes BIGINT,
    content_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT artifacts_storage_backend_ck CHECK (storage_backend IN ('s3', 'hippius', 'local-cache'))
);

CREATE TABLE IF NOT EXISTS remote_gpu_hosts (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('PRE_EVAL', 'EVAL')),
    base_url TEXT NOT NULL,
    tunnel_name TEXT,
    state TEXT NOT NULL,
    gpu_count INT NOT NULL,
    free_gpu_count INT NOT NULL,
    accelerator_type TEXT,
    capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_heartbeat_at TIMESTAMPTZ,
    last_health JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT remote_gpu_hosts_state_ck CHECK (state IN ('READY', 'DEGRADED', 'DRAINING', 'OFFLINE'))
);

CREATE INDEX IF NOT EXISTS remote_gpu_hosts_role_state_idx
    ON remote_gpu_hosts (role, state, free_gpu_count DESC);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES model_submissions(id),
    stage_attempt_id UUID NOT NULL REFERENCES stage_attempts(id),
    king_submission_id UUID REFERENCES model_submissions(id),
    king_model_hash TEXT NOT NULL,
    challenger_model_hash TEXT NOT NULL,
    remote_host_id TEXT REFERENCES remote_gpu_hosts(id),
    remote_run_id TEXT,
    state TEXT NOT NULL,
    gpu_count INT,
    gpu_ids TEXT[],
    gpu_topology JSONB,
    dataset_version TEXT NOT NULL,
    dataset_manifest_hash TEXT NOT NULL,
    dataset_artifact_id UUID REFERENCES artifacts(id),
    dataset_sample_seed TEXT NOT NULL,
    dataset_sample_ids TEXT[],
    dataset_max_turns_per_sample INT NOT NULL DEFAULT 10,
    dataset_sampling_algo TEXT NOT NULL,
    judge_config_hash TEXT NOT NULL,
    judge_count INT NOT NULL DEFAULT 3,
    sample_count INT,
    generated_sample_count INT NOT NULL DEFAULT 0,
    scored_sample_count INT NOT NULL DEFAULT 0,
    score_challenger NUMERIC,
    score_king NUMERIC,
    win_margin NUMERIC,
    challenger_won BOOLEAN,
    valid_turns INT,
    total_turns INT,
    king_vllm_errors INT NOT NULL DEFAULT 0,
    chal_vllm_errors INT NOT NULL DEFAULT 0,
    judge_errors INT NOT NULL DEFAULT 0,
    fault_class TEXT,
    fault_code TEXT,
    fault_message TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    CONSTRAINT eval_runs_state_ck CHECK (state IN (
        'QUEUED',
        'DISPATCHED',
        'GENERATING',
        'SCORING',
        'VERDICT_READY',
        'SUCCEEDED',
        'FAILED_RETRYABLE',
        'FAILED_TERMINAL'
    )),
    CONSTRAINT eval_runs_fault_class_ck CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS eval_runs_one_active_uidx
    ON eval_runs ((true))
    WHERE state IN ('QUEUED', 'DISPATCHED', 'GENERATING', 'SCORING', 'VERDICT_READY');
CREATE INDEX IF NOT EXISTS eval_runs_submission_started_idx
    ON eval_runs (submission_id, started_at DESC);
CREATE INDEX IF NOT EXISTS eval_runs_remote_run_idx
    ON eval_runs (remote_host_id, remote_run_id)
    WHERE remote_run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS king_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES model_submissions(id),
    model_hash TEXT NOT NULL,
    artifact_id UUID NOT NULL REFERENCES artifacts(id),
    eval_run_id UUID REFERENCES eval_runs(id),
    version BIGINT NOT NULL UNIQUE,
    entered_reign_id UUID,
    entered_slot INT NOT NULL,
    retired_at TIMESTAMPTZ,
    retire_reason TEXT,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_by TEXT NOT NULL,
    CONSTRAINT king_versions_entered_slot_ck CHECK (entered_slot BETWEEN 1 AND 5)
);

CREATE TABLE IF NOT EXISTS reigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version BIGINT NOT NULL UNIQUE,
    reason TEXT NOT NULL,
    trigger_eval_run_id UUID REFERENCES eval_runs(id),
    trigger_submission_id UUID REFERENCES model_submissions(id),
    previous_reign_id UUID REFERENCES reigns(id),
    state TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at TIMESTAMPTZ,
    CONSTRAINT reigns_reason_ck CHECK (reason IN ('CORONATION', 'SERVICE_REPLAY', 'GENESIS')),
    CONSTRAINT reigns_state_ck CHECK (state IN ('ACTIVE', 'SUPERSEDED', 'REPAIRING'))
);

ALTER TABLE king_versions
    ADD CONSTRAINT king_versions_entered_reign_id_fkey
    FOREIGN KEY (entered_reign_id) REFERENCES reigns(id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE UNIQUE INDEX IF NOT EXISTS reigns_one_active_uidx
    ON reigns ((true))
    WHERE state = 'ACTIVE';

CREATE TABLE IF NOT EXISTS reign_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reign_id UUID NOT NULL REFERENCES reigns(id),
    slot INT NOT NULL CHECK (slot BETWEEN 1 AND 5),
    king_version_id UUID NOT NULL REFERENCES king_versions(id),
    submission_id UUID NOT NULL REFERENCES model_submissions(id),
    hotkey TEXT NOT NULL,
    uid INT NOT NULL,
    model_hash TEXT NOT NULL,
    weight_bps INT NOT NULL,
    entered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (reign_id, slot),
    UNIQUE (reign_id, king_version_id)
);

CREATE TABLE IF NOT EXISTS weight_epochs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    netuid INT NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('CORONATION', 'PERIODIC_REFRESH', 'SERVICE_REPLAY')),
    reign_id UUID REFERENCES reigns(id),
    state TEXT NOT NULL CHECK (state IN ('PENDING', 'RUNNING', 'SUBMITTED', 'SUCCESS', 'FAILED_RETRYABLE', 'FAILED_TERMINAL')),
    uids INT[] NOT NULL,
    weights NUMERIC[] NOT NULL,
    weight_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    weight_hash TEXT NOT NULL,
    attempt_count INT NOT NULL DEFAULT 0,
    last_fault_class TEXT CHECK (
        last_fault_class IS NULL OR last_fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    ),
    last_fault_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    succeeded_at TIMESTAMPTZ,
    CHECK (cardinality(uids) = cardinality(weights)),
    UNIQUE (netuid, weight_hash)
);

CREATE INDEX IF NOT EXISTS weight_epochs_state_created_idx
    ON weight_epochs (state, created_at);
CREATE INDEX IF NOT EXISTS weight_epochs_reign_idx
    ON weight_epochs (reign_id, created_at DESC);

CREATE TABLE IF NOT EXISTS weight_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    weight_epoch_id UUID NOT NULL REFERENCES weight_epochs(id),
    stage_attempt_id UUID REFERENCES stage_attempts(id),
    wallet_hotkey TEXT NOT NULL,
    subtensor_url TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('CREATED', 'SUBMITTED', 'SUCCESS', 'FAILED_RETRYABLE', 'FAILED_TERMINAL')),
    extrinsic_hash TEXT,
    block_number BIGINT,
    fault_class TEXT CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    ),
    fault_code TEXT,
    fault_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS weight_transactions_epoch_created_idx
    ON weight_transactions (weight_epoch_id, created_at DESC);
CREATE INDEX IF NOT EXISTS weight_transactions_state_created_idx
    ON weight_transactions (state, created_at);

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES model_submissions(id),
    stage_attempt_id UUID REFERENCES stage_attempts(id),
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT events_severity_ck CHECK (severity IN ('DEBUG', 'INFO', 'WARN', 'ERROR'))
);

CREATE INDEX IF NOT EXISTS events_submission_created_idx
    ON events (submission_id, created_at DESC);
CREATE INDEX IF NOT EXISTS events_stage_attempt_created_idx
    ON events (stage_attempt_id, created_at DESC);
CREATE INDEX IF NOT EXISTS events_type_created_idx
    ON events (event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS sanity_results (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    digest TEXT NOT NULL UNIQUE,
    passed BOOLEAN NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    responses JSONB NOT NULL DEFAULT '[]'::jsonb,
    timing JSONB NOT NULL DEFAULT '{}'::jsonb,
    checked_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS sanity_results_passed_checked_idx
    ON sanity_results (passed, checked_at DESC);
