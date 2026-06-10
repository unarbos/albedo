-- Full Albedo subnet schema completion from Systemdesign.md.
-- Run after 001_eval_service.sql.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE model_submissions
    ADD CONSTRAINT model_submissions_state_ck
    CHECK (state IN (
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
    )) NOT VALID;

ALTER TABLE model_submissions
    ADD CONSTRAINT model_submissions_fault_class_ck
    CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    ) NOT VALID;

ALTER TABLE stage_attempts
    ADD CONSTRAINT stage_attempts_stage_ck
    CHECK (stage IN ('HIPPIUS', 'PRE_EVAL', 'EVAL', 'SET_REIGN', 'WEIGHT_SET')) NOT VALID;

ALTER TABLE stage_attempts
    ADD CONSTRAINT stage_attempts_state_ck
    CHECK (state IN (
        'PENDING',
        'CLAIMED',
        'RUNNING',
        'SUCCEEDED',
        'FAILED_RETRYABLE',
        'FAILED_TERMINAL',
        'ABANDONED'
    )) NOT VALID;

ALTER TABLE stage_attempts
    ADD CONSTRAINT stage_attempts_fault_class_ck
    CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    ) NOT VALID;

CREATE UNIQUE INDEX IF NOT EXISTS stage_attempts_one_active_per_stage_uidx
    ON stage_attempts (submission_id, stage)
    WHERE state IN ('CLAIMED', 'RUNNING');

ALTER TABLE artifacts
    ADD CONSTRAINT artifacts_storage_backend_ck
    CHECK (storage_backend IN ('s3', 'hippius', 'local-cache')) NOT VALID;

ALTER TABLE remote_gpu_hosts
    ADD CONSTRAINT remote_gpu_hosts_state_ck
    CHECK (state IN ('READY', 'DEGRADED', 'DRAINING', 'OFFLINE')) NOT VALID;

ALTER TABLE eval_runs
    ADD CONSTRAINT eval_runs_state_ck
    CHECK (state IN (
        'QUEUED',
        'DISPATCHED',
        'GENERATING',
        'SCORING',
        'VERDICT_READY',
        'SUCCEEDED',
        'FAILED_RETRYABLE',
        'FAILED_TERMINAL'
    )) NOT VALID;

ALTER TABLE eval_runs
    ADD CONSTRAINT eval_runs_fault_class_ck
    CHECK (
        fault_class IS NULL OR fault_class IN (
            'MINER_FAULT',
            'INFRA_FAULT',
            'REMOTE_EVAL_FAULT',
            'CHAIN_FAULT',
            'PROVIDER_FAULT',
            'UNKNOWN_FAULT'
        )
    ) NOT VALID;

CREATE INDEX IF NOT EXISTS eval_runs_submission_started_idx
    ON eval_runs (submission_id, started_at DESC);
CREATE INDEX IF NOT EXISTS eval_runs_remote_run_idx
    ON eval_runs (remote_host_id, remote_run_id)
    WHERE remote_run_id IS NOT NULL;

ALTER TABLE king_versions
    ADD CONSTRAINT king_versions_entered_slot_ck
    CHECK (entered_slot BETWEEN 1 AND 5) NOT VALID;

ALTER TABLE reigns
    ADD CONSTRAINT reigns_reason_ck
    CHECK (reason IN ('CORONATION', 'SERVICE_REPLAY', 'GENESIS')) NOT VALID;

ALTER TABLE reigns
    ADD CONSTRAINT reigns_state_ck
    CHECK (state IN ('ACTIVE', 'SUPERSEDED', 'REPAIRING')) NOT VALID;

CREATE UNIQUE INDEX IF NOT EXISTS reigns_one_active_uidx
    ON reigns ((true))
    WHERE state = 'ACTIVE';

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

ALTER TABLE events
    ADD CONSTRAINT events_severity_ck
    CHECK (severity IN ('DEBUG', 'INFO', 'WARN', 'ERROR')) NOT VALID;

CREATE INDEX IF NOT EXISTS events_stage_attempt_created_idx
    ON events (stage_attempt_id, created_at DESC);
CREATE INDEX IF NOT EXISTS events_type_created_idx
    ON events (event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
