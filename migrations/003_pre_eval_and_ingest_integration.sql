-- Integration tables and constraints for chain ingest, Hippius validation, and pre-eval cache.

CREATE UNIQUE INDEX IF NOT EXISTS model_submissions_chain_commit_id_uidx
    ON model_submissions (chain_commit_id);

CREATE INDEX IF NOT EXISTS model_submissions_hotkey_state_idx
    ON model_submissions (hotkey, state, created_at DESC);

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
