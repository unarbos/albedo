# Eval Service Status

This document tracks the current eval-service-only implementation against `Systemdesign.md`.
It is intended for repo users who need to know what can run today and what still needs to be built.

## Finished

- Python package scaffold for the backend-side eval service under `src/albedo_eval_service`.
- `uv`/Python project metadata in `pyproject.toml`, filled local `.env`, and example runtime settings in `.env.example`.
- Postgres migrations:
  - `001_eval_service.sql` creates the eval-service base schema.
  - `002_full_system_schema.sql` completes the full `Systemdesign.md` schema with weight epochs, weight transactions, stricter state/fault constraints, and operational indexes.
  - `albedo-db-migrate` applies migrations idempotently.
- Docker Compose Postgres uses only `.env` values and non-default local credentials/port.
- PM2 ecosystem files are split by service: backend API, dispatcher, reconciler, sweeper, remote eval API, and GPU-host tunnels.
- Two-way SSH tunnel PM2 config supports backend-to-GPU remote API access and GPU-to-backend access.
- Postgres migration for the eval-service slice:
  - Minimal shared dependencies: `chain_commits`, `miners`, and `model_submissions`.
  - Eval state: `stage_attempts`, `remote_gpu_hosts`, `eval_runs`, `artifacts`, and `events`.
  - Lead-king snapshot dependencies: `king_versions`, `reigns`, and `reign_members`.
  - Constraints/indexes for model-hash dedupe, queued eval lookup, one active eval run, active attempts, and EVAL host selection.
- SWE-ZERO dataset contract defaults:
  - Dataset: `AlienKevin/SWE-ZERO-12M-trajectories`.
  - Initial manifest hash: `982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f`.
  - `sample_count = 128`, `max_turns_per_sample = 10`, `sampling_algo = swe-zero-manifest-sample-v1`.
  - `eval_runs` persists `dataset_manifest_hash`, `dataset_max_turns_per_sample`, `dataset_sample_seed`, `dataset_sample_ids`, and `dataset_sampling_algo`.
- Deterministic SWE-ZERO manifest coordinate sampler for `data/train-*.parquet` shard manifests.
- Optional local manifest loading via `ALBEDO_EVAL_DATASET_MANIFEST_PATH`, including SHA-256 verification and dispatcher-side `sample_ids` generation.
- Remote failure classification helpers:
  - Explicit miner faults are terminal.
  - Provider, infra, unknown, and broken remote streams are retryable.
- Remote eval request models and tunnel client for `/ready`, `/eval-runs`, `/eval-runs/{id}`, and `/eval-runs/{id}/events`.
- Backend-side eval dispatcher skeleton:
  - Claims only `EVAL_QUEUED` submissions with a stored commit block hash.
  - Selects only `role = EVAL`, `state = READY` GPU hosts with at least 8 free GPUs.
  - Uses a Postgres advisory lock plus active-state checks for sequential full eval.
  - Creates `stage_attempts` and `eval_runs` before remote work starts.
  - Persists remote progress events.
  - Persists remote run IDs after remote start.
  - Refreshes the stage-attempt lease while remote events are replayed.
  - Provides `--sweep-abandoned` to mark expired `EVAL_RUNNING` attempts as `ABANDONED` and return submissions to `EVAL_RETRYABLE`.
  - Provides `--reconcile-running` to replay remote events/status for active evals with stored `remote_run_id`.
  - Records known verdict artifact links into `artifacts` rows on successful eval completion.
  - Marks successful verdicts as `EVAL_WIN` or `COMPLETE_LOSS`.
  - Marks remote HTTP/stream failures as retryable `REMOTE_EVAL_FAULT`.
- Minimal backend FastAPI surface: `/health`, `/ready`, and `/submissions/{id}`.
- Minimal remote eval smoke API: `/health`, `/ready`, `/capacity`, `/eval-runs`, event replay, status lookup, and cancel.
- Remote smoke mode can emit a deterministic mock verdict for dispatcher/tunnel testing before GPU generation is implemented.
- Focused tests for SWE-ZERO sampling, manifest verification, dispatcher request building, artifact mapping, remote smoke API, fault classification, and Postgres claim/sequential/recovery behavior.

## Unfinished

- Remote GPU host service is only a smoke control plane right now:
  - No real GPU reservation implementation.
  - No vLLM model loading or generation workers.
  - No scoring worker or judge provider integration.
- S3/Hippius artifact upload is not implemented yet. Remote-produced verdict artifact links are recorded after successful eval completion.
- S3 dataset manifest fetching is not implemented yet. The dispatcher can generate `sample_ids` only when a local `ALBEDO_EVAL_DATASET_MANIFEST_PATH` is configured.
- Retry backoff/requeue scheduling is not implemented yet.
- Other subnet services remain out of this eval-service-only slice: chain reader, Hippius validation worker, pre-eval dispatcher, set reign worker, and weight setter.

## Run Notes

- Set `ALBEDO_EVAL_DATASET_MANIFEST_PATH` to a local SWE-ZERO `manifest.json` to include deterministic `sample_ids` in eval requests.
- Start local Postgres: `docker compose up -d albedo-postgres`.
- Apply migrations: `uv run albedo-db-migrate`.
- Install/sync dependencies with `uv sync` once network/package access is available.
- Run backend API: `uv run albedo-eval-api`.
- Run remote eval smoke API: `uv run albedo-remote-eval-api`.
- Run dispatcher once: `uv run albedo-eval-dispatcher --once`.
- Sweep expired eval leases: `uv run albedo-eval-dispatcher --sweep-abandoned`.
- Reconcile active remote runs: `uv run albedo-eval-dispatcher --reconcile-running`.
- Run unit tests: `uv run pytest -q`.
- Run Postgres integration tests: `ALBEDO_TEST_DATABASE_URL= uv run pytest -q tests/integration`.
- Start PM2 service files individually from `pm2/`, for example `pm2 start pm2/ecosystem.eval-dispatcher.config.js`.
