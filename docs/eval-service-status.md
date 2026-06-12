# Eval Service Status And Runbook

This document describes the eval-service state in this repo right now and the shortest reliable way to run the eval stack. It is intentionally operational: what exists, where it runs, what each service does, and what is still missing.

## Current State

The repo contains a backend-side eval coordinator, a remote GPU eval API, a judge API, a scoring bridge, an up-to-date Postgres schema, and PM2 ecosystem files for each long-running or scheduled process.

The runtime Python entrypoints used by the eval stack are registered in `pyproject.toml`:

- `albedo-eval-api`: backend status API on port `8080`.
- `albedo-eval-dispatcher`: claims queued eval submissions and sends them to a remote GPU host.
- `albedo-eval-requeuer`: moves `EVAL_RETRYABLE` submissions back to `EVAL_QUEUED`.
- `albedo-remote-eval-api`: GPU-host control plane on port `8090`.
- `albedo-judge-api`: backend judge/scoring API on port `8091`.
- `albedo-score-bridge`: backend-to-remote WebSocket scoring bridge.
- `chain-reader` and `hippius-validation`: upstream ingestion/validation services, present but separate from the eval coordinator loop.

## Database

The eval stack uses Postgres. `docker-compose.yml` provides one local Postgres service named `albedo-postgres`. The current configured database schema is up to date with `schema.sql`.

`schema.sql` is the canonical fresh-database schema. It includes these eval-relevant tables:

- `chain_commits`, `miners`, `model_submissions`
- `stage_attempts`, `eval_runs`, `remote_gpu_hosts`, `artifacts`, `events`
- `king_versions`, `reigns`, `reign_members`
- weight/reign support tables

Important dispatcher behavior:

- Only `model_submissions.state = EVAL_QUEUED` is claimed by the dispatcher.
- Only one full eval is claimed at a time, guarded by a Postgres advisory lock and an `EVAL_RUNNING` check.
- The dispatcher only chooses hosts from `remote_gpu_hosts` where `role = EVAL`, `state = READY`, and `free_gpu_count >= 8`.
- Retryable eval failures become `EVAL_RETRYABLE`; the requeuer moves them back to `EVAL_QUEUED` for another dispatcher attempt.

## Backend Services

Run these on the backend/controller host.

### Required For Eval

- `albedo-eval-backend-api`
  - PM2 config: `pm2/ecosystem.backend-api.config.js`
  - Command: `uv run albedo-eval-api`
  - Serves `/health`, `/ready`, and `/submissions/{id}`.

- `albedo-judge-api`
  - PM2 config: `pm2/ecosystem.judge-api.config.js`
  - Command: `uv run albedo-judge-api`
  - Serves `/health`, `/ready`, and `/score-batch`.
  - Needs OpenRouter settings if real judging is used.

- `albedo-score-bridge`
  - PM2 config: `pm2/ecosystem.score-bridge.config.js`
  - Command: `uv run albedo-score-bridge`
  - Connects from backend to the remote GPU API WebSocket `/score-bridge` and forwards scoring requests to `albedo-judge-api`.

- `albedo-eval-dispatcher`
  - PM2 config: `pm2/ecosystem.eval-dispatcher.config.js`
  - Command: `uv run albedo-eval-dispatcher`
  - Long-running loop. Claims `EVAL_QUEUED` submissions, starts remote evals, records events, and completes successful or failed evals.

- `albedo-eval-reconciler`
  - PM2 config: `pm2/ecosystem.eval-reconciler.config.js`
  - Command: `uv run albedo-eval-dispatcher --reconcile-running`
  - PM2 cron: every minute.
  - Replays remote status/events for active evals that already have a `remote_run_id`.

- `albedo-eval-sweeper`
  - PM2 config: `pm2/ecosystem.eval-sweeper.config.js`
  - Command: `uv run albedo-eval-dispatcher --sweep-abandoned`
  - PM2 cron: every minute.
  - Marks expired `EVAL_RUNNING` attempts abandoned, sets the eval run to `FAILED_RETRYABLE`, and sets the submission to `EVAL_RETRYABLE`.

- `albedo-eval-requeuer`
  - PM2 config: `pm2/ecosystem.eval-requeuer.config.js`
  - Command: `uv run albedo-eval-requeuer`
  - PM2 cron: every minute.
  - Moves `EVAL_RETRYABLE` submissions with no active eval attempt back to `EVAL_QUEUED`.

### Optional Upstream Services

- `chain_reader`
  - PM2 config: `pm2/ecosystem.chain-validation.config.js`
  - Command: `uv run chain-reader`
  - Reads chain commits into the database.

- `hippius_validation`
  - PM2 config: `pm2/ecosystem.chain-validation.config.js`
  - Command: `uv run hippius-validation`
  - Validates model artifacts before eval eligibility.

These are needed for a complete production pipeline, but not for a local smoke test if you seed the database yourself.

## GPU Host Services

Run these on the GPU host unless you are doing local smoke testing.

- `albedo-remote-eval-api`
  - PM2 config: `pm2/ecosystem.remote-eval-api.config.js`
  - Command: `uv run albedo-remote-eval-api`
  - Serves `/health`, `/ready`, `/capacity`, `/eval-runs`, `/eval-runs/{id}`, `/eval-runs/{id}/events`, `/eval-runs/{id}/cancel`, and WebSocket `/score-bridge`.

The remote API can run in smoke mode with `ALBEDO_REMOTE_MOCK_AUTO_VERDICT=true`. For real GPU evals, it uses the remote worker path, vLLM generation, dataset loading, scoring, and artifact upload settings.

## Tunnel

The backend needs to reach the GPU host remote API. The provided PM2 tunnel is:

- `albedo-backend-to-gpu-api-tunnel`
  - PM2 config: `pm2/ecosystem.gpu-host-tunnel.config.js`
  - Opens `ALBEDO_TUNNEL_BACKEND_LOCAL_GPU_PORT` on the backend and forwards it to `127.0.0.1:ALBEDO_REMOTE_EVAL_API_PORT` on the GPU host.

Set these in `.env` on the backend:

```bash
ALBEDO_GPU_HOST_USER=...
ALBEDO_GPU_HOST_SSH_HOST=...
ALBEDO_TUNNEL_BACKEND_LOCAL_GPU_PORT=18090
ALBEDO_REMOTE_EVAL_API_PORT=8090
```

Then the backend should register the remote eval host in `remote_gpu_hosts` with a `base_url` that points to the tunnel, usually `http://127.0.0.1:18090`.

## Required Environment

Start from `.env.example` and fill a real `.env`.

Backend/controller host needs at least:

```bash
ALBEDO_EVAL_DATABASE_URL=postgresql://user:password@127.0.0.1:65432/db
ALBEDO_EVAL_WORKER_ID=eval-dispatcher-1
ALBEDO_EVAL_REMOTE_AUTH_TOKEN=shared-remote-token
ALBEDO_EVAL_DATASET_MANIFEST_URI=s3://albedo-artifacts/datasets/swe-zero/manifest.json
ALBEDO_EVAL_DATASET_MANIFEST_HASH=982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f
ALBEDO_EVAL_JUDGE_CONFIG_HASH=sha256:replace-with-real-hash
ALBEDO_EVAL_ARTIFACT_PREFIX=s3://albedo-artifacts
ALBEDO_JUDGE_API_AUTH_TOKEN=shared-judge-token
ALBEDO_JUDGE_OPENROUTER_API_KEY=...
ALBEDO_SCORE_BRIDGE_REMOTE_WS_URL=ws://127.0.0.1:18090/score-bridge
ALBEDO_SCORE_BRIDGE_REMOTE_AUTH_TOKEN=shared-remote-token
ALBEDO_SCORE_BRIDGE_JUDGE_BASE_URL=http://127.0.0.1:8091
ALBEDO_SCORE_BRIDGE_JUDGE_AUTH_TOKEN=shared-judge-token
```

GPU host needs at least:

```bash
ALBEDO_REMOTE_AUTH_TOKEN=shared-remote-token
ALBEDO_REMOTE_HOST_ID=eval-gpu-1
ALBEDO_REMOTE_HOST_ROLE=EVAL
ALBEDO_REMOTE_GPU_COUNT=8
ALBEDO_REMOTE_FREE_GPU_COUNT=8
ALBEDO_REMOTE_ACCELERATOR_TYPE=B200
ALBEDO_REMOTE_READY=true
ALBEDO_REMOTE_GENERATION_BACKEND=vllm
ALBEDO_REMOTE_DATASET_ROOT=/path/to/swe-zero
ALBEDO_REMOTE_PREVIOUS_KING_GPU_IDS=0,1,2,3
ALBEDO_REMOTE_CHALLENGER_GPU_IDS=4,5,6,7
ALBEDO_REMOTE_SCORING_BACKEND=websocket
ALBEDO_REMOTE_UPLOAD_ARTIFACTS=true
ALBEDO_REMOTE_S3_ENDPOINT_URL=...
ALBEDO_REMOTE_S3_ACCESS_KEY_ID=...
ALBEDO_REMOTE_S3_SECRET_ACCESS_KEY=...
```

For smoke testing only, the GPU host can use:

```bash
ALBEDO_REMOTE_MOCK_AUTO_VERDICT=true
ALBEDO_REMOTE_MOCK_CHALLENGER_WON=false
```

## One-Time Setup

On a fresh backend host/database:

```bash
cp .env.example .env
set -a
source .env
set +a
uv sync
docker compose up -d albedo-postgres
docker compose exec -T albedo-postgres psql -U "$ALBEDO_POSTGRES_USER" -d "$ALBEDO_POSTGRES_DB" < schema.sql
```

If the remote GPU host is reached through the SSH tunnel, upsert a host row after the tunnel URL is chosen:

```sql
INSERT INTO remote_gpu_hosts (
    id, role, base_url, state, gpu_count, free_gpu_count,
    accelerator_type, capabilities, last_heartbeat_at
)
VALUES (
    'eval-gpu-1', 'EVAL', 'http://127.0.0.1:18090', 'READY', 8, 8,
    'B200', '{}'::jsonb, now()
)
ON CONFLICT (id) DO UPDATE
SET base_url = EXCLUDED.base_url,
    role = EXCLUDED.role,
    state = EXCLUDED.state,
    gpu_count = EXCLUDED.gpu_count,
    free_gpu_count = EXCLUDED.free_gpu_count,
    accelerator_type = EXCLUDED.accelerator_type,
    last_heartbeat_at = now();
```

The dispatcher also needs at least one active lead king in `reigns`/`reign_members`/`king_versions`, and eval candidates must already be in `model_submissions` as `EVAL_QUEUED` with a linked `chain_commits.block_hash`.

## Run All Eval Services With PM2

Backend/controller host:

```bash
pm2 start pm2/ecosystem.backend-api.config.js
pm2 start pm2/ecosystem.judge-api.config.js
pm2 start pm2/ecosystem.gpu-host-tunnel.config.js
pm2 start pm2/ecosystem.score-bridge.config.js
pm2 start pm2/ecosystem.eval-dispatcher.config.js
pm2 start pm2/ecosystem.eval-reconciler.config.js
pm2 start pm2/ecosystem.eval-sweeper.config.js
pm2 start pm2/ecosystem.eval-requeuer.config.js
```

GPU host:

```bash
pm2 start pm2/ecosystem.remote-eval-api.config.js
```

Optional complete-ingestion services on the backend:

```bash
pm2 start pm2/ecosystem.chain-validation.config.js
```

Useful PM2 checks:

```bash
pm2 list
pm2 logs albedo-eval-dispatcher
pm2 logs albedo-eval-reconciler
pm2 logs albedo-eval-sweeper
pm2 logs albedo-eval-requeuer
pm2 logs albedo-remote-eval-api
pm2 logs albedo-score-bridge
```

## Manual Commands

Manual one-shot or foreground commands are useful for smoke tests and debugging:

```bash
uv run albedo-eval-api
uv run albedo-judge-api
uv run albedo-remote-eval-api
uv run albedo-score-bridge
uv run albedo-eval-dispatcher --once
uv run albedo-eval-dispatcher --reconcile-running --limit 10
uv run albedo-eval-dispatcher --sweep-abandoned
uv run albedo-eval-requeuer --limit 100
```

Run the dispatcher forever without PM2:

```bash
uv run albedo-eval-dispatcher
```

## Health Checks

Backend API:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

Judge API:

```bash
curl http://127.0.0.1:8091/health
curl http://127.0.0.1:8091/ready
```

Remote API through the backend tunnel:

```bash
curl -H "Authorization: Bearer $ALBEDO_EVAL_REMOTE_AUTH_TOKEN" http://127.0.0.1:18090/ready
curl -H "Authorization: Bearer $ALBEDO_EVAL_REMOTE_AUTH_TOKEN" http://127.0.0.1:18090/capacity
```

## Eval State Flow

Normal success path:

```text
EVAL_QUEUED
  -> EVAL_RUNNING
  -> EVAL_WIN or COMPLETE_LOSS
```

Retryable failure path:

```text
EVAL_RUNNING
  -> EVAL_RETRYABLE
  -> EVAL_QUEUED
  -> EVAL_RUNNING
```

The sweeper performs `EVAL_RUNNING -> EVAL_RETRYABLE` for expired leases. The requeuer performs `EVAL_RETRYABLE -> EVAL_QUEUED`. The dispatcher only claims `EVAL_QUEUED`.

## Current Gaps

- There is no automatic `remote_gpu_hosts` registration/heartbeat writer in this repo slice; seed or update the row manually or from external orchestration.
- Retry backoff is not implemented. The requeuer immediately makes retryable eval submissions eligible again.
- S3 dataset manifest fetching on the backend is not implemented. Dispatcher-side deterministic `sample_ids` require a local `ALBEDO_EVAL_DATASET_MANIFEST_PATH`; otherwise the request can omit `sample_ids`.
- The remote worker path exists, but production correctness depends on real dataset files, model artifact resolution, vLLM runtime setup, scoring bridge connectivity, and artifact upload credentials.
- Chain ingestion, Hippius validation, pre-eval, set-reign, and weight-setting are not fully covered by this eval-service runbook.

## Tests

Run normal tests:

```bash
uv run pytest -q
```

Run Postgres integration tests after pointing `ALBEDO_TEST_DATABASE_URL` at a test database. The fixture loads `schema.sql`:

```bash
ALBEDO_TEST_DATABASE_URL=postgresql://user:password@127.0.0.1:65432/db uv run pytest -q tests/integration
```
