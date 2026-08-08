# KubeTEE PoC — SN97 Denrite (Albedo) GPU Evaluation

This directory lives in the `albedo` submodule on the `kubetee-poc` branch
(based off `dev`). It holds the KubeTEE-side PoC code for running Albedo's
king-of-the-hill GPU evaluations as one-off Jobs on the KubeTEE staging
cluster (`na-us-oakland-56`).

## What this is

**Always-on king** (`deploy/king.yaml`, 4 GPU / TP=4) + **per-eval challenger
Job** (`deploy/eval.yaml`, 4 GPU / TP=4). The challenger loads the dataset,
calls the king over HTTP for previous-king completions, runs local vLLM for
the challenger, scores via the shared `albedo-judge-api`, uploads artifacts
to S3, and exits.

See [PLAN.md](PLAN.md) for architecture. Smoke / second-node / 8-GPU parity:
[deploy/SMOKE.md](deploy/SMOKE.md), [deploy/JOIN-SECOND-H200.md](deploy/JOIN-SECOND-H200.md),
[deploy/PARITY-KING-8GPU.md](deploy/PARITY-KING-8GPU.md).

## Files

| File | Purpose |
|------|---------|
| `app/run_eval.py` | Per-job self-drive entrypoint (challenger). |
| `app/king_serve.py` | King control plane (`/ready` + `/v1` proxy; 503 `king_changing`). |
| `Dockerfile` | Reference eval image (live Job uses `vllm/vllm-openai` + git clone). |
| `Dockerfile.judge-api` | Slim shared judge image → `ghcr.io/kubetee-ai/albedo-judge-api`. |
| `deploy/king.yaml` | Always-on king (ConfigMap + Deployment + Service + NetworkPolicy). |
| `deploy/eval.yaml` | Challenger ConfigMap + Job + PVCs (`ALBEDO_REMOTE_KING_BASE_URL`). |
| `deploy/judge-api.yaml` | Shared judge stack (ConfigMap + Deployment + Service + NetworkPolicy). |
| `deploy/secret-template.yaml` | Secret template (LiteLLM key + auth token + S3/HF). |
| `deploy/armada-job-template.yaml` | Future Armada submit shape (not used in PoC). |

## Quick start

1. Label the king node and apply judge + king (after secrets exist):
   ```bash
   kubectl --context na-us-oakland-56-direct label node am-h200-25 \
     kubetee.ai/albedo-king=true --overwrite
   kubectl --context na-us-oakland-56-direct apply -f \
     kubetee/deploy/judge-api.yaml \
     kubetee/deploy/king.yaml
   ```
2. Apply the Secret out-of-band (`ALBEDO_JUDGE_API_AUTH_TOKEN` must match
   `SELFDRIVE_SCORING_AUTH_TOKEN`; HF token for gated king weights):
   ```bash
   cp kubetee/deploy/secret-template.yaml ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   $EDITOR ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   kubectl --context na-us-oakland-56-direct apply -f \
     ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   ```
3. **Wait until king is Ready** (weights loaded, `/ready` = 200) — do **not**
   apply the challenger Job while the king pod is Pending/loading, or it can
   race for the remaining GPUs on `am-h200-25`:
   ```bash
   kubectl --context na-us-oakland-56-direct -n albedo-poc rollout status deploy/albedo-king --timeout=45m
   kubectl --context na-us-oakland-56-direct -n albedo-poc exec deploy/albedo-king -c king-api -- \
     wget -qO- http://127.0.0.1:8000/ready
   ```
4. Then submit the challenger Job (delete first — Jobs are immutable). The Job
   also has a `wait-for-king` initContainer as a safety gate:
   ```bash
   kubectl --context na-us-oakland-56-direct -n albedo-poc delete job albedo-poc-eval --ignore-not-found
   kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/eval.yaml
   ```
