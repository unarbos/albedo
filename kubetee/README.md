# KubeTEE PoC — SN97 Denrite (Albedo) GPU Evaluation

This directory lives in the `albedo` submodule on the `kubetee-poc` branch
(based off `dev`). It holds the KubeTEE-side PoC code for running Albedo's
king-of-the-hill GPU evaluations as one-off Jobs on the KubeTEE staging
cluster (`na-us-oakland-56`).

## What this is

One Job = one Kubernetes pod = one king-of-the-hill evaluation. The pod
runs vLLM (king + challenger), scores via the **shared** `albedo-judge-api`
Service in `albedo-poc`, uploads artifacts to S3, prints the verdict JSON
to stdout, and exits.

See [PLAN.md](PLAN.md) for architecture, apply order, auth, NetworkPolicy,
`replicas: 1`, and `GET /eval-cost/{eval_run_id}` cost isolation.

## Files

| File | Purpose |
|------|---------|
| `app/run_eval.py` | Per-job self-drive entrypoint. |
| `Dockerfile` | Reference eval image (live Job uses `vllm/vllm-openai` + git clone). |
| `Dockerfile.judge-api` | Slim shared judge image → `ghcr.io/kubetee-ai/albedo-judge-api`. |
| `deploy/eval.yaml` | Eval ConfigMap + Job + PVCs (no judge sidecar). |
| `deploy/judge-api.yaml` | Shared judge stack (ConfigMap + Deployment + Service + NetworkPolicy). |
| `deploy/secret-template.yaml` | Secret template (LiteLLM key + auth token + S3/HF). |
| `deploy/armada-job-template.yaml` | Future Armada submit shape (not used in PoC). |

## Quick start

1. Build + push the shared judge image (from the albedo repo root):
   ```bash
   docker buildx build --platform linux/amd64 \
     -f kubetee/Dockerfile.judge-api \
     -t ghcr.io/kubetee-ai/albedo-judge-api:latest --push .
   ```
2. Apply judge + eval manifests:
   ```bash
   kubectl --context na-us-oakland-56-direct apply -f \
     kubetee/deploy/judge-api.yaml \
     kubetee/deploy/eval.yaml
   ```
3. Apply the Secret out-of-band (`ALBEDO_JUDGE_API_AUTH_TOKEN` must match
   `SELFDRIVE_SCORING_AUTH_TOKEN`):
   ```bash
   cp kubetee/deploy/secret-template.yaml ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   $EDITOR ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   kubectl --context na-us-oakland-56-direct apply -f \
     ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   ```
4. Re-submit an eval Job (delete first — Jobs are immutable):
   ```bash
   kubectl --context na-us-oakland-56-direct -n albedo-poc delete job albedo-poc-eval --ignore-not-found
   kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/eval.yaml
   ```
