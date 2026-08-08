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
| `deploy/pod-template.yaml` | Eval Job (no judge sidecar). |
| `deploy/judge-api-*.yaml` | Shared judge Deployment, Service, NetworkPolicy. |
| `deploy/configmap-judge-env.yaml` | Judge `ALBEDO_JUDGE_*` ConfigMap. |
| `deploy/configmap-poc-env.yaml` | Eval Job ConfigMap (`SELFDRIVE_JUDGE_BASE_URL`, etc.). |
| `deploy/secret-template.yaml` | Secret template (LiteLLM key + auth token + S3/HF). |
| `deploy/armada-job-template.yaml` | Future Armada submit shape (not used in PoC). |

## Quick start

1. Build + push the shared judge image (from the albedo repo root):
   ```bash
   docker buildx build --platform linux/amd64 \
     -f kubetee/Dockerfile.judge-api \
     -t ghcr.io/kubetee-ai/albedo-judge-api:latest --push .
   ```
2. Apply judge + eval ConfigMaps and judge workloads:
   ```bash
   kubectl --context na-us-oakland-56-direct apply -f \
     kubetee/deploy/configmap-judge-env.yaml \
     kubetee/deploy/judge-api-deployment.yaml \
     kubetee/deploy/judge-api-service.yaml \
     kubetee/deploy/judge-api-networkpolicy.yaml \
     kubetee/deploy/configmap-poc-env.yaml
   ```
3. Apply the Secret out-of-band (`ALBEDO_JUDGE_API_AUTH_TOKEN` must match
   `SELFDRIVE_SCORING_AUTH_TOKEN`):
   ```bash
   cp kubetee/deploy/secret-template.yaml ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   $EDITOR ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   kubectl --context na-us-oakland-56-direct apply -f \
     ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   ```
4. Submit an eval Job:
   ```bash
   kubectl --context na-us-oakland-56-direct -n albedo-poc delete job albedo-poc-eval --ignore-not-found
   kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/pod-template.yaml
   ```
