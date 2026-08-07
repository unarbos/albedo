# KubeTEE PoC — SN97 Denrite (Albedo) GPU Evaluation

This directory lives in the `albedo` submodule on the `kubetee-poc` branch
(based off `dev`). It holds the KubeTEE-side PoC code for running Albedo's
king-of-the-hill GPU evaluations as one-off Armada jobs on the KubeTEE
staging cluster (`na-us-oakland-56`, 8x H200 + TDX).

## What this is

One Armada job submission = one Kubernetes pod = one king-of-the-hill
evaluation. Denrite's external dispatcher submits one job per miner model
evaluation; the pod runs the eval, uploads artifacts to S3, prints the
verdict JSON to stdout, and exits. Denrite's external pipeline picks up the
verdict from S3.

See [PLAN.md](PLAN.md) for the full architecture, per-job inputs, submission
flow, and follow-ups.

## Files

| File | Purpose |
|------|---------|
| `app/run_eval.py` | Per-job self-drive entrypoint (builds `EvalRequest` from env, runs upstream `RemoteEvalWorker`, prints verdict). |
| `Dockerfile` | PoC image — Hopper-compatible vLLM base + albedo's non-vLLM deps. |
| `deploy/pod-template.yaml` | K8s pod spec (eval + judge-api sidecar) — reference shape. |
| `deploy/armada-job-template.yaml` | Armada JobSubmitRequest template Denrite's dispatcher fills in per submission. |
| `deploy/configmap-poc-env.yaml` | Non-secret env (judge URL, judge model IDs, sample count, GPU split). Applied out-of-band. |
| `deploy/secret-template.yaml` | Template for the out-of-band Secret (LiteLLM virtual key, HF token, S3 creds). |

## Quick start

1. Build + push the image (from the albedo repo root, on `kubetee-poc`):
   ```bash
   docker buildx build --platform linux/amd64 \
     -f kubetee/Dockerfile \
     -t ghcr.io/kubetee-ai/albedo-poc:latest --push .
   ```
2. Apply the ConfigMap on `na-us-oakland-56`:
   ```bash
   kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/configmap-poc-env.yaml
   ```
3. Apply the Secret out-of-band (never committed):
   ```bash
   cp kubetee/deploy/secret-template.yaml ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   $EDITOR ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   kubectl --context na-us-oakland-56-direct apply -f ~/kubetee-secret-backends/albedo-poc-secrets.yaml
   ```
4. Submit one eval via Armada:
   ```bash
   armadactl submit kubetee/deploy/armada-job-template.yaml \
     --armadaUrl armada.armada.svc.cluster.local:50051
   ```
   (Denrite's dispatcher does the same via an SSH tunnel — see [PLAN.md](PLAN.md).)
