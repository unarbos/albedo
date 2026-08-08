# Join a second non-CC H200 for concurrent challenger evals

Goal: two challenger Jobs at once against the shared `albedo-king` + `albedo-judge-api`.

## Layout

| Node | Role |
|------|------|
| `am-h200-25` | Pin king (`kubetee.ai/albedo-king=true`), 4 GPU |
| Second H200 | Challenger Job(s), 4 GPU each |

## Candidate second nodes (inventory)

Prefer an **idle non-tenant** H200. From `ansible/inventory.yaml` (probe-dated):

| Host | Notes |
|------|--------|
| `am-h200-160` / `161` / `166` | Lium (docker compose) when idle — verify with tenant probe before joining |
| Do **not** steal | Active Chutes (`chutes-td`) / Targon (`tdxvm`) / in-cluster CC nodes (`am-h200-23/28`) |

`am-h200-25` stays the king pin (`kubetee.ai/albedo-king=true`).

## Steps

1. Pick an idle non-CC 8×H200 (not a tenant Chutes/Targon/Lium node). Confirm in `ansible/inventory.yaml`.
2. Run host setup if needed (`ansible/run-fast … site-2604.yaml --limit <host>` with non-CC / container workload — not `vm-passthrough` for this PoC).
3. Join the node to `na-us-oakland-56` via Rancher registration (same cluster as `am-h200-25`).
4. Label for scheduling (do **not** set `kubetee.ai/albedo-king` on the second node):

```bash
kubectl --context na-us-oakland-56-direct label node <second-h200> \
  nvidia.com/gpu.workload.config=container \
  --overwrite
```

5. Confirm both nodes advertise `nvidia.com/gpu` allocatable ≥ 4 and Longhorn V2 disks / RWX PVCs mount from both.
6. Ensure king is labeled and deployed:

```bash
kubectl --context na-us-oakland-56-direct label node am-h200-25 \
  kubetee.ai/albedo-king=true --overwrite
kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/king.yaml
```

7. Launch two Jobs with distinct `EVAL_RUN_ID` / `SUBMISSION_ID` (copy `eval.yaml`, rename Job, change IDs).

## Verify concurrent smoke

```bash
kubectl --context na-us-oakland-56-direct -n albedo-poc get pods -o wide
# Expect: albedo-king on am-h200-25; two eval pods on different nodes (or one on each)
kubectl --context na-us-oakland-56-direct -n albedo-poc logs deploy/albedo-king -c king-api --tail=50
```
