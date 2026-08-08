# Smoke checklist — split king vs challenger

Run after `kubetee-poc` has the remote-king code and `king.yaml` is Ready.

## 0. Preflight — king must be Ready before challenger

Do **not** apply `eval.yaml` until the king Deployment is Available and
`GET /ready` returns 200. The current 8-GPU co-located Job holds all GPUs on
`am-h200-25`; `albedo-king` stays Pending until that Job finishes and frees
at least 4 GPUs.

```bash
kubectl --context na-us-oakland-56-direct label node am-h200-25 \
  kubetee.ai/albedo-king=true --overwrite
kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/king.yaml
kubectl --context na-us-oakland-56-direct -n albedo-poc rollout status deploy/albedo-king --timeout=45m
# Confirm Ready (not Pending / not king_changing):
kubectl --context na-us-oakland-56-direct -n albedo-poc exec deploy/albedo-king -c king-api -- \
  wget -qO- http://127.0.0.1:8000/ready
```

Unit tests (local, no GPUs):

```bash
cd albedo && PYTHONPATH=src pytest tests/test_http_king_generator.py -q
```

## 1. Sample-id parity + 4+4 remote king

Use a short run (`ALBEDO_EVAL_SAMPLE_COUNT=2`, `ALBEDO_EVAL_MAX_TURNS` via
`ALBEDO_REMOTE_TRAJECTORY_ASSISTANT_TURNS=1` if exposed — else accept 8 turns
and a small sample count). Distinct `EVAL_RUN_ID`.

```bash
kubectl --context na-us-oakland-56-direct -n albedo-poc delete job albedo-poc-eval --ignore-not-found
# edit eval.yaml EVAL_RUN_ID / sample count, then:
kubectl --context na-us-oakland-56-direct apply -f kubetee/deploy/eval.yaml
kubectl --context na-us-oakland-56-direct -n albedo-poc logs -f job/albedo-poc-eval
```

Expect:
- spool `{artifacts}/spool/{eval_run_id}/sample-ids.json` written early
- generation events with `"king_remote": true`
- king pod logs completions; challenger uses local GPUs 0-3 only
- Job Completes with a scored verdict (not `king_changed`)

## 2. Mid-eval `king_changed` abort

While a challenger Job is generating:

```bash
kubectl --context na-us-oakland-56-direct -n albedo-poc exec deploy/albedo-king -c king-api -- \
  sh -c 'echo "{\"status\":\"changing\",\"king_generation_id\":\"2\",\"king_model_uri\":\"test\"}" > /tmp/king-state.json'
```

Expect Job fails with `fault_code=king_changed`, **no** registering verdict /
eval-summary upload. Restore king:

```bash
kubectl --context na-us-oakland-56-direct -n albedo-poc exec deploy/albedo-king -c king-api -- \
  sh -c 'echo "{\"status\":\"ready\",\"king_generation_id\":\"2\",\"king_model_uri\":\"$KING_MODEL_URI\"}" > /tmp/king-state.json'
```

## 3. Two concurrent challengers (needs second H200)

See [JOIN-SECOND-H200.md](JOIN-SECOND-H200.md). Copy `eval.yaml` to a second
Job name + distinct `EVAL_RUN_ID` / `SUBMISSION_ID`. Confirm pods land on
different nodes (or non-overlapping GPU sets) and both complete against the
same king Service.
