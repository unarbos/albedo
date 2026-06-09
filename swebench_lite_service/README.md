# Albedo SWE-bench Lite Service

This is an isolated benchmark worker for replaying historical Albedo kings on
`SWE-bench/SWE-bench_Lite`.

The service stores local JSON state under `ALBEDO_SWEBENCH_STATE_DIR`:

- `state.json` tracks each benchmark by immutable `repo@sha256:digest`.
- `runs/<run_id>/predictions.jsonl` stores SWE-bench predictions.
- `runs/<run_id>/raw_generations.json` stores raw model replies.
- `reports/<run_id>/` stores official SWE-bench harness output.

## Pod Setup

On the GPU pod:

```bash
cd /root/albedo-benchmarks
bash swebench_lite_service/setup_pod.sh
pm2 start swebench_lite_service/ecosystem.config.js
pm2 logs albedo-swebench-lite-worker
```

The PM2 worker is configured for one king only. To smoke-test the pipeline without
running all 300 tasks, set:

```bash
export ALBEDO_SWEBENCH_LIMIT=1
pm2 start swebench_lite_service/ecosystem.config.js
```

Unset `ALBEDO_SWEBENCH_LIMIT` or set it to `0` for the full Lite benchmark.

## Tunnel

From your local machine:

```bash
pm2 start swebench_lite_service/ecosystem.tunnel.config.js
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/state
```

Defaults point at `root@216.243.220.131 -p 40008`.

## Notes

Predictions are generated with `mini-swe-agent` running against a local vLLM
OpenAI-compatible endpoint for the selected king. The official SWE-bench harness
then grades the resulting `predictions.jsonl`.

`mini-swe-agent` uses Docker testbed images from Docker Hub. Run `docker login`
on the pod or configure an image mirror/cache before full Lite runs, otherwise
Docker Hub unauthenticated pull limits can stop the benchmark before the model is
called. Set `HF_TOKEN` as well if Hugging Face rate limits become an issue.

Official SWE-bench Lite is 300 tasks, and the official harness entrypoint is
`python -m swebench.harness.run_evaluation`.


## S3 Publishing

After each completed run, the pod worker records planned public S3 URLs, but it
does not upload to S3. Run the uploader on the host that owns dashboard S3
credentials; it mirrors pod artifacts over SSH and uploads from this host. It
reads `ALBEDO_SWEBENCH_S3_*` first and falls back to `ALBEDO_DS_*` variables:

```text
s3://albedo/swebench-lite/index.json
s3://albedo/swebench-lite/state.json
s3://albedo/swebench-lite/runs/<run_id>/summary.json
s3://albedo/swebench-lite/runs/<run_id>/predictions.jsonl
s3://albedo/swebench-lite/runs/<run_id>/raw_generations.json
s3://albedo/swebench-lite/runs/<run_id>/official-report.json
s3://albedo/swebench-lite/kings/<king_slug>.json
```

Run once from this host:

```bash
python3 -m swebench_lite_service.host_s3_uploader
```

Or run continuously with PM2 from this host:

```bash
pm2 start swebench_lite_service/ecosystem.tunnel.config.js --only albedo-swebench-lite-s3-uploader
```
