# Reference eval: ca530856-ffa8-4e66-9175-7400e829e8c0

Public Albedo SN97 run used as the target shape for KubeTEE PoC parity
**and as the pinned dataset sample set for apple-to-apple scoring.**

## Dataset pin (apple-to-apple)

| Field | Value |
|-------|--------|
| Source | [`request.json`](./request.json) → `dataset.sample_ids` (also mirrored in [`sample-ids.json`](./sample-ids.json)) |
| `manifest_hash` | `e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40` |
| `dataset.version` | `mini-coder+open-swe+smith-rs+hero-v1` |
| `sample_count` | **100** (explicit IDs — not seed-resampled) |
| Denrite `manifest_uri` | `s3://albedo/datasets/manifest.json` |
| KubeTEE corpus URI | `s3://sn97-albedo/datasets/manifest.json` (same hash; see `deploy/dataset-prep.yaml`) |

KubeTEE eval Jobs set:

```text
ALBEDO_EVAL_SAMPLE_IDS_FILE=/app/shared/albedo/kubetee/compare/reference-ca530856-ffa8-4e66-9175-7400e829e8c0/request.json
```

`run_eval.py` loads `dataset.sample_ids` from that file so challenger/king
scoring runs the **same 100 trajectories** as this Denrite eval. Seed-only
resampling (`ALBEDO_EVAL_SAMPLE_SEED=kubetee-poc`) is disabled while the
file is set — the earlier PoC run `7e09f071-…` used the seed and had **0**
sample overlap with this reference.

## Shape to match
- sample_count: **100**
- generation_batch_size: **32** → 4 gen batches
- scoring_batch_size: **32** → 4 score batches
- judge_count: **1** (z-ai/glm-5.2 only)

## Artifacts
See [`MANIFEST.json`](./MANIFEST.json) for sha256 checksums.

| file | role |
|------|------|
| request.json | eval request (batch sizes, **pinned sample_ids**, models) |
| sample-ids.json | same 100 IDs extracted for convenience |
| progress.jsonl | phase events (no wall-clock timestamps) |
| generated-samples.jsonl | king/challenger outputs |
| scoring-results.jsonl | per-sample judge scores |
| verdict.json | final scores / win |
| remote-logs.txt | short summary |

## Notes
- Dashboard finish: 2026-08-09T07:57:17Z
- Challenger was a cold HF download (~70GB); king was cache-hit
- Hardware: 8× RTX PRO 6000 Blackwell (not H200) — wall-clock not apples-to-apples with KubeTEE; **sample set is** apples-to-apples when `ALBEDO_EVAL_SAMPLE_IDS_FILE` points at this `request.json`
