# Reference eval: ca530856-ffa8-4e66-9175-7400e829e8c0

Public Albedo SN97 run used as the target shape for KubeTEE PoC parity.

## Shape to match
- sample_count: **100**
- generation_batch_size: **32** → 4 gen batches
- scoring_batch_size: **32** → 4 score batches
- judge_count: **1** (z-ai/glm-5.2 only)

## Artifacts
See  for sha256 checksums.

| file | role |
|------|------|
| request.json | eval request (batch sizes, sample count) |
| progress.jsonl | phase events (no wall-clock timestamps) |
| generated-samples.jsonl | king/challenger outputs |
| scoring-results.jsonl | per-sample judge scores |
| verdict.json | final scores / win |
| remote-logs.txt | short summary |

## Notes
- Dashboard : 2026-08-09T07:57:17Z
- Challenger was a cold HF download (~70GB); king was cache-hit
- Hardware: 8× RTX PRO 6000 Blackwell (not H200) — wall-clock not apples-to-apples with KubeTEE
