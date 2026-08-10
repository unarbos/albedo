# Reference eval: bae48552-f673-4786-b788-e9d6893b01fa

Public Albedo SN97 run ([dashboard](https://pub-e2a73e9642e74a2ea78d2910c7a86025.r2.dev/detail.html?eval_run_id=bae48552-f673-4786-b788-e9d6893b01fa))
— challenger won and was coronated as king v110.

## Models

| Side | URI |
|------|-----|
| Previous king | `hf://power612/albedo-qwen3.6-35b-9abd59b0@cceca114b8575c0d350c99b1b95e553a74f2dcd7` |
| Challenger (won) | `seed429/albedo-qwen3.6-35b-hot@7e53e451bf0c6f0d8ac3f776af71dbcd6d436eb8` |

## Scores (Albedo / Denrite)

| Field | Value |
|-------|--------|
| Challenger | **0.693868** |
| King | **0.630353** |
| Margin | **+6.35 pp** (required 0.03) → won / coronated |
| Samples | 100/100 |
| Judge | `z-ai/glm-5.2` only |
| Hardware | B200 (origin); KubeTEE pin uses H200 |
| `manifest_hash` | `e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40` |

**Sample set:** 100 pinned IDs in [`request.json`](./request.json) — **0 overlap** with `ca530856…`.

KubeTEE pin:

```text
ALBEDO_EVAL_SAMPLE_IDS_FILE=/app/shared/albedo/kubetee/compare/reference-bae48552-f673-4786-b788-e9d6893b01fa/request.json
```
