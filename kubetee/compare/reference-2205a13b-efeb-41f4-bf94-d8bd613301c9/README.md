# Reference eval: 2205a13b-efeb-41f4-bf94-d8bd613301c9

Public Albedo SN97 run ([dashboard](https://pub-e2a73e9642e74a2ea78d2910c7a86025.r2.dev/detail.html?eval_run_id=2205a13b-efeb-41f4-bf94-d8bd613301c9))
— challenger **lost** (margin below 3 pp win threshold). King stayed at v108 (`bkn1890/…hk971`).

## Models

| Side | URI |
|------|-----|
| Previous king | `hf://bkn1890/albedo-qwen3.6-35b-hk971@1ba87c52697f154a8f75a33ddf5714c1d314bcc7` |
| Challenger | `hf://darius3th/albedo-qwen3.6-35b-kkk@90bd03cba51b5b895e9b0e7dc4f1cb5caf7f4238` |

## Scores (Albedo / Denrite)

| Field | Value |
|-------|--------|
| Challenger | **0.673169** |
| King | **0.665314** |
| Margin | **+0.79 pp** (required 0.03) → lost |
| Samples | 100/100 |
| Judge | `z-ai/glm-5.2` only |
| `manifest_hash` | `e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40` |

Both HF repos are **public** (`gated=False`; anonymous weight HEAD 200).

KubeTEE pin:

```text
ALBEDO_EVAL_SAMPLE_IDS_FILE=/app/shared/albedo/kubetee/compare/reference-2205a13b-efeb-41f4-bf94-d8bd613301c9/request.json
```
