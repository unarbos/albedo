# KubeTEE pin runs vs Denrite `ca530856`

Local copies of apple-to-apple eval artifacts (same pinned `sample_ids` /
`manifest_hash=e3cff617…` as `../reference-ca530856-ffa8-4e66-9175-7400e829e8c0/`).

| Run | Scores (chal / king) | n | Wall (`worker.execute`) | Total $ (judge+GPU) | Notes |
|-----|----------------------|---|-------------------------|---------------------|-------|
| `9a0f5f09…` | 0.68715 / 0.665788 (+2.14pp) | 100/100 | **1891 s** (~31.5 min) | **$11.89** | **latest** |
| `3e60c4c8…` | 0.680927 / 0.678479 | 100/100 | (see S3 eval-summary) | (see S3) | prior pin |
| `e6c56798…` | 0.677038 / 0.679567 | 99/100 | (see S3 eval-summary) | (see S3) | 1 king context overflow |

`9a0f5f09` economics (from [`9a0f5f09-eval-summary.json`](./9a0f5f09-eval-summary.json)): judge **$6.64** / 3881 req (glm-5.2 $5.51 + dsv4-flash $1.13); GPU **$5.25** (4× H200 @ $2.50/gpu/h); Job wall ~**32.6 min**.

Aggregate + paired stats: [`compare-vs-origin.json`](./compare-vs-origin.json).

Public S3: `https://s3.hippius.com/sn97-albedo/kubetee-poc/<eval_run_id>/`.
