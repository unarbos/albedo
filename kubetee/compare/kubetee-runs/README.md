# KubeTEE pin runs vs Denrite `ca530856`

Local copies of apple-to-apple eval artifacts (same pinned `sample_ids` /
`manifest_hash=e3cff617…` as `../reference-ca530856-ffa8-4e66-9175-7400e829e8c0/`).

| Run | Scores (chal / king) | n | Notes |
|-----|----------------------|---|-------|
| `9a0f5f09…` | 0.68715 / 0.665788 (+2.14pp) | 100/100 | **latest** |
| `3e60c4c8…` | 0.680927 / 0.678479 | 100/100 | prior pin |
| `e6c56798…` | 0.677038 / 0.679567 | 99/100 | 1 king context overflow |

Aggregate + paired stats: [`compare-vs-origin.json`](./compare-vs-origin.json).

Public S3: `https://s3.hippius.com/sn97-albedo/kubetee-poc/<eval_run_id>/`.
