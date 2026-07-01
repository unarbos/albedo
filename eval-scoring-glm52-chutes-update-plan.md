# Evaluation and Scoring Update Plan: GLM 5.2 Categories via Chutes

## Goal

Update the full-eval scoring pipeline so the scoring service generates exactly 5 per-sample scoring categories with GLM 5.2, based on a normal candidate-style GLM 5.2 response to the same sample prompt. Use Chutes as the primary GLM 5.2 provider. If Chutes fails, fall back to OpenRouter GLM 5.2 with provider routing restricted to FP8 endpoints only. Judges score king/challenger outputs against those categories.

The GLM 5.2 answer generation should overlap with previous-king and challenger generation. The remote worker starts category prep in the scoring service as soon as samples are loaded, while king/challenger generation runs on GPUs.

No reusable category cache. No category persistence in Postgres. No GLM response/category/reference artifacts uploaded to Hippius S3.

Target flow:

```text
Load eval samples
Start scoring-service category prep with sample prompts
  -> scoring service calls GLM 5.2 through Chutes to generate a normal response to each sample
  -> if Chutes fails, scoring service falls back to OpenRouter `z-ai/glm-5.2` with FP8-only provider routing
  -> scoring service calls GLM 5.2 again to generate exactly 5 scoring categories from that GLM response
  -> category generation also uses Chutes first, then OpenRouter FP8 fallback if Chutes fails
In parallel:
  generate previous king output
  generate challenger output
Send valid king/challenger pairs to scoring service with the category prep id
Scoring service joins prepared categories with king/challenger outputs
Judges score king/challenger outputs on those 5 categories through the existing OpenRouter judge path
If GLM-category scoring cannot recover, fall back to the current fixed-metric scoring path
Aggregate scores
Check whether challenger won
```

## Current Repo Touchpoints

- Remote eval orchestration: `src/albedo_eval_service/remote_worker.py`
- Parallel king/challenger generation: `RemoteEvalWorker._execute`
- Scoring transport from remote worker to judge API: `src/albedo_eval_service/remote_scoring.py`
- Scoring service / judge API: `src/albedo_eval_service/judge_api.py`
- Current fixed metric rubric and aggregation: `src/albedo_eval_service/judge_core.py`
- Current judge provider implementation: `src/albedo_eval_service/judge_openrouter.py`
- Judge/scoring config: `src/albedo_eval_service/judge_config.py`
- Error notification hook: add a small Slack webhook helper used by eval/scoring paths

## GLM 5.2 Provider Strategy

Use Chutes first for both GLM response generation and scoring-category generation:

- Base URL: `https://llm.chutes.ai`
- Chat endpoint: `POST /v1/chat/completions`
- Auth: `Authorization: Bearer $CHUTES_API_KEY`
- Model identifier to pin in config/results: `zai-org/GLM-5.2-TEE`
- Chute ID to record for traceability: `08901219-159f-55a7-87cf-9d0d02744668`
- Do not route GLM 5.2 response generation or category generation through OpenRouter unless the Chutes request fails.

Fallback to OpenRouter only after Chutes fails:

- OpenRouter model slug: `z-ai/glm-5.2`
- Base URL: `https://openrouter.ai/api/v1`
- Auth: `Authorization: Bearer $OPENROUTER_API_KEY`
- Restrict provider routing to FP8 endpoints with `provider.quantizations = ["fp8"]`.
- Keep `provider.allow_fallbacks = true` so OpenRouter can try multiple eligible providers, but every eligible provider must still match the FP8 quantization filter.
- Use `provider.require_parameters = true` for structured/strict requests so OpenRouter only selects providers that support requested parameters.
- If OpenRouter has no available FP8 endpoint for GLM 5.2, fail the sample/run as a provider fault; do not route to non-FP8 providers.

Example OpenRouter fallback provider block:

```json
{
  "provider": {
    "quantizations": ["fp8"],
    "allow_fallbacks": true,
    "require_parameters": true
  }
}
```

Implementation notes:

- Add a dedicated Chutes client module, for example `src/albedo_eval_service/chutes_glm.py`.
- Add scoring-service config under `ALBEDO_JUDGE_` or equivalent: `CHUTES_API_KEY`, `CHUTES_BASE_URL`, `GLM52_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_GLM52_MODEL=z-ai/glm-5.2`, `OPENROUTER_GLM52_QUANTIZATIONS=fp8`, `CATEGORY_COUNT=5`, `CATEGORY_PROMPT_VERSION`, retry/timeout/concurrency settings, and optional `SLACK_ERROR_WEBHOOK_URL`.
- Keep the Chutes client separate from `OpenRouterJudgeClient`; add a small OpenRouter GLM fallback client or shared provider wrapper so judge OpenRouter behavior and GLM fallback behavior remain separately configurable.
- Use non-streaming Chutes requests for scoring-service simplicity.
- Parse and validate Chutes/OpenAI-style response bodies from both Chutes and OpenRouter fallback, but do not create separate S3 or Postgres records for raw GLM responses or raw category responses.

## No External Category Persistence

Remove all external storage work from the category plan:

- Do not add `eval_category_sets` or any other category table to `schema.sql`.
- Do not add backend category get/upsert APIs.
- Do not check Postgres before category generation.
- Do not save GLM candidate-style responses or category payloads to Postgres.
- Do not upload GLM candidate-style responses, raw category responses, or category sets to Hippius S3.
- Do not add category artifact types to `VERDICT_ARTIFACT_TYPES`.

The scoring service may keep in-flight category prep state only long enough to join it with the later `/score-batch` request. That state is scoped to the active eval/category-prep id, has a TTL, and is not reused across evals.

The only durable-ish category copy in the eval output is inside the scoring service result for that score request, such as each scoring record returned by `/score-batch` and whatever existing scoring-results artifact already records.

Recommended scoring-record category shape:

```json
{
  "category_source": {
    "provider": "chutes",
    "model": "zai-org/GLM-5.2-TEE",
    "chute_id": "08901219-159f-55a7-87cf-9d0d02744668",
    "prompt_version": "glm52-categories-v1"
  },
  "categories": [
    {
      "id": "cat_01",
      "name": "Functional correctness",
      "description": "What the judge should evaluate.",
      "scoring_guidance": "How to distinguish better, equal, and worse answers."
    }
  ],
  "category_hash": "sha256:..."
}
```

`category_hash` is computed in the scoring service for traceability inside the scoring result only. It is not a reusable cache key and does not imply external persistence.

## Worker Flow Changes

Remote worker starts scoring-service category prep as soon as samples are loaded, then runs king/challenger generation in parallel with that prep.

1. Load samples and resolve models.
2. Call scoring service to start category prep with sample IDs, prompts, sample indexes, and total sample count.
3. Generate previous king and challenger outputs in parallel as it works today.
4. Send valid output pairs to the scoring service with the category prep id.
5. Scoring service waits for or retrieves the prepared categories for each sample, then runs judges.
6. Receive scoring records that include the GLM-generated categories and judge scores.
7. Build the verdict from returned scoring records and aggregate summary.

Suggested structure:

```text
samples = load_samples(...)

category_prep = scoring_client.start_category_prep(samples)

with ThreadPoolExecutor(max_workers=2):
    king_future = generate(previous_king)
    challenger_future = generate(challenger)

king_results = king_future.result()
challenger_results = challenger_future.result()

scoring_result = score_pairs(
    samples,
    king_results,
    challenger_results,
    category_prep_id=category_prep.id,
)
```

Remote worker should not call Chutes for categories. It only asks the scoring service to start the in-flight prep.

Events to add or adjust:

- `category_prep_started`
- `category_prep_done`
- `category_prep_failed`
- `scoring_started`
- `scoring_batch_done`
- optionally include category generation counters in scoring events, such as `category_generated_count` and `category_generation_errors`

Do not add category artifact upload events or persistent category storage events.

## Scoring Service Flow

Extend `judge_api.py` with a two-phase scoring-service flow:

1. `POST /category-prep` receives sample IDs, prompts, sample indexes, total sample count, and eval/batch identity.
2. The scoring service starts in-flight work for each sample:
   - call GLM 5.2 through Chutes to generate a normal candidate-style response to the same prompt, the same way a challenger would;
   - if Chutes fails, retry that GLM call through OpenRouter `z-ai/glm-5.2` with FP8-only provider routing;
   - call GLM 5.2 through Chutes again to generate exactly 5 scoring categories from the original prompt plus that GLM response;
   - if Chutes fails for category generation, retry through OpenRouter `z-ai/glm-5.2` with FP8-only provider routing;
   - validate and normalize the 5 categories.
3. `POST /score-batch` receives king/challenger outputs plus `category_prep_id`.
4. For each sample, `/score-batch` waits for prepared categories if they are still running.
5. Build pairwise judge prompts using the prompt, generated categories, previous king output, and challenger output.
6. Call existing OpenRouter judges.
7. Parse judge category verdicts.
8. Return scoring records containing the categories, category hash, judge results, sample score, and aggregate summary.

Fallback behavior:

- If `/score-batch` arrives without a `category_prep_id`, generate GLM responses and categories synchronously inside `/score-batch`.
- If category prep fails for a sample, retry synchronous GLM/category generation during `/score-batch`.
- If category prep id is unknown or expired, retry synchronous GLM/category generation during `/score-batch`.
- If GLM-category scoring remains unrecoverable after Chutes and OpenRouter FP8 fallback, fall back to the current fixed-metric scoring path for the affected batch/run.

GLM response generation and category generation can run concurrently across samples within the scoring service, bounded by Chutes concurrency settings. For a given sample, category generation begins only after the GLM candidate-style response is available. Judge calls begin only after categories for that sample are available.

## GLM Response And Category Prompt Contract

First GLM call: generate a normal candidate-style response to the sample prompt, using the same kind of prompt/input the king and challenger answer. This response is not a judge-visible reference answer; it is only the basis for category generation. Run this on Chutes first; if Chutes fails, retry on OpenRouter GLM 5.2 with FP8-only provider routing.

Second GLM call: GLM 5.2 chooses the category topics from the original prompt plus its own candidate-style response, but the output must contain exactly 5 categories. Run this on Chutes first; if Chutes fails, retry on OpenRouter GLM 5.2 with FP8-only provider routing.

Category generation output must be strict JSON only:

```json
{
  "categories": [
    {
      "id": "cat_01",
      "name": "...",
      "description": "...",
      "scoring_guidance": "..."
    }
  ]
}
```

Validation rules:

- Exactly 5 categories.
- Stable IDs: `cat_01`, `cat_02`, `cat_03`, `cat_04`, `cat_05`.
- No duplicate names after normalization.
- Each category must be directly relevant to the prompt, the GLM candidate-style response, and the king/challenger next-turn comparison.
- Reject categories that mention hidden implementation details or judge instructions.
- On malformed output, retry with a repair prompt before failing that sample's scoring record.

## Scoring/Judge Changes

Use the 5 generated categories for each sample as the primary scoring path. Keep the current fixed five global metrics as the fallback scoring path when GLM-category scoring is unrecoverable.

Request shape changes:

- Add a scoring-service category prep request that carries prompt-only sample data.
- Add optional `category_prep_id` to score-batch payloads.
- Remote worker still does not send categories.
- `JudgeSample` remains centered on sample pair inputs; category fields are produced inside the scoring service.

Judge prompt changes:

- Build pairwise judge messages from:
  - conversation/prompt
  - GLM-generated 5 categories
  - previous king output
  - challenger output
- Do not include the GLM candidate-style response in judge prompts.
- Ask each judge to return strict JSON keyed by category ID, where each value is `0`, `1`, or `2`:
  - `0`: equal
  - `1`: model 1 is better
  - `2`: model 2 is better

Aggregation changes:

- Generalize `METRIC_KEYS` into dynamic category IDs per sample.
- Parse category verdicts using the same challenger-position mapping already used today.
- Compute per-sample judge mean across the 5 categories.
- Compute `by_category` across all scored samples.
- Keep `by_judge`.
- Keep the existing win margin `0.06` unless product explicitly changes it.
- When falling back, use the existing fixed-metric aggregation unchanged so score semantics stay compatible with the current system.

Scoring record additions:

- `categories`
- `category_source`
- `category_hash`
- `category_scores`
- `judge_category_scores`
- `category_generation_error` when applicable
- `scoring_mode`, either `glm_categories` or `fixed_metrics_fallback`

Verdict additions:

- `score_breakdown.by_category`
- category prompt version
- GLM 5.2 provider/model metadata from scoring records

## Slack Error Reporting

Add a best-effort Slack webhook notifier for eval/scoring errors.

Configuration:

- Env var: `ALBEDO_SLACK_ERROR_WEBHOOK_URL` or service-local equivalents such as `ALBEDO_JUDGE_SLACK_ERROR_WEBHOOK_URL` and `ALBEDO_REMOTE_SLACK_ERROR_WEBHOOK_URL`.
- Optional env vars: `ALBEDO_SLACK_ERROR_MIN_SEVERITY`, `ALBEDO_SLACK_ERROR_ENV`, `ALBEDO_SLACK_ERROR_TIMEOUT_SECONDS`.
- If no webhook is configured, notifications are disabled and code paths continue normally.

Notification behavior:

- Send Slack notifications for unrecoverable eval/scoring failures, provider exhaustion, category-prep failures after fallback, judge API failures, remote worker failures, and fallback-to-fixed-metrics events.
- Include enough context to triage: `eval_run_id`, `submission_id`, `batch_id` if present, stage/component, fault class/code, provider route attempted, `scoring_mode`, retryability, and a short error message.
- Redact secrets, auth headers, prompts, model outputs, and raw judge/category responses.
- Rate-limit or deduplicate repeated errors by `(eval_run_id, component, fault_code)` so one bad provider incident does not flood Slack.
- Slack send failures must never fail eval/scoring; log them and continue.

Suggested helper:

```text
src/albedo_eval_service/notifications.py
  notify_eval_error(event: EvalErrorNotification) -> None
```

Use the helper from:

- `judge_api.py` for category generation, judge scoring, and fixed-metric fallback events.
- `remote_worker.py` for remote eval execution failures.
- `repository.py` or dispatcher/reconciler paths when an eval is marked failed terminal/retryable.

## Failure Semantics

- GLM response generation and category generation should try Chutes first, then OpenRouter GLM 5.2 with FP8-only provider routing.
- Missing or invalid category JSON after all GLM retries should trigger fixed-metric fallback before failing the eval.
- If too few samples receive valid GLM categories and judge scores, fall back to the current fixed-metric scoring path for the batch/run.
- Only fail with `PROVIDER_FAULT` if both the GLM-category path and the current fixed-metric scoring fallback are unrecoverable.
- Report unrecoverable scoring/eval errors to Slack when a webhook is configured.
- Report fallback-to-fixed-metrics events to Slack as warnings so operators can see degraded scoring even when eval completes.
- If king/challenger generation fails for a sample, skip scoring for that pair as today.
- If category prep expires before `/score-batch`, retry synchronously, then fall back to fixed-metric scoring if the GLM-category path remains unavailable; do not look in Postgres/S3.
- Do not fail because category artifacts are missing; there are no category artifacts.

## Tests

Add focused unit tests:

- Chutes client builds the correct URL, headers, and payload.
- Chutes client parses non-streaming chat completions.
- OpenRouter GLM fallback client sends `model: z-ai/glm-5.2` and `provider.quantizations: ["fp8"]`.
- `POST /category-prep` starts GLM candidate-style response generation before king/challenger outputs exist.
- Scoring service generates a GLM candidate-style response before category generation for each sample.
- Category JSON validation accepts exactly 5 valid categories and rejects malformed/duplicate/missing fields.
- `/score-batch` uses prepared categories when `category_prep_id` is present.
- `/score-batch` can fall back to synchronous category generation when prep is absent or expired.
- `/score-batch` falls back to current fixed-metric scoring when category generation or category validation remains unrecoverable.
- Judge prompts include generated categories and exclude the GLM candidate-style response.
- Judge parser handles dynamic category IDs and challenger-first ordering.
- Aggregation produces `by_category`, `by_judge`, `score_challenger`, `score_king`, and `challenger_won`.
- Remote worker scoring payloads do not include categories.
- No Postgres category table/API or S3 category artifact path is required.
- Slack notifier sends expected payloads for provider errors, fallback events, and terminal eval/scoring failures.
- Slack notifier redacts prompts/outputs/secrets and is best-effort when the webhook fails.

Add integration/smoke tests:

- Mock Chutes + mock OpenRouter judges end-to-end category-prep plus `/score-batch` run.
- Mock remote worker run where category prep overlaps king/challenger generation.
- Failure test where Chutes category generation fails, OpenRouter FP8 fallback succeeds, and aggregation proceeds.
- Failure test where Chutes fails and OpenRouter has no eligible FP8 provider, then scoring falls back to the current fixed-metric path.
- Failure test where both GLM-category scoring and fixed-metric fallback fail, producing a retryable provider fault and Slack error notification.
- Failure test where Slack webhook delivery fails but eval/scoring error handling continues.

Likely existing tests to update:

- `tests/test_judge_core.py`
- `tests/test_judge_api.py`
- `tests/test_remote_worker.py`
- `tests/test_remote_scoring.py` if added or available.
- `tests/test_judge_openrouter.py` only if prompt construction assumptions change.

## Rollout Plan

1. Add Chutes GLM client with scoring-service config/env wiring.
2. Add OpenRouter GLM 5.2 fallback client with FP8-only provider routing.
3. Add GLM candidate-style response generator.
4. Add category prompt builder and validator that consume the GLM response.
5. Add scoring-service in-flight category prep endpoint/state with TTL.
6. Extend remote scoring client and remote worker to start category prep immediately after sample load.
7. Extend `/score-batch` to accept `category_prep_id`, wait for prepared categories, and fall back to synchronous generation if needed.
8. Generalize judge parsing/aggregation from fixed metrics to dynamic category IDs.
9. Preserve the existing fixed-metric scorer as a fallback mode and expose `scoring_mode` in records/summaries.
10. Add category data to scoring records and aggregate summaries.
11. Add Slack webhook notifier and wire it into eval/scoring error paths.
12. Add tests and run unit suite.
13. Run one mock end-to-end eval with category prep overlapping king/challenger generation.
14. Run one mock eval where GLM-category scoring fails and fixed-metric fallback succeeds.
15. Run one mock eval/scoring failure with Slack webhook enabled.
16. Run one canary eval with Chutes primary, forced Chutes-fail/OpenRouter-FP8 fallback, and OpenRouter judges.

## Acceptance Criteria

- GLM 5.2 candidate-style response generation starts through the scoring service while king/challenger generation is running.
- GLM 5.2 candidate-style response generation and category generation use Chutes first, then OpenRouter GLM 5.2 fallback only on Chutes failure.
- OpenRouter GLM fallback uses `z-ai/glm-5.2` and only FP8 provider endpoints.
- If GLM-category scoring cannot recover, scoring falls back to the current fixed-metric scoring path.
- Judges remain on OpenRouter.
- The scoring service generates a GLM candidate-style response, then exactly 5 categories from that response, per scored sample.
- No reusable category cache exists in the plan or implementation.
- No category data is saved to Postgres.
- No GLM response/category/reference data is uploaded to Hippius S3.
- Remote workers do not call Chutes for categories and do not talk to Postgres for categories.
- Scoring records returned by the scoring service include `scoring_mode`. In GLM-category mode they include the generated categories, category source, and category hash.
- Judges score on the generated 5 categories in primary mode, and on the old fixed five metrics only when fallback mode is used.
- Final verdict still sets `challenger_won` and can promote the challenger through the existing win path.
- Configured Slack webhook receives redacted notifications for eval/scoring errors and fallback-to-fixed-metrics warnings without blocking eval/scoring.

## Locked Decisions

- Category count is fixed at 5. GLM 5.2 chooses the category topics, but it must return exactly 5 categories.
- No reusable category cache.
- In-flight scoring-service category prep is allowed only for the active eval request and expires by TTL.
- No category Postgres persistence.
- No GLM/category Hippius S3 uploads.
- The scoring service owns GLM response generation and category generation; remote workers only start prep and later submit outputs.
- Current fixed-metric scoring remains available as the fallback when GLM-category scoring is unrecoverable.
- Judges remain on OpenRouter for this update.
- GLM 5.2 candidate-style response generation and category generation go through Chutes first. OpenRouter fallback is allowed only after Chutes fails and must be restricted to FP8 providers.
- GLM candidate-style responses are not visible to judges.
- Slack webhook reporting is best-effort, redacted, and non-blocking.
