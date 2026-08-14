# Scoring on Albedo (SN97)

How a challenger and the reigning king are compared. Everything below is what
`src/albedo_eval_service/judge_core.py` and `judge_api.py` actually do — constants are quoted from
the code, not from policy documents.

For what the models are scored *on*, see [DATASETS.md](DATASETS.md).

---

## The shape of one eval

An eval is a **duel**, not a benchmark run. Both models answer the same sampled coding-trajectory
prefixes, and each sample is scored by the same checklist for both sides.

```
sample prefix ──► king model      ──► king trajectory      ─┐
              └─► challenger model──► challenger trajectory ─┤
                                                             ├─► judges answer the SAME
reference model ──► reference trajectory ──► checklist ──────┘   yes/no checklist per side
```

Per sample the pipeline is:

1. **Reference trajectory** — a SOTA model (`ALBEDO_JUDGE_SOTA_MODELS`) runs the same task through
   the same simulated-observation loop the candidates face, for
   `ALBEDO_JUDGE_SOTA_TRAJECTORY_TURNS` (8) turns.
2. **Checklist generation** — the evaluator model (`ALBEDO_JUDGE_EVALUATOR_MODEL`) writes
   `ALBEDO_JUDGE_NUM_QUESTIONS` (50) yes/no questions anchored on that reference trajectory.
3. **Judging** — each judge model answers the whole checklist twice: once for the king's
   trajectory, once for the challenger's. Judges never see which side is which, and never see the
   reference (leak-filtered, see below).
4. **Aggregation** — weighted yes-rate per judge → mean across judges → mean across samples.

If the reference cannot be produced (and a re-roll also fails), or the sample carries no prior
context to anchor a reference to, `QuestionService.prepare` raises `QuestionScoringUnavailable` —
there is no task-only fallback checklist. `question_source.question_mode` is always
`"sota_anchored"`.

---

## The checklist

50 questions, split by `STEP_SHARES_PCT` into five sections:

| section | share | what it asks about |
|---|---|---|
| `workflow` | 48% | the actual work: locating, editing, propagating a fix |
| `grounding` | 22% | correctness and evidence — claims tied to real output |
| `terminal` | 12% | how the trajectory ends (submitted, working state) |
| `reaction` | 8% | response to failure — does the next turn change tool/target/approach |
| `length` | 10% | output economy: padding, restated plans, repetition |

Shares set only the per-section question counts (largest-remainder rounding in `step_counts`);
scoring weights are separate.

Each question carries a label that drives scoring:

- **`requires`** — `action` (needs real work), `read` (a read-only step can satisfy it), or
  `neutral` (size/protocol hygiene).

### Enforcement at parse time

Prose rules in a prompt get ignored, so `enforce_question_labels` re-enforces them on the parsed
output and records the drops in `question_source.enforcement_drops`:

- `read_cap` — at most `READ_ONLY_QUESTION_CAP` (5) read-only-passable questions survive.
- `unfolded_avoid` — "avoids X" checks with no action verb are dropped; inaction sweeps them.
- `no_edit_dead_weight` — when the reference never edited in its window, `requires: action`
  questions about completed edits are dropped.

A sample is rejected outright if fewer than `question_floor(n)` = **22%** of the requested questions
come back well-formed (`QUESTION_FLOOR_FRACTION`).

---

## From answers to a score

### 1. Per judge: a weighted yes-rate


`judge_yes_rate` is the plain mean of every answered bit (1/0), measurement/size questions
included — there is no separate size multiplier or per-`requires` weighting in the running code, size questions vote like any other question.

### 2. The measurement gate

Before weighting, `apply_measurement_gate` applies two deterministic corrections per candidate — no
judge involved:

- A candidate that made **no edit** has inaction-conditional do-no-harm questions **removed from its
  denominator**. They are dropped, never awarded: inaction is the adversary, and a free `1` would
  reward it.
- If the reference proved an edit was reachable, the candidate made no edit, **and** its final turn
  is still a read, every `requires: action` question is forced to `0`. Well-groomed exploration
  must not out-score imperfect work.

### 3. Across judges and samples

- `response_score` — mean of the per-judge rates for one side of one sample.
- `aggregate_scores` — mean across samples, per side. **King and challenger scores are
  independent; they do not sum to 1.**
- `by_judge` in the verdict is **challenger-only**. The dashboard recomputes the king's per-judge
  rates from the `SCORING_RESULTS` artifact (`website/monitor.py`).

### 4. The verdict

```python
challenger_beats_king = (score_challenger - score_king) >= CHALLENGER_WIN_MARGIN   # 0.03
```

A **3-point absolute margin** — beating the king by a hair is a loss. Scores are the mean weighted
yes-rate, so the margin is in the same units.

An eval **fails** rather than returning a verdict if fewer than `min_valid_fraction` of samples
scored (`0.8` in code, `ALBEDO_JUDGE_MIN_VALID_FRACTION`) — fault `scoring_invalid`, retryable. A
sample counts as scored only when **every** judge parsed for **both** sides.

---

## Anti-gaming

The checklist is generated per sample from a model's own reference trajectory, so it cannot be
pre-computed. On top of that:

- **Reference leak filter** — `filter_reference_leaks` drops questions that reveal the reference as
  shared history; judges must never learn a "correct" trajectory exists.
- **Near-duplicate questions** — questions too similar to each other are collapsed
  (`_near_duplicate`: Jaccard 0.75 / containment 0.90 / char-ratio 0.87) so one property cannot be
  asked five ways to inflate its weight.
- **Template caps** — at most 4 questions per template key (`_TEMPLATE_MAX_PER_KEY`).
- **Generic hygiene cap** — `GENERIC_HYGIENE_QUESTION_LIMIT` (3) — questions any competent output
  passes.
- **Negative-question cap** — `NEGATIVE_QUESTION_LIMIT` (8).
- **Prompt-injection stripping** — `strip_reply_injection` removes verdict-shaped text
  (`accept` / `weak_pass` / `reject`) and delimiter injections from a candidate's own output before
  a judge reads it, so a model cannot address the judge directly.
- **Unbounded-submit detection** — `is_unbounded_submit_question` catches checks a model passes by
  submitting unconditionally.

---

## Configuration

Judge-side settings live in `src/albedo_eval_service/judge_config.py`, prefix `ALBEDO_JUDGE_`:

| setting | code default | meaning |
|---|---|---|
| `evaluator_model` | `z-ai/glm-5.2` | writes the checklist |
| `sota_models` | `z-ai/glm-5.2` | pool the reference trajectory is drawn from |
| `num_questions` | 50 | checklist size before enforcement drops |
| `sota_trajectory_turns` | 8 | reference trajectory length |
| `min_valid_fraction` | 0.8 | below this the eval fails instead of scoring |
| `max_concurrency_per_model` | 128 | per-model in-flight judge calls |
| `simulation_model` / `simulation_providers` | `deepseek/deepseek-v4-flash-0731` / `cloudflare` | observation simulator (see [DATASETS.md](DATASETS.md)) |

`JUDGE_MODELS` in `judge_core.py` is the default judge panel (`z-ai/glm-5.2`,
`qwen/qwen3.5-397b-a17b`, `deepseek/deepseek-v3.2`); production overrides the panel and count
through the environment, so **the deployed panel may be a single judge** — read the run's
`judge-results` in `scoring-results.jsonl` to see who actually voted.

`ScoringConfig.allowed_scores` is `[0, 1]`: answers are binary, and the verdict reports
`scoring_mode: "binary"`.
