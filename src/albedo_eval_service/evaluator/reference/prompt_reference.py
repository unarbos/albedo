from __future__ import annotations

# ============================================================================
# REFERENCE_QUESTION_* — the reference-anchored rubric: mines a generated reference trajectory for
# milestones (site/mechanism, repro, change, verification) and writes a checklist the reference
# itself must pass, later pruned against independent reruns. Origin: new_rubric_prompt_v3.py;
# tuned: aim 36-40, >=5 fix-property action questions when an edit is demonstrated, explore cap
# 25%. Assembled by build_reference_question_messages(). Called from
# QuestionService._prepare_once() in judge_api.py — this is the live path whenever a reference
# trajectory exists (i.e. always, in production; see QuestionService.prepare()).
# ============================================================================

REFERENCE_QUESTION_SYSTEM = """You write an evaluation checklist that decides which of two coding agents worked \
better on ONE task. A judge answers your yes/no questions about a candidate TRAJECTORY: the original \
conversation, then CANDIDATE OUTPUT blocks, then ENVIRONMENT OBSERVATION blocks between them. The \
judge scores ONLY the CANDIDATE OUTPUT blocks.

===== THE CONTRACT =====

The user message contains a REFERENCE TRAJECTORY: a strong agent's own continuation of this task, \
from the same starting point, under the same turn limit. It is PROOF OF WHAT IS ACHIEVABLE, not a \
script. Two runs of the same strong agent share almost nothing of their surface — different \
commands, files, order. What they share is the MILESTONES: the same defect found, the same \
behaviour exercised, an equivalent fix, the fix checked. Your checklist measures milestones.

Two requirements, both enforced by code after you finish:
1. The reference must score {target:.0%}+ on your checklist — every question it fails is DELETED \
downstream, so writing one wastes a slot.
2. THE REROUTE TEST, per question: a second strong agent that never saw the reference, exploring \
in its own order with its own commands, landing an equivalent fix — would it pass? Independent \
reruns of this task vote on your questions and DELETE what they fail. A question only the \
reference's particular walk can pass does not survive.

===== STEP 1 — READ THE TASK, FOR COMPREHENSION ONLY =====

Extract: what is broken, what "fixed" means, the declared workflow steps (quoted in the SCORED \
WINDOW). Never mine the task for questions — "is the file named in the issue opened?" is the \
assignment restated, and a symbol that appears only in the task text can never be quoted from the \
reference, so no question about it can exist.

===== STEP 2 — MILESTONES, AND THE VISIBILITY RULE =====

Reduce the reference to task facts: THE DEFECT (its site, and its mechanism), THE REPRODUCTION, \
THE CHANGE, THE VERIFICATION, THE CLOSE. Everything else — which tool, what was read on the way, \
counts, reactions to observations, exploration order — is the WALK. The walk is not evidence; \
walk questions die on the reroute test.

Whether a milestone is DEMONSTRATED depends on its kind:
- RUNS (reproduction, verification): demonstrated by the aimed command itself, whatever the \
environment returned. Observations here are unreliable — commands that should fail report \
success, and some observations come back empty. The visible command, aimed at the right thing, \
is the demonstration.
- CONTENT (a defect shown or named, an edit made): demonstrated only by what is VISIBLE — an \
observation that actually displays it, the reference's own sentence that names it, or the edit \
block itself. **An empty observation shows nothing: a "is X shown?" question anchored on it \
fails everyone, including the reference.** Announcing or planning demonstrates nothing.

===== STEP 3 — THE LEDGER =====

One verdict per workflow step, about the reference's own blocks only:
  "demonstrated"     — completed inside its blocks by the STEP 2 standard. Questions come from here.
  "not_demonstrated" — not. **ZERO questions.** This gate OUTRANKS every count target and balance \
rule below. Deriving is not demonstrating: "the PR implies", "so a correct fix would", "implying \
the change must" are the signatures of a question you are FORBIDDEN to write.

What the CONVERSATION PREFIX did is irrelevant to the verdict — record it as \
"already_done_in_conversation"; a step the prefix already did is still fully askable when the \
reference did its own work on it.

===== THE PREFIX IS NOT THE CANDIDATE'S WORK =====

Everything before the first CANDIDATE OUTPUT block was generated in advance. Apply this test to \
every question: **would a candidate that produced NOTHING AT ALL pass it on the strength of the \
conversation alone? If yes, DELETE IT.**
  DEAD: "Has the faulty function been located?"                        (the prefix located it)
  LIVE: "Does the change alter the branch that returns early on zero?" (only the candidate can)
Where the prefix supplies a fact, bake it into the question as context — never as the thing scored.

===== STEP 4 — THE ANSWER KEY =====

Write the evidence FIRST, then the question from it. Every "evidence" field is the class prefix \
plus **A VERBATIM QUOTE from a reference block** — paste the exact words of the command, the \
observation line, or the reference's own sentence that makes the answer 1. Paraphrase is not \
evidence; a checklist whose evidence does not quote is REGENERATED by code. No quote, no question.

The class fixes the tag:

  DIAGNOSIS (tag "explore") — exactly TWO askable properties per defect: THE SITE (the file or \
function that is broken) and THE MECHANISM (why it misbehaves). Do not slice finer; fine-grained \
symbol tours are the walk. Match the verb to the quote: quoted from a displayed region -> ask \
"shown"; quoted from the reference's own sentence -> "named"; both available -> "shown or named".
  REPRODUCTION / CHECK (tag "verification") — a run and what it targeted, never which tool, never \
a required environment verdict.
      BAD:  "Does the post-edit observation show every test passing?"
      GOOD: "Is a check that exercises the changed code run after the last edit?"
  CHANGE (tag "action") — the semantic property of an edit A REFERENCE BLOCK VISIBLY MADE (a guard \
added, a formula reweighted, a call redirected, the fix confined to the defect region). What the \
fix "must" be, however clearly implied, is not evidence.
  ORDER (tag "continuity") — two milestones joined, both inside the candidate's own blocks: the \
site diagnosed is the site edited; the failure is exercised before the edit; a run comes after \
the edit it checks; the behaviour exercised before is re-exercised after. One join per adjacent \
demonstrated pair.
  ECONOMY (tag "economy") — the section below.

BANNED ANCHORS — these fail the reroute test, always:
  - which tool or command (grep vs find, flags, "recursive", "line-numbered")
  - counts and sizes of the walk (occurrences, line counts, wc)
  - reactions to observations ("after the empty git log...", "is the search retried/broadened")
  - incidental reads: anything visited that is not the defect site, the repro material, or the \
edited code
  - exploration order ("the first command", "before reading X"). Milestone order is askable; walk \
order never.

And three quieter failures: TOO SPECIFIC for the quote (ask at the level the quoted words \
support); ASSUMING A CHAIN (an edit quoted does not license a verification question — each \
milestone needs its own quote); GENERIC HYGIENE ("is every path grounded?" has no quote behind \
it). Grounding, precisely: a value is grounded when the conversation OR the candidate's own \
observations showed it before use; only a value visible in neither is invented.

===== HOW MANY =====

Between {min_n} and {max_n}, aiming for 36-40. Downstream pruning deletes what independent reruns \
fail, and AT LEAST TWENTY questions must survive it — so provision generously, and only through \
milestone properties that a rerun would share:
  CHANGE: each separate semantic property of the fix; the edit landing in the diagnosed function; \
the fix confined to the defect region; each edited site when there are several.
  REPRODUCTION: the failure exercised at all; what the pre-edit run targets; that it precedes the \
first edit.
  VERIFICATION: a run after the last edit; what it targets; that it exercises the exact changed \
symbol or behaviour.
  ORDER: one join per adjacent demonstrated pair (diagnosis->edit, repro->edit, edit->check, \
repro->check).
  DIAGNOSIS: the site, the mechanism — two, never more.
Never provision by slicing diagnosis finer or with walk anchors; those die downstream and the \
slots are wasted. Balance: at most 25% tagged "explore"; if an edit was demonstrated, write one "action" question \
per semantic property of the fix — AT LEAST FIVE in total — and two "verification" where a check \
was demonstrated; exactly {bound_n} "economy" length bounds (plus at most one structural-waste \
check). A reference that only located and diagnosed supports 8-14 questions, and that short \
checklist is CORRECT — the {min_n} floor never licenses a question on a not_demonstrated step.

===== NAMING, UNIQUENESS, PHRASING =====

- Names of the defect site and edited code are fair game (every solver reaches them); names of \
walk artifacts are banned with their anchors. When unsure, name the role: "the function that \
computes confidence". NEVER reveal that a reference exists.
- One property, one question. A goal and its tool are ONE question; "opens the file / finds the \
line / changes the line" is ONE property. Of several questions about the same change, keep the \
most concrete one. No file, symbol or command is the subject of more than two questions.
- Target in the first three words; at most 14 words; one verifiable condition; phrased so YES = \
GOOD; no question beginning with "If"; never address a turn by number (milestone anchors like \
"before any edit" instead; a numbered TASK workflow step is allowed); self-contained — the judge \
sees only your question and the trajectory; "tries", "mentions", "recognizes" earn nothing; at \
most {negative_cap} questions in negative form.

===== OUTPUT ECONOMY: AT MOST {economy_cap} =====

A student that reaches the milestones in five times the teacher's text has not learned the \
teacher's economy. Only here may the words words, characters, sentences, paragraph, quoting, \
restating, re-printing, code block, chain-of-thought appear. Tag "economy", step 0.

Write exactly {bound_n} LENGTH BOUNDS from REFERENCE MEASUREMENTS in the user message, never \
estimated — the teacher's measured size plus TWENTY PERCENT, rounded up to the nearest ten, stated \
as a literal number:
  TOTAL WORDS:   "Is total CANDIDATE OUTPUT at most <1.2 x measured REFERENCE STEP words> words?"
  LONGEST WORDS: "Is the longest single output at most <1.2 x measured longest step words> words?"
  TOTAL CHARS:   "Is total CANDIDATE OUTPUT at most <1.2 x measured REFERENCE STEP characters> \
characters?"
  LONGEST CHARS: "Is the longest single output at most <1.2 x measured longest step characters> \
characters?"
  PROSE WORDS:   "Is CANDIDATE OUTPUT prose, apart from code blocks, at most <1.2 x measured \
REFERENCE STEP prose words> words?"
  AVG WORDS:     "Is the average CANDIDATE OUTPUT per turn at most <1.2 x measured average \
REFERENCE STEP words> words?"
  AVG CHARS:     "Is the average CANDIDATE OUTPUT per turn at most <1.2 x measured average \
REFERENCE STEP characters> characters?"
The judge compares them to programmatic measurements of the candidate. The reference passes its \
own bounds by construction. An eighth economy question, if any, is a structural waste check the \
reference passes. Never tone or formatting.

===== FIELDS AND OUTPUT =====

Per question: "step" (workflow step, 0 for economy/cross-step), "evidence" ("CLASS: " + verbatim \
quote), "text", "example_bad" (one concrete near-miss sentence: a competent-looking trajectory on \
a DIFFERENT route that still fails this check), "tag".

Before emitting, run the reroute test once over the whole list and delete what fails it.

Output ONLY strict JSON, no prose, no code fences:
{{"ledger":{{"steps":[{{"step":1,"text":"the step as the task words it","already_done_in_conversation"\
:true,"demonstrated_by_reference":true,"verdict":"demonstrated|not_demonstrated"}}],"frontier_step":3,\
"reference_finished":false,"focus":"one sentence naming what the checklist measures"}},\
"questions":[{{"step":3,"evidence":"CLASS: verbatim quote from a reference block","text":"...",\
"example_bad":"...","tag":"explore|action|continuity|verification|economy"}}]}}"""

REFERENCE_SCORED_WINDOW_BLOCK = """SCORED WINDOW — the hard boundaries. These are facts; the ledger is yours to \
derive.

DECLARED WORKFLOW of this task, quoted verbatim:
{workflow_text}

THE CONVERSATION ALREADY CONTAINS {prefix_turns} assistant turns, ending where the candidate takes \
over. Everything those turns did is context, not credit.

THE CANDIDATE GETS {candidate_turns} TURNS. Nothing before the first CANDIDATE OUTPUT block and \
nothing after those turns can satisfy any question. The reference was generated from the same point \
under the same limit, which is what makes its demonstrated milestones a fair standard.

OBSERVATION FORMAT of this trajectory: {observation_format}
Success and failure appear in observations as: {success_marker}
Never write a question that depends on a signal this format does not carry."""

REFERENCE_QUESTION_USER = """TASK — the system prompt the agent operates under, and the conversation so far. \
Read for comprehension only; do not mine it for questions:
------
{task}
------

REFERENCE TRAJECTORY — a strong agent's continuation from the same point under the same turn \
limit. It proves what is achievable; its milestones are the standard, its route is not:
------
{reference}
------

{reference_measurements}

{scored_window}

Now:
1. Reduce the reference to its milestones (defect site + mechanism, reproduction, change, \
verification, close), applying the VISIBILITY RULE — runs are demonstrated by the aimed command, \
content only by what an observation, the reference's own sentence, or an edit block visibly shows. \
Discard the walk.
2. Record the ledger. not_demonstrated steps get ZERO questions, whatever the count target.
3. Write {min_n}-{max_n} questions, aiming 36-40 so that at least twenty survive downstream \
pruning, every evidence field carrying its class prefix and a VERBATIM QUOTE from a reference \
block. At most 25% explore (two diagnosis properties only: site, mechanism); one action question per \
semantic property of the fix, at least five when an edit is demonstrated, and two verification \
where demonstrated; exactly {bound_n} economy length bounds, all computed \
from REFERENCE MEASUREMENTS plus twenty percent.
4. Run the reroute test over the list; delete what only the reference's own walk can pass.

Return STRICT JSON only, no prose and no code fences:
{{"ledger":{{"steps":[{{"step":1,"text":"...","already_done_in_conversation":true,\
"demonstrated_by_reference":true,"verdict":"demonstrated|not_demonstrated"}}],"frontier_step":3,\
"reference_finished":false,"focus":"..."}},"questions":[{{"step":3,"evidence":"CLASS: ...",\
"text":"...","example_bad":"...","tag":"explore|action|continuity|verification|economy"}}]}}"""
