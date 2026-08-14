from __future__ import annotations

from dataclasses import dataclass

# ============================================================================
# BEHAVIOR_* — per-trim-phase process checks (precision reads, grounded edits, no waste),
# independent of any reference trajectory. Assembled by build_behavior_messages(),
# filtered by filter_behavior_questions(). Called from
# QuestionService._prepare_once() in judge_api.py whenever num_questions >= 3 * BEHAVIOR_K.
# ============================================================================

BEHAVIOR_K = 6

_WASTE_NOTE = """ WASTE MEANS ONLY: re-issuing a command whose output was already received, \
verbatim retries of failed commands, or spending most turns on one repeated action type. \
Re-reading a region AFTER changing it is verification, not waste. The existence, creation, \
timing, or repetition of reproduction scripts is OUTSIDE this behaviour — never mention \
reproduction at all."""


@dataclass(frozen=True)
class BehaviorPart:
    name: str
    text: str


@dataclass(frozen=True)
class BehaviorPhase:
    head: str
    parts: tuple[BehaviorPart, ...]
    gold: tuple[str, ...]


BEHAVIOR_PHASES: dict[str, BehaviorPhase] = {
    "cold": BehaviorPhase(
        head="Opening trim: the conversation was cut near its start; competent work is "
        "exploration that CONVERGES. A completed or verified fix must NOT be required by any "
        "question.",
        parts=(
            BehaviorPart(
                "precision_reads",
                """PRECISION READS: code is read with explicit line ranges, positioned by lines or \
symbols earlier output showed; later reads zoom into regions earlier searches located; no \
whole-file dumps of large files.""",
            ),
            BehaviorPart(
                "issue_anchored_narrowing",
                """ISSUE-ANCHORED NARROWING: searches use identifiers from the issue text; each \
successive search narrows using what prior output showed; the faulty region is displayed \
before any conclusion about it.""",
            ),
            BehaviorPart(
                "convergence_and_orientation",
                """CONVERGENCE AND ORIENTATION: repository state is checked with a git command; \
the final output commits to one specific file or symbol as the target, with a stated reason.""",
            ),
        ),
        gold=(
            "Does the candidate read code with explicit line ranges instead of whole-file dumps?",
            "Is a ranged read positioned by a line or symbol a previous observation showed?",
            "Does a search use identifiers taken from the issue text itself?",
            "Does the candidate check repository state with a git command before concluding?",
        ),
    ),
    "pre_edit": BehaviorPhase(
        head="At-the-fix trim: the conversation was cut just before the first edit; competent "
        "work makes a grounded, well-aimed edit.",
        parts=(
            BehaviorPart(
                "grounded_targets",
                """GROUNDED TARGETS: every modified file was displayed beforehand; every path and \
symbol used appeared in prior context or observations; line numbers in edits are consistent \
with displayed line numbers. Phrase each so a single invented target fails the question.""",
            ),
            BehaviorPart(
                "anchored_edit",
                """ANCHORED EDIT: the exact region being changed is displayed shortly before the \
modifying command; the edit targets the symbol the displayed code showed as faulty; claims \
about earlier output are backed by visibly displayed content.""",
            ),
            BehaviorPart(
                "no_waste",
                """NO WASTE: no command is re-issued after its output was already received; \
after a failed command the next differs in tool, target, or arguments."""
                + _WASTE_NOTE,
            ),
        ),
        gold=(
            "Is every file the candidate modifies previously displayed in an observation?",
            "Is the exact region being changed displayed shortly before the modifying command?",
            "Does the candidate avoid re-issuing a command whose output it already received?",
        ),
    ),
    "at_edit": BehaviorPhase(
        head="Verification trim: the conversation already contains the first edit; competent "
        "work verifies it precisely and closes out.",
        parts=(
            BehaviorPart(
                "scoped_tests_and_diff",
                """SCOPED TESTS AND DIFF: the repository's existing test suite is run scoped to \
the changed area; the accumulated change is reviewed with git diff before finishing; the diff \
is confined to files the issue implicates.""",
            ),
            BehaviorPart(
                "targeted_verification",
                """TARGETED VERIFICATION: a ranged read verifies the changed region's exact \
content after modification; each verification command is aimed at the changed symbol; closing \
claims match what the last observations visibly show.""",
            ),
            BehaviorPart(
                "no_waste",
                """NO WASTE: no command whose output was already received is repeated; most \
turns are not one repeated action type."""
                + _WASTE_NOTE,
            ),
        ),
        gold=(
            "Does the candidate run the existing test suite scoped to the changed area?",
            "Does the candidate review the accumulated change with git diff before finishing?",
            "Does a ranged read verify the changed region's exact content after modification?",
        ),
    ),
}

BEHAVIOR_COMMON = """Each question: yes/no, at most 16 words, YES = the good behaviour is \
present, judgeable from the trajectory text alone, one property per question. NEVER ask for: \
writing or running a reproduction script, seeing a failure before editing, running code after \
every edit, or generic diligence. Return STRICT JSON only: \
{"questions":[{"text":"...","example_bad":"..."}]}"""

NO_TESTS_NOTE = (
    "\nNo test suite is visible in this sample's context — replace test-suite "
    "questions with additional diff-review or targeted-verification variants."
)
