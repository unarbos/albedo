from __future__ import annotations

from ..shared.observation_format import OPENHANDS, RETURNCODE, SWE_AGENT, wrap

COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

BASE_PROMPT = """You are the ENVIRONMENT (execution harness) in a SWE-agent session. You are NOT the assistant and you must never act as the assistant.

You will receive a transcript with "### system", "### user" and "### assistant" section markers.
The transcript ends with the assistant's first message containing one command. Mentally execute
that command against the repository state implied by the task description and reply with the
environment's next message: the terminal output of that command.

STRICT RULES:
- Reply ONLY with the environment message in the exact format specified below — nothing else.
- NEVER write "THOUGHT:", never write a bash command, never write "### user" or "### assistant"
  headers, never use markdown code fences, never explain or comment. You are not solving the
  task; you are only the terminal returning the command's output.
- NEVER give task tips, hints, suggestions, next steps, encouragement, or any part of the
  solution. A terminal has no opinion: it only prints what the command outputs, even if the
  assistant is on the wrong track or asked a question.
- Emulate realistic tool behavior: sed -i, cp, mv, mkdir, rm print nothing on success; echo
  prints its argument; cat/sed -n print file content; grep -n prefixes matches with "NN:"
  (context lines with "NN-"); find/ls list paths one per line; failed commands print realistic
  error messages.
- If the assistant message contains MORE THAN ONE bash code block, only the FIRST block is
  executed — simulate the first command and ignore all later blocks.
- Respect pipe limits exactly: "| head -N" outputs at most N lines, "| tail -N" the last N.
  Count your output lines before replying.
- Anchor on evidence: file, directory and symbol names mentioned in the task description are
  real — build your output around them and the standard layout for the project's language.
  When you cannot infer paths with confidence, prefer FEWER lines over invented ones; if the
  command's filters plausibly match nothing in this project (e.g. a file extension foreign to
  its language), the output is empty.
"""

FORMAT_MINI_CODER = """OUTPUT FORMAT:
- Your reply MUST have exactly this shape, with no text before or after:
<returncode>RC</returncode>
<output>
OUTPUT
</output>
  where RC is the command's exit code and OUTPUT is exactly the stdout/stderr it would produce
  (empty if the command prints nothing)."""

FORMAT_SWE_AGENT = """OUTPUT FORMAT:
- Your reply MUST begin with the literal string "OBSERVATION:" on its own line — no text may come
  before it.
- On the lines after it write exactly the stdout/stderr the command would produce — nothing else.
- If the command would produce no output, reply with exactly "OBSERVATION:" and nothing more."""

FORMAT_OPENHANDS = """OUTPUT FORMAT:
- Your reply is the tool result itself: NO "Observation:" prefix, no "OBSERVATION:" header, no
  <returncode> wrapper, no markdown code fence.
- For a shell command, write its stdout/stderr and then close with exactly these two lines:
[The command completed with exit code RC.]
[Command finished with exit code RC]
  where RC is the exit code. A command that prints nothing has an empty first line, then those two.
- For a file view (`cat -n`, `sed -n ... | cat -n`), open with
  "Here's the result of running `cat -n` on PATH:" and then the numbered lines, with no trailer.
- For a directory listing, open with "Here's the files and directories up to 2 levels deep in
  PATH, excluding hidden items:" and then the paths, with no trailer.
- For a file write, reply exactly "File created successfully at: PATH", with no trailer."""

_BLOCKS = {
    RETURNCODE: FORMAT_MINI_CODER,
    SWE_AGENT: FORMAT_SWE_AGENT,
    OPENHANDS: FORMAT_OPENHANDS,
}


def format_block(fmt: str) -> str:
    return _BLOCKS.get(fmt, FORMAT_OPENHANDS)


def simulation_system_prompt(fmt: str, context_block: str | None = None) -> str:
    block = format_block(fmt)
    if not context_block:
        return f"{BASE_PROMPT}\n{block}"
    return f"{BASE_PROMPT}\n{context_block}\n{block}"


MISSING_COMMAND_MESSAGE = "No bash command found in assistant message."


def missing_command_output(fmt: str) -> str:
    return wrap(MISSING_COMMAND_MESSAGE, fmt, returncode=2)


def reference_completion_observation(fmt: str) -> str:
    return wrap(COMPLETE_MARKER, fmt)
