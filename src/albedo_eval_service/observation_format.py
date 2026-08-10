
from __future__ import annotations

import re
from dataclasses import dataclass

RETURNCODE = "returncode"
SWE_AGENT = "swe_agent"
OPENHANDS = "openhands"

TRUNCATION_SENTINEL = "MODEL_RESPONSE_TOKEN_LIMIT_EXCEEDED"
UNCLOSED_THINK_BLOCK_SENTINEL = "MODEL_UNCLOSED_THINK_BLOCK"

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


def classify(observation: str) -> str:
    text = (observation or "").lstrip()
    if text.startswith("<returncode>"):
        return RETURNCODE
    if text.startswith("OBSERVATION:"):
        return SWE_AGENT
    return OPENHANDS


def detect_format(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    observation = _first_observation(messages)
    if observation is not None:
        return classify(observation)
    return RETURNCODE if "mini-coder" in sample_id.casefold() else OPENHANDS


def _first_observation(messages: list[dict[str, str]] | None) -> str | None:
    seen_assistant = False
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and seen_assistant:
            content = str(message.get("content") or "")
            if content.strip():
                return content
    return None


def valid_output(raw: str, fmt: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if fmt == RETURNCODE:
        return (
            text.startswith("<returncode>")
            and "</returncode>" in text
            and "<output>\n" in text
            and text.endswith("\n</output>")
        )
    if fmt == SWE_AGENT:
        return text.startswith("OBSERVATION:")
    return not text.startswith(("<returncode>", "Observation:", "OBSERVATION:"))


def repair_output(raw: str, fmt: str) -> str:
    text = (raw or "").strip()
    if fmt != RETURNCODE or not text.startswith("<returncode>"):
        return text
    if "<output>" in text and "<output>\n" not in text:
        text = text.replace("<output>", "<output>\n", 1)
    if text.endswith("</output>") and not text.endswith("\n</output>"):
        text = text[: -len("</output>")].rstrip("\n") + "\n</output>"
    return text


def wrap(body: str, fmt: str, *, returncode: int = 0) -> str:
    if fmt == RETURNCODE:
        inner = f"{body}\n" if body else ""
        return f"<returncode>{returncode}</returncode>\n<output>\n{inner}</output>"
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    return (
        f"{body}\n[The command completed with exit code {returncode}.]\n"
        f"[Command finished with exit code {returncode}]"
    )


def empty_output(fmt: str) -> str:
    return wrap("", fmt)


def truncation_notice(token_limit: int) -> str:
    return (
        f"{TRUNCATION_SENTINEL}: Model returned over {token_limit} tokens in one response, "
        "stopping conversation."
    )

def is_truncated(text: str) -> bool:
    return TRUNCATION_SENTINEL in (text or "")

def unclosed_think_block_notice() -> str:
    return (
        f"{UNCLOSED_THINK_BLOCK_SENTINEL}: Model returned an unclosed think block, "
        "stopping conversation."
    )

def has_unclosed_think_block(text: str) -> bool:
    return UNCLOSED_THINK_BLOCK_SENTINEL in (text or "")

THINK_PAIR_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_FENCED_SPAN_RE = re.compile(r"```.*?```", re.DOTALL)
_FENCE_PLACEHOLDER = "\x00fence{}\x00"


def mask_fenced_spans(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def _take(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _FENCE_PLACEHOLDER.format(len(spans) - 1)

    return _FENCED_SPAN_RE.sub(_take, text or ""), spans


def unmask_fenced_spans(text: str, spans: list[str]) -> str:
    for index, span in enumerate(spans):
        text = text.replace(_FENCE_PLACEHOLDER.format(index), span)
    return text

MISSING_COMMAND_MESSAGE = "No bash command found in assistant message."


def missing_command_output(fmt: str) -> str:
    return wrap(MISSING_COMMAND_MESSAGE, fmt, returncode=2)


_TRAILER_RE = re.compile(
    r"^\s*\[(?:The command (?:completed|timed out)|Current working directory|"
    r"Python interpreter|Command finished)\b.*\]\s*$"
)
_VIEW_HEADER_RE = re.compile(r"^\s*Here's the (?:result of running|files and directories)\b")


def observation_body(raw: str, fmt: str) -> str:
    """The payload of an observation, with envelope, trailers and view header removed.

    Horizontal whitespace is kept because source indentation is meaningful.
    """
    text = (raw or "").strip()
    if fmt == RETURNCODE:
        match = re.search(r"<output>\n(.*)\n?</output>", text, re.DOTALL)
        return _strip_view_header(match.group(1).strip("\n") if match else "")
    if fmt == SWE_AGENT:
        if not text.startswith("OBSERVATION:"):
            return _strip_view_header(text)
        return _strip_view_header(text[len("OBSERVATION:") :].strip("\n"))
    kept = [line for line in text.splitlines() if not _TRAILER_RE.match(line)]
    return _strip_view_header("\n".join(kept).strip("\n"))


def _strip_view_header(body: str) -> str:
    lines = body.splitlines()
    return "\n".join(lines[1:]).strip("\n") if lines and _VIEW_HEADER_RE.match(lines[0]) else body


def has_content(raw: str, fmt: str) -> bool:
    return bool(observation_body(raw, fmt).strip())


_FIRST_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)


def first_bash_block(assistant_output: str) -> str:
    match = _FIRST_BLOCK_RE.search(assistant_output or "")
    return match.group(1).strip() if match else ""


_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_CHAINED_RE = re.compile(r"&&|\|\||;|\n")
_READ_HEAD_RE = re.compile(r"^\s*(?:cat|nl|head|tail|less|more|sed\s+-n)\b")
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+\S+\s*&&\s*")
_ALWAYS_PRINTS_RE = re.compile(
    r"^\s*(?:git\s+(?:log|show|status|branch)\b|ls\b|pwd\b|tree\b|which\s+\S|wc\s+\S|echo\s+\S)"
)
_WRITE_RE = re.compile(r"(?<![0-9<>])>>?[ \t]*\S|<<-?[ \t]*['\"]?\w")
_MAY_BE_EMPTY_TAIL_RE = re.compile(
    r"\|[ \t]*(?:grep|rg|ag|ack|egrep|fgrep|awk|find|comm|diff|uniq|sort[ \t]+-u)\b"
)


def _unquoted(text: str) -> str:
    """Blank quoted spans so metacharacters inside arguments are not read as chaining:
    `sed -n '1,5p;10,12p' f.py` is one command."""
    return _QUOTED_SPAN_RE.sub(lambda m: "'" + "_" * (len(m.group(0)) - 2) + "'", text)


def requires_output(command: str) -> bool:
    """True when the command cannot legitimately produce an empty observation.

    A read of an existing file prints its content; a read of a missing one prints an error. Writes,
    in-place edits and pipelines ending in a filter that can match nothing are excluded.
    """
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    if not text or _WRITE_RE.search(text) or _MAY_BE_EMPTY_TAIL_RE.search(text):
        return False
    if _ALWAYS_PRINTS_RE.match(text):
        return True
    if not _READ_HEAD_RE.match(text.split("&&")[0]):
        return False
    return bool(re.search(r"[\w./-]*[./][\w./-]+|\s[\w-]+\.\w{1,6}\b", text.split("&&")[0]))


_SED_RANGE_RE = re.compile(r"^sed\s+-n\s+['\"][^'\"]*['\"]\s+\S+(?:\s*\|\s*(?:cat\s+-n|nl\b.*))?$")
_SED_NUM_RANGE_RE = re.compile(r"(\d+)\s*,\s*(\d+)\s*p")
_HEAD_TAIL_TAIL_RE = re.compile(r"\|\s*(?:head|tail)\s+-n?\s*(\d+)\s*$")
_HEAD_TAIL_ONLY_RE = re.compile(r"^(?:head|tail)\s+-n?\s*(\d+)\s+\S+$")


@dataclass(frozen=True)
class CommandContract:
    """The output length the command itself guarantees, when it states one unambiguously."""

    max_lines: int | None = None

    def __bool__(self) -> bool:
        return self.max_lines is not None


def command_contract(command: str) -> CommandContract:
    masked = _unquoted((command or "").strip())
    if not masked:
        return CommandContract()
    capped = _HEAD_TAIL_TAIL_RE.search(masked)          # a trailing pipe caps whatever precedes it
    if capped:
        return CommandContract(max_lines=int(capped.group(1)))
    if _CHAINED_RE.search(masked):
        return CommandContract()
    only = _HEAD_TAIL_ONLY_RE.match(masked)
    if only:
        return CommandContract(max_lines=int(only.group(1)))
    if _SED_RANGE_RE.match(masked):
        ranges = _SED_NUM_RANGE_RE.findall(command)
        if ranges:
            return CommandContract(max_lines=sum(int(b) - int(a) + 1 for a, b in ranges))
    return CommandContract()


def contract_violation(raw: str, fmt: str, contract: CommandContract) -> str | None:
    """The unmet guarantee, or None. Emptiness is left to has_content()/requires_output()."""
    if not contract:
        return None
    lines = [line for line in observation_body(raw, fmt).splitlines() if line.strip()]
    if lines and len(lines) > contract.max_lines:
        return f"too_many_lines:{len(lines)}>{contract.max_lines}"
    return None


_RC_OUTPUT_RE = re.compile(r"(<returncode>\d+</returncode>\n<output>\n).*\n?</output>", re.DOTALL)


def repair_to_contract(raw: str, fmt: str, contract: CommandContract) -> str:
    """Truncate to the length the command states — exactly what `head`/`sed` do.

    Deterministic and content-preserving, so it cannot add fabrication.
    """
    if not contract or not has_content(raw, fmt):
        return raw
    lines, kept, seen = observation_body(raw, fmt).splitlines(), [], 0
    for line in lines:
        if line.strip():
            if seen >= contract.max_lines:
                break
            seen += 1
        kept.append(line)
    return raw if kept == lines else _replace_body(raw, fmt, kept)


def _replace_body(raw: str, fmt: str, lines: list[str]) -> str:
    """Put lines back inside the original envelope, keeping return code, header and trailer."""
    body, text = "\n".join(lines), (raw or "").strip()
    if fmt == RETURNCODE:
        match = _RC_OUTPUT_RE.search(text)
        return f"{match.group(1)}{body}\n</output>" if match else wrap(body, fmt)
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    rest, head, tail = text.splitlines(), [], []
    if rest and _VIEW_HEADER_RE.match(rest[0]):
        head, rest = rest[:1], rest[1:]
    while rest and _TRAILER_RE.match(rest[-1]):
        tail.insert(0, rest.pop())
    return "\n".join(head + lines + tail)
