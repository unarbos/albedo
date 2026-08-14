from __future__ import annotations

from dataclasses import dataclass

from albedo_eval_service.shared.observation_format import has_unclosed_think_block, is_truncated

_CODE_KEYWORDS = {
    "def",
    "class",
    "import",
    "from",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "as",
    "lambda",
    "yield",
    "raise",
    "assert",
    "pass",
    "del",
    "global",
    "nonlocal",
    "and",
    "or",
    "not",
    "in",
    "is",
    "async",
    "await",
    "print",
    "self",
    "function",
    "const",
    "let",
    "var",
    "switch",
    "case",
    "break",
    "continue",
    "new",
    "extends",
    "export",
    "default",
    "typeof",
    "instanceof",
    "this",
    "super",
    "catch",
    "throw",
    "interface",
    "type",
    "enum",
    "namespace",
    "public",
    "private",
    "protected",
    "readonly",
    "console",
    "static",
    "final",
    "void",
    "int",
    "char",
    "bool",
    "string",
    "struct",
    "using",
    "package",
    "implements",
    "abstract",
    "override",
    "virtual",
    "template",
    "typename",
    "include",
    "#include",
    "std",
    "sizeof",
    "typedef",
    "func",
    "range",
    "map",
    "chan",
    "go",
    "defer",
    "select",
    "nil",
    "fn",
    "mut",
    "impl",
    "trait",
    "pub",
    "use",
    "mod",
    "match",
    "loop",
    "where",
    "unsafe",
    "dyn",
    "crate",
    "end",
    "module",
    "require",
    "unless",
    "until",
    "do",
    "done",
    "then",
    "fi",
    "begin",
    "rescue",
    "ensure",
    "puts",
    "elsif",
    "foreach",
    "echo",
    "grep",
    "rgrep",
    "tree",
    "find",
    "cat",
    "sed",
    "awk",
    "ls",
    "cd",
    "head",
    "tail",
    "sort",
    "uniq",
    "wc",
    "cut",
    "tr",
    "xargs",
    "diff",
    "patch",
    "cp",
    "mv",
    "rm",
    "rmdir",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "ln",
    "pwd",
    "which",
    "tee",
    "printf",
    "export",
    "source",
    "env",
    "stat",
    "basename",
    "dirname",
    "tar",
    "curl",
    "wget",
    "ssh",
    "rsync",
    "git",
    "make",
    "cmake",
    "gcc",
    "clang",
    "node",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "cargo",
    "rustc",
    "java",
    "javac",
    "mvn",
    "gradle",
    "ruby",
    "gem",
    "bundle",
    "rake",
    "php",
    "composer",
    "perl",
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "tox",
    "ruff",
    "black",
    "mypy",
    "flake8",
    "pylint",
    "bash",
    "sh",
    "docker",
    "kubectl",
}


@dataclass
class CheckResult:
    passed: bool
    reason: str = ""


def check_unclosed_think_block(text: str) -> CheckResult:
    if has_unclosed_think_block(text):
        return CheckResult(False, "response contains an unclosed think block")
    return CheckResult(True)


def check_truncated(text: str) -> CheckResult:
    if is_truncated(text):
        return CheckResult(False, "response exceeded the model response token limit")
    return CheckResult(True)


def check_empty(text: str) -> CheckResult:
    if not text.strip():
        return CheckResult(False, "empty response")
    return CheckResult(True)


def check_length(text: str, min_tokens: int = 5) -> CheckResult:
    tokens = text.split()
    if len(tokens) < min_tokens:
        return CheckResult(False, f"too short ({len(tokens)} tokens, min={min_tokens})")
    return CheckResult(True)


def check_repetition(text: str, max_repetition: float = 0.85) -> CheckResult:
    tokens = text.split()
    if len(tokens) < 3:
        return CheckResult(True)
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    diversity = len(set(trigrams)) / len(trigrams)
    if diversity < (1.0 - max_repetition):
        return CheckResult(False, f"repetitive output (diversity={diversity:.2f})")
    return CheckResult(True)


def check_encoding(text: str) -> CheckResult:
    if len(text) > 20 and sum(1 for c in text if ord(c) > 127) / len(text) > 0.6:
        return CheckResult(False, "excessive non-ASCII (encoding broken)")
    return CheckResult(True)


def check_vocabulary(text: str, min_ratio: float = 0.3) -> CheckResult:
    tokens = text.lower().split()
    if len(tokens) < 8:
        return CheckResult(True)
    ratio = len(set(tokens)) / len(tokens)
    if ratio < min_ratio:
        return CheckResult(False, f"low vocabulary diversity ({ratio:.2f}, min={min_ratio})")
    return CheckResult(True)


def check_one(
    text: str,
    min_tokens: int = 5,
    max_repetition: float = 0.85,
    min_vocab_ratio: float = 0.3,
) -> CheckResult:
    for result in [
        check_unclosed_think_block(text),
        check_truncated(text),
        check_empty(text),
        check_length(text, min_tokens),
        check_repetition(text, max_repetition),
        check_encoding(text),
        check_vocabulary(text, min_vocab_ratio),
    ]:
        if not result.passed:
            return result
    return CheckResult(True)


def check_collapsed(responses: list[str]) -> CheckResult:
    if len(responses) < 2:
        return CheckResult(True)
    if len({r.strip()[:100] for r in responses}) == 1:
        return CheckResult(False, "identical response to all prompts (collapsed model)")
    return CheckResult(True)


def check_uniform_length(responses: list[str]) -> CheckResult:
    lengths = [len(r.split()) for r in responses]
    if len(responses) >= 2 and len(set(lengths)) == 1:
        return CheckResult(
            False, f"all responses identical length ({lengths[0]} tokens) - possible collapse"
        )
    return CheckResult(True)


def check_code_present(responses: list[str]) -> CheckResult:
    for resp in responses:
        if set(resp.lower().split()) & _CODE_KEYWORDS:
            return CheckResult(True)
    return CheckResult(False, "no code keywords in any response (def/return/import/etc)")
