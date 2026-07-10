"""`albedo on` — full-screen TUI (prompt_toolkit).

A pinned `albedo>` input bar at the bottom, a scrollable log (command history + background
output) in the middle, and the publish steps checklist (✓/✗) up top. The same subcommands run
inside the TUI and headless. Background chatter (e.g. "connecting to finney", download progress)
is captured into the log instead of corrupting the screen. PgUp/PgDn + mouse wheel scroll the log;
↑/↓ recall command history. Ctrl+C / Ctrl+Q / `off` exits.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import threading

from miner import check_commits, commit as commit_mod, publish, register as register_mod, upload, validate

_GLYPH = {
    "pending": "<grey>⬜</grey>",
    "running": "<ansiyellow>⏳</ansiyellow>",
    "ok": "<ansigreen>✓</ansigreen>",
    "fail": "<ansired>✗</ansired>",
}
_HELP = ("check-model (--path | --repo --digest) · upload --path --namespace --name · "
         "register --coldkey --hotkey · commit --repo --digest --coldkey --hotkey · "
         "check-commit [--hotkey] · publish --path --namespace --name --coldkey --hotkey · help · off")


class _State:
    def __init__(self):
        self.steps = {k: "pending" for k, _ in publish.STEPS}
        self.detail = {k: "" for k, _ in publish.STEPS}

    def reset_steps(self):
        self.steps = {k: "pending" for k, _ in publish.STEPS}
        self.detail = {k: "" for k, _ in publish.STEPS}


class _LogWriter:
    """File-like sink that turns stray stdout/stderr/loguru into log lines."""

    def __init__(self, log):
        self._log = log
        self._buf = ""

    def write(self, s: str):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log(line.rstrip())
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._log(self._buf.rstrip())
            self._buf = ""


def _clipboard_text() -> str:
    """Read the system clipboard (for Ctrl+V). Handles WSL (Windows clipboard via PowerShell),
    Wayland (wl-paste), and X11 (xclip/xsel). Returns '' if none available."""
    import shutil
    import subprocess

    for cmd in (["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"],
                ["wl-paste", "-n"],
                ["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "-b", "-o"]):
        if shutil.which(cmd[0]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout:
                    return r.stdout.rstrip("\r\n")
            except Exception:  # noqa: BLE001
                continue
    return ""


def _parse_opts(tokens: list[str]) -> dict:
    opts: dict = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[key] = tokens[i + 1]; i += 2
            else:
                opts[key] = True; i += 1
        else:
            i += 1
    return opts


def _dispatch(cmd, opts, state, log, confirm, refresh, netuid, network):
    """Run one command. log(str) appends to the log; refresh() repaints the steps; confirm(text)->bool."""

    def on_step(key, status, detail=""):
        state.steps[key] = status
        if detail:
            state.detail[key] = detail
        refresh()

    if cmd == "help":
        log(_HELP)
    elif cmd == "check-model":
        if opts.get("path"):
            ok, res = validate.validate_local(opts["path"])
        elif opts.get("repo") and opts.get("digest"):
            ok, res = validate.validate_remote(opts["repo"], opts["digest"])
        else:
            log("check-model needs --path OR (--repo and --digest)"); return
        for k, v in res.items():
            log(f"  {k}: {'PASS' if v['ok'] else 'FAIL — ' + v['reason']}")
        log("VALID" if ok else "INVALID")
    elif cmd == "upload":
        repo = opts.get("repo") or upload.make_repo(opts["namespace"], opts["name"])
        ref = upload.upload_model(opts["path"], repo)
        log(f"uploaded {ref.immutable_ref}")
        log(f"  reveal: {commit_mod.build_reveal(ref)}")
    elif cmd == "commit":
        from config_validation.models import ModelRef
        ref = ModelRef(repo=opts["repo"], digest=opts["digest"])
        nu, nw = int(opts.get("netuid", netuid)), opts.get("network", network)
        ss58, reg = commit_mod.registration_check(opts["coldkey"], opts["hotkey"], nu, nw)
        log(f"  hotkey {ss58} — {'registered' if reg else 'NOT registered'}")
        if not reg:
            log("aborted — hotkey not registered"); return
        if not confirm(commit_mod.preview(ref, ss58=ss58, coldkey=opts["coldkey"],
                                          hotkey=opts["hotkey"], netuid=nu, network=nw)):
            log("aborted — nothing committed"); return
        commit_mod.submit(ref, coldkey=opts["coldkey"], hotkey=opts["hotkey"], netuid=nu, network=nw)
        log("committed")
    elif cmd == "register":
        nu, nw = int(opts.get("netuid", netuid)), opts.get("network", network)
        uid = register_mod.register(opts["coldkey"], opts["hotkey"], nu, nw, confirm=confirm)
        log(f"uid {uid}" if uid is not None else "not registered")
    elif cmd == "check-commit":
        commits = check_commits.fetch(int(opts.get("netuid", netuid)),
                                      opts.get("network", network), opts.get("hotkey"))
        for c in commits:
            log(f"  block={c.block_number} {c.model_uri}")
        log(f"{len(commits)} commit(s)")
    elif cmd == "publish":
        state.reset_steps()
        ok, _ = publish.run(
            path=opts["path"], namespace=opts["namespace"], name=opts["name"],
            coldkey=opts["coldkey"], hotkey=opts["hotkey"],
            netuid=int(opts.get("netuid", netuid)), network=opts.get("network", network),
            on_step=on_step, log=log, confirm=confirm)
        log("PUBLISHED" if ok else "stopped")
    else:
        log(f"unknown command: {cmd}  (try `help`)")


def run() -> None:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.document import Document
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.mouse_events import MouseEventType
        from prompt_toolkit.widgets import Frame, TextArea
    except ImportError:
        print("prompt_toolkit not installed: pip install prompt_toolkit")
        return

    state = _State()
    netuid = int(os.environ.get("CHAIN_NETUID", "97"))
    network = os.environ.get("CHAIN_NETWORK", "finney")

    ref: dict = {"app": None, "loop": None}
    confirming = {"on": False}
    cresult = {"v": False}
    cevent = threading.Event()
    follow = {"on": True}  # auto-scroll to newest output unless the user scrolled up

    log_area = TextArea(text="welcome — type a command (`help`), `off` to quit.\n",
                        read_only=True, scrollbar=True, focusable=True, wrap_lines=True)
    input_area = TextArea(height=1, prompt="albedo> ", multiline=False,
                          history=InMemoryHistory())

    def _ui_append(line: str):
        text = log_area.text + line + "\n"
        # When following, keep the cursor at the end so the view tracks newest output;
        # when the user has scrolled up, leave the cursor put so the view stays still.
        pos = len(text) if follow["on"] else log_area.buffer.cursor_position
        log_area.buffer.set_document(Document(text, cursor_position=pos), bypass_readonly=True)
        if ref["app"]:
            ref["app"].invalidate()

    def log(line: str):
        loop = ref["loop"]
        if loop:
            loop.call_soon_threadsafe(_ui_append, line)
        else:
            _ui_append(line)

    def refresh():
        loop = ref["loop"]
        if loop and ref["app"]:
            loop.call_soon_threadsafe(ref["app"].invalidate)

    def confirm(text: str) -> bool:
        log(text)
        log("proceed with commit?  press  y / n")
        cresult["v"] = False
        cevent.clear()
        confirming["on"] = True
        refresh()
        cevent.wait()
        return cresult["v"]

    writer = _LogWriter(log)
    from loguru import logger as _logger
    _logger.remove()
    _logger.add(writer.write, format="{message}", level="INFO", colorize=False)

    def run_command(line: str):
        tokens = shlex.split(line)
        cmd, opts = tokens[0], _parse_opts(tokens[1:])
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            try:
                _dispatch(cmd, opts, state, log, confirm, refresh, netuid, network)
            except Exception as exc:  # noqa: BLE001
                log(f"error: {exc}")

    def accept(buff) -> bool:
        line = buff.text.strip()
        if not line:
            return False
        if line in ("off", "quit", "exit"):
            ref["app"].exit()
            return False
        follow["on"] = True  # snap back to newest output for the new command
        log(f"albedo> {line}")
        ref["loop"] = asyncio.get_running_loop()
        ref["loop"].run_in_executor(None, run_command, line)
        return False  # clear the input

    input_area.accept_handler = accept

    def header_text():
        return HTML(f"<b><ansicyan>albedo</ansicyan></b> — miner publish console\n<grey>{_HELP}</grey>")

    def steps_text():
        rows = [f"{_GLYPH[state.steps[k]]} {label}"
                + (f"   <grey>{state.detail.get(k, '')}</grey>" if state.detail.get(k) else "")
                for k, label in publish.STEPS]
        return HTML("\n".join(rows))

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("c-q")
    def _(event):
        event.app.exit()

    @kb.add("c-v")   # works if the terminal forwards Ctrl+V (most do; VS Code grabs it)
    @kb.add("c-y")   # always reaches the app — use this in VS Code
    def _(event):
        text = _clipboard_text().replace("\r", "").replace("\n", " ").strip()
        if text:
            event.current_buffer.insert_text(text)

    # VS Code's Ctrl+V / Ctrl+Shift+V / right-click send the clipboard as a *bracketed paste*.
    # Insert it into the focused input with newlines collapsed to spaces, so a copied value with
    # a trailing newline is inserted instead of submitting (which looked like "it clears everything").
    from prompt_toolkit.keys import Keys

    @kb.add(Keys.BracketedPaste)
    def _(event):
        text = event.data.replace("\r", " ").replace("\n", " ").strip()
        if text:
            event.current_buffer.insert_text(text)

    def _page(event):
        return max(1, event.app.output.get_size().rows - 8)

    @kb.add("pageup")
    @kb.add("c-up")
    def _(event):
        follow["on"] = False
        log_area.buffer.cursor_up(count=_page(event))

    @kb.add("pagedown")
    @kb.add("c-down")
    def _(event):
        buf = log_area.buffer
        buf.cursor_down(count=_page(event))
        # back at the last line → resume auto-scroll to newest output
        if buf.document.cursor_position_row >= buf.document.line_count - 1:
            follow["on"] = True
            buf.cursor_position = len(buf.text)

    @kb.add("home")
    def _(event):
        follow["on"] = False
        log_area.buffer.cursor_position = 0

    @kb.add("end")
    def _(event):
        follow["on"] = True
        log_area.buffer.cursor_position = len(log_area.buffer.text)

    @kb.add("y", filter=Condition(lambda: confirming["on"]))
    def _(event):
        confirming["on"] = False; cresult["v"] = True; cevent.set()

    @kb.add("n", filter=Condition(lambda: confirming["on"]))
    def _(event):
        confirming["on"] = False; cresult["v"] = False; cevent.set()

    # Mouse wheel over the log pane scrolls the log (not the terminal scrollback). We move the
    # buffer cursor — the renderer keeps the cursor on screen, so scrolling sticks — and detach
    # follow so new output doesn't yank the view back to the bottom.
    _orig_log_mouse = log_area.control.mouse_handler

    def _log_mouse(mouse_event):
        et = mouse_event.event_type
        if et == MouseEventType.SCROLL_UP:
            follow["on"] = False
            log_area.buffer.cursor_up(count=3)
            return None
        if et == MouseEventType.SCROLL_DOWN:
            buf = log_area.buffer
            buf.cursor_down(count=3)
            if buf.document.cursor_position_row >= buf.document.line_count - 1:
                follow["on"] = True
                buf.cursor_position = len(buf.text)
            return None
        return _orig_log_mouse(mouse_event)

    log_area.control.mouse_handler = _log_mouse

    body = HSplit([
        Window(FormattedTextControl(header_text), height=3, style="bg:default"),
        Frame(Window(FormattedTextControl(steps_text), height=len(publish.STEPS)), title="steps"),
        Frame(log_area, title="log (history + output) — PgUp/PgDn or Ctrl+↑/↓ scroll · Home/End jump"),
        Frame(input_area),
    ])
    # mouse_support ON so the wheel scrolls the log pane (above). Trade-off: the app captures the
    # mouse, so native click-select is off — hold SHIFT while dragging to select/copy via the
    # terminal, and use Ctrl+Y to paste.
    app = Application(layout=Layout(body, focused_element=input_area), key_bindings=kb,
                      full_screen=True, mouse_support=True)
    ref["app"] = app
    ref["loop"] = asyncio.get_event_loop_policy().get_event_loop()
    try:
        app.run()
    finally:
        confirming["on"] = False
        cevent.set()  # release any waiting worker
    print("albedo off — bye.")
