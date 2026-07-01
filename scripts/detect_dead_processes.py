#!/usr/bin/env python3
"""Detect zombie/defunct processes on this machine (read-only).

One-shot tool. Lists any zombie (``Z`` state) / ``<defunct>`` processes running
on the local machine, including each zombie's parent process (a zombie can only
be reaped by signalling its parent).

This script is DETECTION ONLY: it never signals, kills, or otherwise mutates
anything. The interactive confirm-and-kill step is intentionally deferred (see
the note near the bottom of this file).

Usage:
    python3 detect_dead_processes.py
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass

# `=` headers suppress the ps header row, giving one clean record per line.
PS_FIELDS = ["pid", "ppid", "user", "stat", "etime", "comm", "args"]
PS_CMD = ["ps", "-eo", ",".join(f"{f}=" for f in PS_FIELDS)]


@dataclass
class Proc:
    pid: str
    ppid: str
    user: str
    stat: str
    etime: str
    comm: str
    args: str

    @property
    def is_zombie(self) -> bool:
        return "Z" in self.stat or "<defunct>" in self.comm


def parse_ps(output: str) -> dict[str, Proc]:
    """Parse `ps -eo ...` output into {pid: Proc}."""
    procs: dict[str, Proc] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split off the first 6 fixed fields; the remainder is `args` (may be
        # empty for zombies and may itself contain spaces).
        parts = line.split(None, 6)
        if len(parts) < 6:
            continue
        pid, ppid, user, stat, etime, comm = parts[:6]
        args = parts[6] if len(parts) == 7 else ""
        procs[pid] = Proc(pid, ppid, user, stat, etime, comm, args)
    return procs


def run_ps() -> str:
    """Run the local ps and return its stdout."""
    return subprocess.run(PS_CMD, capture_output=True, text=True, check=True).stdout


def main() -> None:
    host = socket.gethostname()
    print(f"Scanning {host} for zombie/defunct processes...\n")

    procs = parse_ps(run_ps())
    zombies = [p for p in procs.values() if p.is_zombie]

    if not zombies:
        print("OK — 0 zombie/defunct processes.")
    else:
        print(f"FOUND {len(zombies)} zombie/defunct:")
        for z in zombies:
            comm = z.comm or "<defunct>"
            print(f"    PID {z.pid}  STAT {z.stat}  etime {z.etime}  comm={comm}")
            parent = procs.get(z.ppid)
            if parent is not None:
                pargs = parent.args or parent.comm
                print(f"        parent PPID {parent.pid}  comm={parent.comm}  args={pargs}")
            else:
                print(f"        parent PPID {z.ppid}  (not in process table)")

    print()
    print(f"Summary: {len(zombies)} zombie/defunct on {host}.")
    print("Detection only — no processes were signalled or killed.")

    # NOTE: future enhancement — an interactive y/n confirm-and-kill step would
    # attach here. Reaping a zombie means signalling its *parent* (PPID), so the
    # existing patterns in src/sanity_remote/worker.py (_kill_port_squatter /
    # _kill_vllm) and chat_to_king/engine.py are the reference. Intentionally
    # not implemented in this read-only version.


if __name__ == "__main__":
    main()
