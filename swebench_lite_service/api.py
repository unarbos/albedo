from __future__ import annotations

import asyncio

from fastapi import BackgroundTasks, FastAPI

from .config import SETTINGS
from .kings import fetch_dashboard, kings_from_dashboard
from .state import load_state, select_next_king
from .worker import run_one

app = FastAPI(title="Albedo SWE-bench Lite Service")

_run_lock = asyncio.Lock()


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "state_dir": str(SETTINGS.state_dir),
        "dataset": SETTINGS.dataset_name,
        "split": SETTINGS.split,
        "limit_instances": SETTINGS.limit_instances,
    }


@app.get("/state")
async def state() -> dict:
    return load_state()


@app.get("/kings")
async def kings() -> dict:
    dashboard = await fetch_dashboard()
    rows = [king.to_dict() for king in kings_from_dashboard(dashboard)]
    return {"count": len(rows), "kings": rows}


@app.get("/next")
async def next_king() -> dict:
    dashboard = await fetch_dashboard()
    king = select_next_king(kings_from_dashboard(dashboard))
    return {"king": king.to_dict() if king else None}


@app.post("/run-one")
async def run_one_endpoint(background_tasks: BackgroundTasks) -> dict:
    if _run_lock.locked():
        return {"ok": False, "error": "benchmark already running"}
    background_tasks.add_task(_locked_run_one)
    return {"ok": True, "started": True}


async def _locked_run_one() -> None:
    async with _run_lock:
        await run_one()

