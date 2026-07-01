"""OpenAI-compatible gateway in front of vLLM.

OpenWebUI points its OPENAI_API_BASE_URL here (not at vLLM directly). Normally it transparently
stream-proxies to vLLM; while a coronation is mid-swap (engine.reloading) or vLLM is otherwise down,
it returns a spec-valid chat completion whose content is an in-chat reload notice instead of an error —
so users see "a new king is being loaded" rather than a broken connection.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask

from config import KingChatSettings
from engine import KingVllmEngine

_RELOAD_ID = "chatcmpl-king-reload"




def _message_text(messages: list) -> str:
    parts: list[str] = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for seg in c:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    parts.append(seg["text"])
    return "\n".join(parts).lower()


def _mentions_albedo(messages: list, keywords: list[str]) -> bool:
    text = _message_text(messages)
    return any(kw in text for kw in keywords)


def _build_knowledge(doc: str) -> str:
    return (
        "You are `albedo-king`, the current top-ranked model on Albedo (Bittensor subnet SN97). "
        "The user is asking about Albedo. Use the following reference document as authoritative "
        "knowledge about Albedo — its subnet mechanics, mining, validation, scoring, and architecture. "
        "Prefer it over your own assumptions; if it doesn't cover something, say so.\n\n"
        "<BEGIN ALBEDO REFERENCE (llms.txt)>\n" + doc + "\n<END ALBEDO REFERENCE>"
    )


def _inject_knowledge(messages: list, block: str) -> list:
    msgs = list(messages)
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
        msgs[0] = {**msgs[0], "content": block + "\n\n" + msgs[0]["content"]}
    else:
        msgs = [{"role": "system", "content": block}, *msgs]
    return msgs


def _notice_text(engine: KingVllmEngine, settings: KingChatSettings) -> str:
    if settings.reload_notice:
        return settings.reload_notice
    king = engine.incoming_king
    if king is None:
        return "The model is starting up — please resend your message in a moment."
    parts = ["👑 A new king has been crowned"]
    if king.uid is not None:
        parts.append(f" — uid {king.uid}")
    hk = (king.hotkey or "")[:10]
    if hk:
        parts.append(f" ({hk}…)")
    if king.king_version is not None:
        parts.append(f", king v{king.king_version}")
    parts.append(". The model is reloading on the GPUs (~1–2 min). Please resend your message shortly.")
    return "".join(parts)


def _notice_response(engine: KingVllmEngine, settings: KingChatSettings, stream: bool):
    text = _notice_text(engine, settings)
    model = settings.served_model_name
    created = int(time.time())

    if not stream:
        body = {
            "id": _RELOAD_ID,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(body)

    def _chunk(delta: dict, finish=None) -> str:
        payload = {
            "id": _RELOAD_ID,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def _gen() -> AsyncIterator[str]:
        yield _chunk({"role": "assistant", "content": text})
        yield _chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


def create_app(engine: KingVllmEngine, settings: KingChatSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0)
        )
        app.state.albedo_keywords = [k.strip().lower() for k in settings.llms_keywords.split(",") if k.strip()]
        doc: str | None = None
        try:
            p = Path(settings.llms_path)
            if p.is_file():
                doc = p.read_text(encoding="utf-8")
            elif settings.llms_url:
                r = await app.state.client.get(settings.llms_url)
                if r.status_code == 200:
                    doc = r.text
                else:
                    logger.warning("[king-chat] llms_url HTTP {}", r.status_code)
        except Exception as exc:
            logger.warning("[king-chat] llms.txt load failed: {}", exc)
        if doc and settings.llms_max_chars and len(doc) > settings.llms_max_chars:
            doc = doc[: settings.llms_max_chars]
        app.state.albedo_knowledge = _build_knowledge(doc) if doc else None
        if app.state.albedo_knowledge:
            logger.info(
                "[king-chat] loaded llms.txt ({:.1f} KB) — Albedo knowledge gated on {}",
                len(doc) / 1024, app.state.albedo_keywords,
            )
        else:
            logger.warning("[king-chat] no llms.txt loaded; Albedo knowledge injection disabled")
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="Albedo King Chat Gateway", version="0.1.0", lifespan=lifespan)

    async def _proxy_or_notice(request: Request, path: str):
        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        stream = bool(payload.get("stream"))

        if engine.reloading or not engine.serving:
            return _notice_response(engine, settings, stream)

        body = raw
        if path == "/v1/chat/completions" and app.state.albedo_knowledge:
            msgs = payload.get("messages")
            if isinstance(msgs, list) and _mentions_albedo(msgs, app.state.albedo_keywords):
                payload["messages"] = _inject_knowledge(msgs, app.state.albedo_knowledge)
                body = json.dumps(payload).encode("utf-8")

        client: httpx.AsyncClient = app.state.client
        url = f"http://127.0.0.1:{settings.vllm_port}{path}"
        req = client.build_request("POST", url, content=body, headers={"content-type": "application/json"})
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError:
            return _notice_response(engine, settings, stream)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={"content-type": upstream.headers.get("content-type", "application/json")},
            background=BackgroundTask(upstream.aclose),
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "reloading": engine.reloading, "serving": engine.serving}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "albedo",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_or_notice(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_or_notice(request, "/v1/completions")

    return app
