#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os

from loguru import logger

from config import get_king_chat_settings, KingChatSettings


_SETTINGS = get_king_chat_settings()
os.environ["ALBEDO_MODEL_CACHE_DIR"] = _SETTINGS.models_dir
os.environ["CV_MODEL_CACHE_DIR"] = _SETTINGS.models_dir

from engine import KingVllmEngine
from gateway import create_app
from king_source import current_king


async def _serve_gateway(app, settings: KingChatSettings) -> None:
    import uvicorn

    config = uvicorn.Config(
        app, host=settings.gateway_host, port=settings.gateway_port, log_level="info"
    )
    await uvicorn.Server(config).serve()


async def _poll_loop(engine: KingVllmEngine, settings: KingChatSettings) -> None:
    while True:
        try:
            king = await asyncio.to_thread(current_king, settings)
            if king is not None:
                await engine.ensure_king(king)
            if engine.serving and not engine.reloading and not await engine.healthy():
                logger.warning("[king-chat] vLLM unhealthy — restarting from cache")
                await engine.restart_loaded()
        except Exception:
            logger.exception("[king-chat] poll tick failed")
        await asyncio.sleep(settings.poll_interval_s)


async def _run(engine: KingVllmEngine, app, settings: KingChatSettings) -> None:
    await asyncio.gather(
        _serve_gateway(app, settings),
        _poll_loop(engine, settings),
    )


def main() -> None:
    settings = _SETTINGS
    logger.info(
        "[king-chat] starting: gateway :{} -> vLLM :{} (served as {!r}), king dir {}, poll {}s",
        settings.gateway_port, settings.vllm_port, settings.served_model_name,
        settings.models_dir, settings.poll_interval_s,
    )
    engine = KingVllmEngine(settings)
    app = create_app(engine, settings)
    asyncio.run(_run(engine, app, settings))


if __name__ == "__main__":
    main()
