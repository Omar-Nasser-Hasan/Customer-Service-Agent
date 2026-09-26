"""Application factory for the Iteration 4 bot surface."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from customer_service.api.bot.conversations import create_router as create_conversations_router
from customer_service.api.dependencies import ApplicationRuntime
from customer_service.api.health import router as health_router


def create_app(graph_provider: Callable[[], Any] | None = None) -> FastAPI:
    """Create a production app or a dependency-injected test app."""

    runtime: ApplicationRuntime | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime
        runtime = ApplicationRuntime()
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="Customer Service Agent",
        version="0.4.0",
        lifespan=None if graph_provider is not None else lifespan,
    )
    app.include_router(health_router)
    app.include_router(create_conversations_router(graph_provider or (lambda: runtime.graph)))
    return app


app = create_app()
