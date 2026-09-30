"""Application factory for the Iteration 4 bot surface."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from customer_service.api.bot.conversations import create_router as create_conversations_router
from customer_service.api.dependencies import ApplicationRuntime
from customer_service.config.settings import get_settings
from customer_service.api.health import router as health_router
from customer_service.api.admin.routes import create_router as create_admin_router
from customer_service.api.bot.whatsapp import create_router as create_whatsapp_router
from customer_service.observability.logs import configure_logging

def create_app(graph_provider: Callable[[], Any] | None = None) -> FastAPI:
    """Create a production app or a dependency-injected test app."""
    configure_logging()
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

    settings = get_settings()
    app = FastAPI(
        title="Customer Service Agent",
        version="0.8.0",
        lifespan=None if graph_provider is not None else lifespan,
    )
    app.include_router(health_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-csrf-token"],
    )
    app.include_router(
        create_conversations_router(
            graph_provider or (lambda: runtime.graph),
            observability_provider=None if graph_provider is not None else (lambda: runtime.observability),
            case_service_provider=None if graph_provider is not None else (lambda: runtime.cases),
        )
    )
    if graph_provider is None:
        app.include_router(create_admin_router(lambda: runtime))
        app.include_router(create_whatsapp_router(lambda: runtime))
    return app


app = create_app()
