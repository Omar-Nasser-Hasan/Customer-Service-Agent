"""Application factory for the current bot-only HTTP surface."""

from typing import Callable

from fastapi import FastAPI

from customer_service.api.bot.conversations import create_router as create_conversations_router
from customer_service.api.health import router as health_router
from customer_service.api.dependencies import get_graph


def create_app(graph_provider: Callable[[], object] = get_graph) -> FastAPI:
    app = FastAPI(title="Customer Service Agent", version="0.3.0")
    app.include_router(health_router)
    app.include_router(create_conversations_router(graph_provider))
    return app


app = create_app()
