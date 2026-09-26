"""Application-lifetime graph and persistence wiring."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from customer_service.config.settings import Settings, get_settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.checkpoints import postgres_checkpointer, prune_inactive_threads
from customer_service.infrastructure.orders import default_order_repository
from customer_service.services.handoffs import HandoffService


class ApplicationRuntime:
    """Own the production graph, pooled saver, and future-facing service contracts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.graph: Any | None = None
        self.checkpointer: Any | None = None
        self.handoffs: HandoffService | None = None
        self._stack = AsyncExitStack()

    async def start(self) -> None:
        self.checkpointer = await self._stack.enter_async_context(postgres_checkpointer(self.settings))
        self.graph = build_graph(
            settings=self.settings,
            repository=default_order_repository(),
            checkpointer=self.checkpointer,
        )
        self.handoffs = HandoffService(self.graph)

    async def stop(self) -> None:
        await self._stack.aclose()

    async def prune_expired_threads(self) -> list[str]:
        if self.graph is None or self.checkpointer is None:
            raise RuntimeError("Application runtime has not started")
        return await prune_inactive_threads(
            graph=self.graph,
            checkpointer=self.checkpointer,
            retention_days=self.settings.checkpoint_retention_days,
        )
