"""Application-lifetime graph and persistence wiring."""

from __future__ import annotations

from contextlib import AsyncExitStack
import asyncio
import logging
from typing import Any

from customer_service.config.settings import Settings, get_settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.checkpoints import postgres_checkpointer, prune_inactive_threads
from customer_service.infrastructure.orders import default_order_repository
from customer_service.observability.runtime import ObservabilityRuntime
from customer_service.services.handoffs import HandoffService

LOGGER = logging.getLogger(__name__)


class ApplicationRuntime:
    """Own the production graph, pooled saver, and future-facing service contracts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.graph: Any | None = None
        self.checkpointer: Any | None = None
        self.handoffs: HandoffService | None = None
        self.observability = ObservabilityRuntime(self.settings)
        self._stack = AsyncExitStack()
        self._retention_task: asyncio.Task[None] | None = None
        self._retention_lock = asyncio.Lock()

    async def start(self) -> None:
        self.checkpointer = await self._stack.enter_async_context(postgres_checkpointer(self.settings))
        self.graph = build_graph(
            settings=self.settings,
            repository=default_order_repository(),
            checkpointer=self.checkpointer,
        )
        self.handoffs = HandoffService(self.graph)
        self.observability.start()
        self._retention_task = asyncio.create_task(
            self._retention_loop(), name="customer-service-retention"
        )

    async def stop(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
        await self._stack.aclose()

    async def prune_expired_threads(self) -> list[str]:
        if self.graph is None or self.checkpointer is None:
            raise RuntimeError("Application runtime has not started")
        return await prune_inactive_threads(
            graph=self.graph,
            checkpointer=self.checkpointer,
            retention_days=self.settings.checkpoint_retention_days,
        )

    async def _retention_loop(self) -> None:
        while True:
            try:
                async with self._retention_lock:
                    deleted = await self.prune_expired_threads()
                    LOGGER.info("retention_completed", extra={"deleted_threads": len(deleted)})
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("retention_failed", extra={"error_type": type(error).__name__})
            await asyncio.sleep(86_400)
