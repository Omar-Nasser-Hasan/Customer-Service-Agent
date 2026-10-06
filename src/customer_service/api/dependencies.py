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
from customer_service.retrieval.runtime import faq_repository_runtime
from customer_service.operations.repository import OperationsRepository, operations_repository_runtime
from customer_service.services.cases import CaseService
from customer_service.auth.service import StaffAuthService
from customer_service.realtime.manager import ConnectionManager
from customer_service.realtime.relay import PostgresCaseEventRelay
from customer_service.verification.attempts import PostgresAttemptLimiter, postgres_attempt_limiter

LOGGER = logging.getLogger(__name__)


class ApplicationRuntime:
    """Own the production graph, pooled saver, and future-facing service contracts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.graph: Any | None = None
        self.checkpointer: Any | None = None
        self.handoffs: HandoffService | None = None
        self.faq_repository: Any | None = None
        self.attempt_limiter: PostgresAttemptLimiter | None = None
        self.operations: OperationsRepository | None = None
        self.cases: CaseService | None = None
        self.auth: StaffAuthService | None = None
        self.realtime = ConnectionManager()
        self.relay: PostgresCaseEventRelay | None = None
        self.observability = ObservabilityRuntime(self.settings)
        self._stack = AsyncExitStack()
        self._retention_task: asyncio.Task[None] | None = None
        self._retention_lock = asyncio.Lock()

    def _verification_hash_key(self) -> str:
        """Key for hashing emails and order IDs in the shared failure table."""

        secret = self.settings.verification_hash_key
        if secret is not None and secret.get_secret_value():
            return secret.get_secret_value()
        if self.settings.app_environment == "production":
            raise ValueError("Missing required configuration: VERIFICATION_HASH_KEY")
        return "local-development-verification-key"

    async def start(self) -> None:
        dsn = self.settings.require_secret("POSTGRES_DSN", self.settings.postgres_dsn)
        self.checkpointer = await self._stack.enter_async_context(postgres_checkpointer(self.settings))
        self.faq_repository = await self._stack.enter_async_context(faq_repository_runtime(self.settings))
        # Failed identity checks are counted across threads and channels.
        self.attempt_limiter = await self._stack.enter_async_context(
            postgres_attempt_limiter(
                dsn,
                hash_key=self._verification_hash_key(),
                limit=self.settings.verification_key_failure_limit,
            )
        )
        self.graph = build_graph(
            settings=self.settings,
            repository=default_order_repository(),
            checkpointer=self.checkpointer,
            faq_repository=self.faq_repository,
            attempt_limiter=self.attempt_limiter,
        )
        self.handoffs = HandoffService(self.graph)
        self.operations = await self._stack.enter_async_context(operations_repository_runtime(dsn))
        self.cases = CaseService(self.graph, self.handoffs, self.operations, self.settings)
        self.auth = StaffAuthService(self.settings, self.operations)
        self.relay = PostgresCaseEventRelay(dsn, self.realtime)
        await self.relay.start()
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
        if self.relay is not None:
            await self.relay.stop()
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
                    if self.operations is not None:
                        await self.operations.prune_resolved(self.settings.checkpoint_retention_days)
                    if self.attempt_limiter is not None:
                        await asyncio.to_thread(self.attempt_limiter.prune)
                    LOGGER.info("retention_completed", extra={"deleted_threads": len(deleted)})
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("retention_failed", extra={"error_type": type(error).__name__})
            await asyncio.sleep(86_400)
