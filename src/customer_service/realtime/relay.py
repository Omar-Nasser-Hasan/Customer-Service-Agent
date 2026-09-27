from __future__ import annotations

import asyncio
import json
import logging

from psycopg import AsyncConnection

from customer_service.realtime.manager import ConnectionManager

LOGGER = logging.getLogger(__name__)


class PostgresCaseEventRelay:
    """One LISTEN connection per API instance; event bodies contain no PII."""

    def __init__(self, dsn: str, manager: ConnectionManager) -> None:
        self.dsn = dsn
        self.manager = manager
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._listen(), name="case-event-relay")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self) -> None:
        while True:
            try:
                async with await AsyncConnection.connect(self.dsn, autocommit=True) as connection:
                    await connection.execute("LISTEN customer_service_case_events")
                    async for notification in connection.notifies():
                        try:
                            payload = json.loads(notification.payload)
                            if set(payload) - {"event", "case_id", "message_id", "version"}:
                                continue
                            await self.manager.publish(payload)
                        except (ValueError, TypeError):
                            LOGGER.warning("case_event_malformed")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("case_event_relay_reconnecting", extra={"error_type": type(error).__name__})
                await asyncio.sleep(1)
