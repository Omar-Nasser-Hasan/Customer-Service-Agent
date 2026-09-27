from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Local WebSocket fan-out; durable fan-out is handled by the DB relay."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = tuple(self._connections)
        for socket in targets:
            try:
                await socket.send_json(payload)
            except Exception:
                await self.disconnect(socket)
