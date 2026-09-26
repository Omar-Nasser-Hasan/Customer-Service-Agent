"""Internal contract for resolving a durable human-handoff interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command


class HandoffNotOpenError(ValueError):
    """Raised when a future admin surface attempts to resolve a non-open case."""


class HandoffService:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def is_open(self, thread_id: str) -> bool:
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        return bool(
            snapshot.values.get("escalated")
            and snapshot.values.get("case_status") in {"open", "claimed"}
            and "human_handoff" in snapshot.next
        )

    async def resolve(self, thread_id: str) -> dict[str, object]:
        config = {"configurable": {"thread_id": thread_id}}
        if not await self.is_open(thread_id):
            raise HandoffNotOpenError(f"No open handoff exists for thread '{thread_id}'")
        return await self._graph.ainvoke(Command(resume={"action": "resolve"}), config=config)
