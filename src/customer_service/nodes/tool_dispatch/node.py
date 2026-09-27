"""Wrap LangGraph's ToolNode so infrastructure failures become handoffs."""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.prebuilt import ToolNode

from customer_service.state.models import AgentState, EscalationReason


def build_node(tool_node: ToolNode) -> Callable[[AgentState], Awaitable[dict[str, object]]]:
    async def tool_dispatch(state: AgentState) -> dict[str, object]:
        try:
            return await tool_node.ainvoke(state)
        except Exception:
            return {"escalation_reason": EscalationReason.TOOL_FAILURE}

    return tool_dispatch
