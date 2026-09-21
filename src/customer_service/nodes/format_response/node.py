"""Merge an optional catalog-owned promotion into the support response."""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage

from customer_service.state.models import AgentState


def build_node() -> Callable[[AgentState], dict[str, object]]:
    def format_response(state: AgentState) -> dict[str, object]:
        # A skipped or declined promotion leaves the assistant's original message intact.
        if not state.promo_line or not state.draft_response or not state.draft_message_id:
            return {
                "draft_response": None,
                "draft_message_id": None,
                "promo_line": None,
            }

        final_response = f"{state.draft_response}\n\n{state.promo_line}"
        # add_messages replaces a message with the same ID, avoiding a duplicate bot reply.
        return {
            "messages": [AIMessage(id=state.draft_message_id, content=final_response)],
            "already_promoted": True,
            "draft_response": None,
            "draft_message_id": None,
            "promo_line": None,
        }

    return format_response
