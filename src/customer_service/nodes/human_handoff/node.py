"""Durably pause an escalated conversation until an internal resolver closes it."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable

from langgraph.types import interrupt

from customer_service.state.models import AgentState


def build_node() -> Callable[[AgentState], dict[str, object]]:
    def human_handoff(state: AgentState) -> dict[str, object]:
        if not state.escalated or state.handoff_summary is None:
            raise ValueError("Human handoff requires an open escalation summary")
        resolution = interrupt(
            {
                "kind": "human_handoff",
                "case_status": state.case_status,
                "summary": state.handoff_summary.model_dump(mode="json"),
            }
        )
        if not isinstance(resolution, dict) or resolution.get("action") != "resolve":
            raise ValueError("Only an internal resolve action can resume a human handoff")
        return {
            "escalated": False,
            "escalation_reason": None,
            "case_status": "resolved",
            "resolved_at": datetime.now(UTC),
        }

    return human_handoff
