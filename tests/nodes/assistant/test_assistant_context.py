from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from customer_service.config.settings import Settings
from customer_service.nodes.assistant.node import build_node
from customer_service.state.models import AgentState
from customer_service.verification.grant import VerificationPolicy
from tests.conftest import ScriptedChatModel

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class RecordingModel(ScriptedChatModel):
    seen: list[list[BaseMessage]] = []

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def context_for(state: AgentState) -> str:
    model = RecordingModel(responses=[AIMessage(content="ok")], seen=[])
    node = build_node(
        settings=Settings(), model=model, tools=[], policy=VerificationPolicy(clock=lambda: NOW)
    )
    node(state)
    return str(model.seen[0][1].content)


def test_context_says_verified_without_ever_naming_the_email() -> None:
    state = AgentState(
        messages=[HumanMessage(content="hi")],
        verified_email="alice@example.com",
        verified_at=NOW - timedelta(minutes=10),
        customer={"email": "alice@example.com", "order_id": "ORD-1001"},
    )

    context = context_for(state)

    assert "This conversation is verified." in context
    assert "alice@example.com" not in context and "ORD-1001" not in context


def test_expired_and_missing_verification_both_read_as_not_verified() -> None:
    expired = AgentState(
        messages=[HumanMessage(content="hi")],
        verified_email="alice@example.com",
        verified_at=NOW - timedelta(hours=3),
    )

    assert "This conversation is not verified." in context_for(expired)
    assert "This conversation is not verified." in context_for(AgentState(messages=[HumanMessage(content="hi")]))
