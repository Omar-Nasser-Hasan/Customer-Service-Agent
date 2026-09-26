from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from customer_service.config.settings import Settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.orders import default_order_repository
from customer_service.services.handoffs import HandoffNotOpenError, HandoffService
from tests.conftest import ScriptedChatModel, allowing_safety_judge, declining_promo_judge


@pytest.mark.asyncio
async def test_resolve_rejects_a_thread_without_an_open_interrupt() -> None:
    graph = build_graph(
        settings=Settings(),
        repository=default_order_repository(),
        model=ScriptedChatModel(responses=[AIMessage(content="Hello")]),
        promo_judge=declining_promo_judge(),
        safety_judge=allowing_safety_judge(),
    )
    graph.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "not-open"}},
    )
    with pytest.raises(HandoffNotOpenError):
        await HandoffService(graph).resolve("not-open")
