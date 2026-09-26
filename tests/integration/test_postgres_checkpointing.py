from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from docker.errors import DockerException
from langchain_core.messages import AIMessage, HumanMessage
from testcontainers.community.postgres import PostgresContainer

from customer_service.config.settings import Settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.checkpoints import postgres_checkpointer, prune_inactive_threads
from customer_service.infrastructure.orders import default_order_repository
from customer_service.services.handoffs import HandoffService
from tests.conftest import ScriptedChatModel, allowing_safety_judge, declining_promo_judge


@pytest.fixture
def postgres_dsn() -> str:
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as postgres:
            yield postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    except DockerException as error:
        pytest.skip(f"Docker daemon is unavailable: {error}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_checkpoints_survive_a_fresh_graph_and_resume(postgres_dsn: str) -> None:
    settings = Settings(postgres_dsn=postgres_dsn)
    async with postgres_checkpointer(settings) as saver:
        graph = build_graph(
            settings=settings,
            repository=default_order_repository(),
            model=ScriptedChatModel(responses=[AIMessage(content="Normal support answer.")]),
            promo_judge=declining_promo_judge(),
            safety_judge=allowing_safety_judge(),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": "postgres-interrupt"}}
        opened = await graph.ainvoke(
            {"messages": [HumanMessage(content="I need a human representative.")]}, config=config
        )
        assert opened["__interrupt__"]

        fresh_graph = build_graph(
            settings=settings,
            repository=default_order_repository(),
            model=ScriptedChatModel(responses=[AIMessage(content="Normal support answer.")]),
            promo_judge=declining_promo_judge(),
            safety_judge=allowing_safety_judge(),
            checkpointer=saver,
        )
        resolved = await HandoffService(fresh_graph).resolve("postgres-interrupt")
        assert resolved["case_status"] == "resolved"
        assert resolved["escalated"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retention_keeps_open_cases_and_prunes_old_resolved_threads(postgres_dsn: str) -> None:
    settings = Settings(postgres_dsn=postgres_dsn)
    async with postgres_checkpointer(settings) as saver:
        graph = build_graph(
            settings=settings,
            repository=default_order_repository(),
            model=ScriptedChatModel(responses=[AIMessage(content="Normal support answer.")]),
            promo_judge=declining_promo_judge(),
            safety_judge=allowing_safety_judge(),
            checkpointer=saver,
        )
        old = datetime.now(UTC) - timedelta(days=91)
        resolved_config = {"configurable": {"thread_id": "old-resolved"}}
        await graph.ainvoke({"messages": [HumanMessage(content="Hello")]}, config=resolved_config)
        await graph.aupdate_state(resolved_config, {"case_status": "resolved", "last_activity_at": old})

        open_config = {"configurable": {"thread_id": "old-open"}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content="I need a human representative.")]}, config=open_config
        )
        await graph.aupdate_state(open_config, {"last_activity_at": old})

        deleted = await prune_inactive_threads(
            graph=graph, checkpointer=saver, retention_days=90, now=datetime.now(UTC)
        )
        assert deleted == ["old-resolved"]
        assert (await graph.aget_state(open_config)).values["case_status"] == "open"
