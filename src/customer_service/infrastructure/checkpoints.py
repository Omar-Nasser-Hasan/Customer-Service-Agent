"""Lifecycle-managed checkpoint persistence and retention utilities."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import asyncio
import os
import sys
from typing import Any

# The serializer reads this switch when checkpoint modules are imported. The
# settings model also fixes the runtime value to true.
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

# Psycopg's async implementation requires a selector loop on Windows. This
# module is imported during app construction, before Uvicorn creates its loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from customer_service.config.settings import Settings

# Strict msgpack mode blocks reconstruction of any custom class that isn't
# explicitly listed here. Every application-defined type that can end up in
# checkpointed graph state must be added below by hand; that manual step is
# the whole point of the allowlist; it stops an attacker who gains write
# access to the checkpoint table from smuggling in arbitrary objects that
# execute code on load. Keep this list scoped to exactly what the graph's
# state schema actually holds, add a new tuple, do not widen the scope.
ALLOWED_MSGPACK_MODULES = (
    ("customer_service.state.models", "CustomerContext"),
    ("customer_service.state.models", "PromoMatch"),
    ("customer_service.state.models", "EscalationReason"),
    ("customer_service.state.models", "HandoffSummary"),
)


def local_checkpointer() -> InMemorySaver:
    """Use process-local state only for isolated tests and injected graphs."""

    return InMemorySaver()


@asynccontextmanager
async def postgres_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """Open one application-lifetime pooled Postgres saver and initialize its schema."""

    # Strict msgpack is part of the central settings contract and must be set
    # before the Postgres checkpointer is imported.
    os.environ["LANGGRAPH_STRICT_MSGPACK"] = str(settings.langgraph_strict_msgpack).lower()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    dsn = settings.require_secret("POSTGRES_DSN", settings.postgres_dsn)
    pool = AsyncConnectionPool(
        conninfo=dsn,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=False,
    )
    await pool.open()
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        saver = AsyncPostgresSaver(pool, serde=serde)
        await saver.setup()
        yield saver
    finally:
        await pool.close()


async def prune_inactive_threads(
    *, graph: Any, checkpointer: BaseCheckpointSaver, retention_days: int, now: datetime | None = None
) -> list[str]:
    """Delete complete thread histories only after the agreed retention window."""

    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    seen_threads: set[str] = set()
    deleted: list[str] = []
    # Materialize first: an AsyncPostgresSaver iterator retains a pool
    # connection. Calling graph state methods inside that iterator can exhaust
    # the pool and deadlock retention on a small production pool.
    checkpoints = [checkpoint async for checkpoint in checkpointer.alist(None)]
    for checkpoint in checkpoints:
        configurable = checkpoint.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        state = snapshot.values
        if state.get("escalated") or state.get("case_status") in {"open", "claimed"}:
            continue
        activity = state.get("last_activity_at")
        if isinstance(activity, str):
            activity = datetime.fromisoformat(activity.replace("Z", "+00:00"))
        if isinstance(activity, datetime) and activity.tzinfo is None:
            activity = activity.replace(tzinfo=UTC)
        if isinstance(activity, datetime) and activity < cutoff:
            await checkpointer.adelete_thread(thread_id)
            deleted.append(thread_id)
    return deleted