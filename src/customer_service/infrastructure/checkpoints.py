"""Checkpoint adapter boundary; Postgres implementation arrives with durable handoffs."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver


def local_checkpointer() -> InMemorySaver:
    """Use process-local memory until the Postgres checkpointer is introduced."""

    return InMemorySaver()
