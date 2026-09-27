"""FAQ retrieval contracts shared by tools, runtime wiring, and tests."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class FaqEntry(BaseModel):
    id: str
    title: str
    answer: str

    model_config = ConfigDict(extra="forbid")


class FaqRepository(Protocol):
    """The public FAQ tool depends on this boundary, not a retrieval engine."""

    async def search(self, query: str) -> FaqEntry | None: ...


class UnavailableFaqRepository:
    """Default for isolated graphs that never invoke FAQ lookup."""

    async def search(self, query: str) -> FaqEntry | None:
        raise RuntimeError("FAQ retrieval has not been configured for this graph")
