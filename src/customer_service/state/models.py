"""State contract shared by graph nodes and later admin-case views."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict, Field


class CustomerContext(BaseModel):
    order_id: str | None = None
    email: str | None = None
    phone: str | None = None
    name: str | None = None

    model_config = ConfigDict(extra="forbid")


class PromoMatch(BaseModel):
    promo_id: str
    text: str
    relevance_score: float

    model_config = ConfigDict(extra="forbid")


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    customer: CustomerContext = Field(default_factory=CustomerContext)
    verified: bool = False
    promo_matches: list[PromoMatch] = Field(default_factory=list)
    already_promoted: bool = False
    draft_response: str | None = None
    draft_message_id: str | None = None
    promo_line: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    assigned_to: str | None = None
    case_status: Literal["active", "open", "claimed", "resolved"] = "active"

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
