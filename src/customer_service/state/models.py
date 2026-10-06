"""State contract shared by graph nodes and later admin-case views."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
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


class EscalationReason(StrEnum):
    HUMAN_REQUEST = "human_request"
    RETURN_EXCEPTION = "return_exception"
    BILLING_DISPUTE = "billing_dispute"
    SAFETY_UNCERTAIN = "safety_uncertain"
    TOOL_FAILURE = "tool_failure"
    MODEL_FAILURE = "model_failure"
    VERIFICATION_FAILED = "verification_failed"


class HandoffSummary(BaseModel):
    reason: EscalationReason
    latest_customer_request: str
    attempted_tools: list[str] = Field(default_factory=list)
    verified: bool
    order_id: str | None = None
    masked_email: str | None = None

    model_config = ConfigDict(extra="forbid")


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    # Only the last order the customer looked up, and only for a verified email.
    customer: CustomerContext = Field(default_factory=CustomerContext)
    # Verification is an identity (the email that proved ownership of an order),
    # not an order. Whether it still counts is decided by VerificationPolicy.
    verified_email: str | None = None
    verified_at: datetime | None = None
    # Timestamps of failed checks; only the last 24 hours count. Cleared when a
    # human resolves a handoff, never by a successful verification.
    failed_attempts: list[datetime] = Field(default_factory=list)
    # Customer message already consumed as a claim, so a pair is never counted twice.
    last_claim_message_id: str | None = None
    promo_matches: list[PromoMatch] = Field(default_factory=list)
    already_promoted: bool = False
    draft_response: str | None = None
    draft_message_id: str | None = None
    promo_line: str | None = None
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    safety_action: Literal["allow", "refuse", "escalate"] | None = None
    handoff_summary: HandoffSummary | None = None
    last_activity_at: datetime | None = None
    resolved_at: datetime | None = None
    assigned_to: str | None = None
    case_status: Literal["active", "open", "claimed", "resolved"] = "active"

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
