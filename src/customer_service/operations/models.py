"""Typed records shared by the admin API, worker, and case projection."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


CaseStatus = Literal["open", "claimed", "resolved"]
MessageDirection = Literal["customer", "bot", "staff"]
DeliveryStatus = Literal["pending", "queued", "sent", "delivered", "read", "failed"]


class StaffIdentity(BaseModel):
    sub: str
    email: str


class CaseRecord(BaseModel):
    case_id: str
    thread_id: str
    status: CaseStatus
    assigned_to_sub: str | None = None
    assigned_to_email: str | None = None
    handoff_summary: dict[str, object]
    version: int
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None


class CaseMessage(BaseModel):
    message_id: str
    case_id: str
    direction: MessageDirection
    content: str
    author_sub: str | None = None
    delivery_status: DeliveryStatus = "pending"
    provider_message_id: str | None = None
    created_at: datetime


class OutboxRecord(BaseModel):
    outbox_id: str
    thread_id: str
    case_id: str | None = None
    message_id: str | None = None
    kind: Literal["text", "template"]
    payload: dict[str, object]
    status: str
    attempts: int
    next_attempt_at: datetime


class CaseEvent(BaseModel):
    event: Literal["case_created", "case_updated", "message"]
    case_id: str
    version: int
    message_id: str | None = None


class TemplateRequest(BaseModel):
    template_id: str
    parameters: list[str] = Field(default_factory=list)
