"""The single result type shared by every layer of the safety check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from customer_service.state.models import EscalationReason

SafetyAction = Literal["allow", "refuse", "escalate"]
Source = Literal["rules", "jev", "gemini", "fail_closed"]


@dataclass(frozen=True)
class Verdict:
    """What the safety check decided, which layer decided it, and why (if escalated)."""

    action: SafetyAction
    source: Source
    reason: EscalationReason | None = None
    detail: str | None = None  # Which rule or question fired. Used for logs only.

    def __post_init__(self) -> None:
        if (self.action == "escalate") != (self.reason is not None):
            raise ValueError("An escalation needs a reason, and only an escalation has one")
