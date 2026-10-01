"""Layer 1: instant keyword rules for the obvious cases.

These run first and cost nothing. They are English-only on purpose; Arabic and
Arabizi messages are handled by the Jev and Gemini layers. Rules are checked in
order and the first match wins, so refusals take priority over escalations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from customer_service.nodes.safety_check.verdict import Verdict
from customer_service.state.models import EscalationReason


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    action: Literal["refuse", "escalate"]
    reason: EscalationReason | None = None


RULES: tuple[Rule, ...] = (
    Rule(
        name="injection",
        pattern=re.compile(
            r"\b(ignore (?:all |any )?(?:previous|prior)|system prompt|developer message|"
            r"reveal .*instructions|jailbreak|act as (?:a )?(?:system|developer))\b",
            re.IGNORECASE,
        ),
        action="refuse",
    ),
    Rule(
        name="abuse",
        pattern=re.compile(r"\b(fuck|shit|asshole|bitch|kill yourself)\b", re.IGNORECASE),
        action="refuse",
    ),
    Rule(
        name="human_request",
        pattern=re.compile(
            r"\b(?:human|representative|support agent|manager|real person)\b|"
            r"(?:need|want|talk|speak|connect).{0,24}\bperson\b",
            re.IGNORECASE,
        ),
        action="escalate",
        reason=EscalationReason.HUMAN_REQUEST,
    ),
    Rule(
        name="return_exception",
        pattern=re.compile(
            r"\b(damaged|defective|wrong item|missing item|broken item)\b", re.IGNORECASE
        ),
        action="escalate",
        reason=EscalationReason.RETURN_EXCEPTION,
    ),
    Rule(
        name="billing_dispute",
        pattern=re.compile(
            r"\b(unauthorized|fraud|duplicate|charged twice|wrong charge|incorrect charge|wrong amount)\b",
            re.IGNORECASE,
        ),
        action="escalate",
        reason=EscalationReason.BILLING_DISPUTE,
    ),
)


def check(text: str) -> Verdict | None:
    """Return a verdict if a rule matches, or None to let the next layer decide."""

    for rule in RULES:
        if rule.pattern.search(text):
            return Verdict(action=rule.action, source="rules", reason=rule.reason, detail=rule.name)
    return None
