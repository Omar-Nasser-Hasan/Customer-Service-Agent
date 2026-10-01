"""Layer 3: Gemini, the careful classifier that decides whenever Jev cannot.

Gemini reads the fixed prompt from prompt.py plus the customer's redacted
message and must answer with exactly one action. Anything else (an outage, a
blocked response, a malformed answer) raises, and the node escalates to a human.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from customer_service.config.settings import Settings
from customer_service.nodes.safety_check.prompt import build_system_prompt
from customer_service.nodes.safety_check.verdict import Verdict
from customer_service.state.models import EscalationReason


class SafetyDecision(BaseModel):
    """The only shape Gemini is allowed to answer with."""

    action: Literal["allow", "refuse", "escalate"]

    model_config = ConfigDict(extra="forbid")


class SafetyJudge(Protocol):
    def invoke(self, input: object) -> SafetyDecision | dict[str, Any]: ...


def create_structured_judge(model: BaseChatModel) -> SafetyJudge:
    """Use Gemini's native JSON-schema mode, not its tool-call emulation."""

    return model.with_structured_output(SafetyDecision, method="json_schema")


class GeminiJudge:
    def __init__(self, judge: SafetyJudge, settings: Settings) -> None:
        self._judge = judge
        self._prompt = SystemMessage(content=build_system_prompt(settings))

    def check(self, text: str) -> Verdict:
        """Classify already-redacted text. Raises on any provider or format problem."""

        decision = SafetyDecision.model_validate(
            self._judge.invoke([self._prompt, HumanMessage(content=text)])
        )
        if decision.action == "escalate":
            return Verdict(action="escalate", source="gemini", reason=EscalationReason.SAFETY_UNCERTAIN)
        return Verdict(action=decision.action, source="gemini")
