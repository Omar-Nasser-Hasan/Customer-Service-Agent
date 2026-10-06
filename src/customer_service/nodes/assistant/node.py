"""Customer-facing assistant reasoning node."""

from __future__ import annotations

from typing import Callable
from uuid import uuid4
from datetime import UTC, datetime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from customer_service.config.settings import Settings
from customer_service.nodes.assistant.prompt import build_system_prompt
from customer_service.state.models import AgentState, EscalationReason
from customer_service.verification.grant import VerificationPolicy


def build_node(
    *,
    settings: Settings,
    model: BaseChatModel,
    tools: list[BaseTool],
    policy: VerificationPolicy | None = None,
) -> Callable[[AgentState], dict[str, object]]:
    model_with_tools = model.bind_tools(tools)
    prompt = SystemMessage(content=build_system_prompt(settings))
    grant = policy or VerificationPolicy()

    def assistant(state: AgentState) -> dict[str, object]:
        # An expired verification counts as not verified. The email is never
        # put in front of the model; ownership is enforced in code, not here.
        verified = grant.is_valid(state.verified_email, state.verified_at)
        verification_context = SystemMessage(
            content=(
                f"Internal context. Today's date: {datetime.now(UTC).date().isoformat()}. "
                + ("This conversation is verified." if verified else "This conversation is not verified.")
            )
        )
        try:
            response = model_with_tools.invoke([prompt, verification_context, *state.messages])
        except Exception:
            return {"escalation_reason": EscalationReason.MODEL_FAILURE}
        if response.id is None:
            response = response.model_copy(update={"id": str(uuid4())})
        return {"messages": [response]}

    return assistant
