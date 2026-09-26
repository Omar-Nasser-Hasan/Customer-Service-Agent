"""Customer-facing assistant reasoning node."""

from __future__ import annotations

from typing import Callable
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from customer_service.config.settings import Settings
from customer_service.nodes.assistant.prompt import build_system_prompt
from customer_service.state.models import AgentState, EscalationReason


def build_node(
    *, settings: Settings, model: BaseChatModel, tools: list[BaseTool]
) -> Callable[[AgentState], dict[str, object]]:
    model_with_tools = model.bind_tools(tools)
    prompt = SystemMessage(content=build_system_prompt(settings))

    def assistant(state: AgentState) -> dict[str, object]:
        verification_context = SystemMessage(
            content=(
                "Internal verification context: this conversation is verified for account-specific support."
                if state.verified
                else "Internal verification context: this conversation is not verified."
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
