"""Customer-facing response extraction from a completed graph state."""

from langchain_core.messages import AIMessage

from customer_service.state.message_text import content_to_text
from customer_service.state.models import AgentState


def final_reply(state: dict[str, object] | AgentState) -> str:
    messages = state["messages"] if isinstance(state, dict) else state.messages
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            if isinstance(message.content, str):
                return message.content
            return content_to_text(message.content)
    raise RuntimeError("The agent completed without a customer-facing response")
