from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from customer_service.nodes.promo_judge.node import PromoDecision
from customer_service.nodes.safety_check.node import SafetyDecision


class ScriptedChatModel(BaseChatModel):
    """Offline model that returns a known sequence of LangChain messages."""

    responses: list[AIMessage]
    response_index: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(self, tools: list[object], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[min(self.response_index, len(self.responses) - 1)]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FailingChatModel(ScriptedChatModel):
    """Offline model stand-in for testing a provider outage."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("provider unavailable")


class ScriptedPromoJudge:
    """Offline structured-output stand-in that also records its model input."""

    def __init__(self, responses: list[PromoDecision | dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[object] = []
        self.response_index = 0

    def invoke(self, input: object) -> PromoDecision | dict[str, Any]:
        self.calls.append(input)
        response = self.responses[min(self.response_index, len(self.responses) - 1)]
        self.response_index += 1
        return response


def declining_promo_judge() -> ScriptedPromoJudge:
    return ScriptedPromoJudge([PromoDecision(include_promo=False)])


class ScriptedSafetyJudge:
    """Offline safety-classifier stand-in that records redacted inputs."""

    def __init__(self, responses: list[SafetyDecision | dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.calls: list[object] = []
        self.response_index = 0

    def invoke(self, input: object) -> SafetyDecision | dict[str, Any]:
        self.calls.append(input)
        response = self.responses[min(self.response_index, len(self.responses) - 1)]
        self.response_index += 1
        if isinstance(response, Exception):
            raise response
        return response


def allowing_safety_judge() -> ScriptedSafetyJudge:
    return ScriptedSafetyJudge([SafetyDecision(action="allow")])
