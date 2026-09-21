"""Use the promo-specific model to make a fail-closed promotion decision."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customer_service.config.settings import Settings
from customer_service.nodes.promo_judge.prompt import build_system_prompt
from customer_service.state.models import AgentState


class PromoDecision(BaseModel):
    include_promo: bool
    selected_promo_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_selection(self) -> "PromoDecision":
        if self.include_promo != bool(self.selected_promo_id):
            raise ValueError("A promotion decision must include exactly one selected ID or none")
        return self


class PromoJudge(Protocol):
    def invoke(self, input: object) -> PromoDecision | dict[str, Any]: ...


def create_structured_judge(model: BaseChatModel) -> PromoJudge:
    """Bind Gemini's structured-output support to the limited decision schema."""

    return model.with_structured_output(PromoDecision)


def build_node(
    *, settings: Settings, judge: PromoJudge
) -> Callable[[AgentState], dict[str, object]]:
    prompt = SystemMessage(content=build_system_prompt(settings))

    def promo_judge(state: AgentState) -> dict[str, object]:
        if not state.draft_response or not state.promo_matches:
            return {"promo_line": None}

        candidates = [
            {
                "promo_id": match.promo_id,
                "text": match.text,
                "relevance_score": match.relevance_score,
            }
            for match in state.promo_matches
        ]
        request = HumanMessage(
            content=(
                "Completed support response:\n"
                f"{state.draft_response}\n\n"
                "Eligible catalog promotions:\n"
                f"{json.dumps(candidates)}"
            )
        )
        try:
            result = PromoDecision.model_validate(judge.invoke([prompt, request]))
        except (TypeError, ValueError, ValidationError):
            return {"promo_line": None}

        if not result.include_promo:
            return {"promo_line": None}
        selected = next(
            (match for match in state.promo_matches if match.promo_id == result.selected_promo_id),
            None,
        )
        # The catalog owns customer-facing offer text; the model cannot invent it.
        return {"promo_line": selected.text if selected else None}

    return promo_judge
