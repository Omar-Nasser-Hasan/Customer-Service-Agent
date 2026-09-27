"""Customer conversation endpoint, including durable-handoff response states."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Callable, Literal

from fastapi import APIRouter, Path, Response
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from customer_service.graph.response import final_reply
from customer_service.observability.runtime import ObservabilityRuntime
from customer_service.services.cases import CaseService


class IncomingMessage(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not (message := value.strip()):
            raise ValueError("message must not be blank")
        return message


class ConversationResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "refused", "handoff_open", "handoff_active"]
    reply: str | None


def _handoff_is_active(state: dict[str, object]) -> bool:
    return bool(state.get("escalated") and state.get("case_status") in {"open", "claimed"})


def create_router(
    graph_provider: Callable[[], Any],
    observability_provider: Callable[[], ObservabilityRuntime] | None = None,
    case_service_provider: Callable[[], CaseService] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/conversations", tags=["bot"])

    @router.post("/{thread_id}/messages", response_model=ConversationResponse)
    async def send_message(
        thread_id: Annotated[str, Path(min_length=1, max_length=128)],
        incoming: IncomingMessage,
        response: Response,
    ) -> ConversationResponse:
        graph = graph_provider()
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if observability_provider is not None:
            config["callbacks"] = observability_provider().callbacks(thread_id)
        snapshot = await graph.aget_state(config)
        if _handoff_is_active(snapshot.values):
            # The graph remains paused, but customer activity still resets the
            # retention clock for an open case.
            if case_service_provider is not None:
                await case_service_provider().record_active_customer_message(thread_id, incoming.message)
            else:
                await graph.aupdate_state(config, {"messages": [HumanMessage(content=incoming.message)], "last_activity_at": datetime.now(UTC)})
            response.status_code = 202
            return ConversationResponse(thread_id=thread_id, status="handoff_active", reply=None)

        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=incoming.message)]},
            config=config,
        )
        if _handoff_is_active(state):
            if case_service_provider is not None:
                await case_service_provider().reconcile_handoff(thread_id, state)
            response.status_code = 202
            return ConversationResponse(
                thread_id=thread_id,
                status="handoff_open",
                reply=final_reply(state),
            )
        return ConversationResponse(
            thread_id=thread_id,
            status="refused" if state.get("safety_action") == "refuse" else "completed",
            reply=final_reply(state),
        )

    return router
