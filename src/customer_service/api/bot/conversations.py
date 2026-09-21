"""Local conversation endpoint retained from Iteration 1."""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, Path
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from customer_service.graph.response import final_reply


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
    reply: str


def create_router(graph_provider: Callable[[], object]) -> APIRouter:
    router = APIRouter(prefix="/conversations", tags=["bot"])

    @router.post("/{thread_id}/messages", response_model=ConversationResponse)
    async def send_message(
        thread_id: Annotated[str, Path(min_length=1, max_length=128)],
        incoming: IncomingMessage,
        graph: object = Depends(graph_provider),
    ) -> ConversationResponse:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=incoming.message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return ConversationResponse(thread_id=thread_id, reply=final_reply(state))

    return router
