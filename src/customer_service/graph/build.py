"""The sole assembly point for graph nodes and routing."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from customer_service.config.settings import Settings
from customer_service.infrastructure.checkpoints import local_checkpointer
from customer_service.infrastructure.faqs import FaqRepository, default_faq_repository
from customer_service.infrastructure.orders import OrderRepository
from customer_service.infrastructure.promotions import (
    PromotionRepository,
    default_promotion_repository,
)
from customer_service.llm.factory import create_chat_model
from customer_service.nodes.assistant.node import build_node as build_assistant_node
from customer_service.nodes.format_response.node import build_node as build_format_response_node
from customer_service.nodes.promo_judge.node import (
    PromoJudge,
    build_node as build_promo_judge_node,
    create_structured_judge,
)
from customer_service.nodes.promo_prefilter.node import build_node as build_promo_prefilter_node
from customer_service.nodes.verify_identity.node import (
    build_node as build_verify_identity_node,
    has_unverified_account_intent,
    requires_identity_verification,
)
from customer_service.state.models import AgentState
from customer_service.tools.billing import create_billing_tool
from customer_service.tools.faq_lookup import create_faq_lookup_tool
from customer_service.tools.order_status import create_order_status_tool
from customer_service.tools.returns import create_returns_tool


def build_graph(
    *,
    settings: Settings,
    repository: OrderRepository,
    faq_repository: FaqRepository | None = None,
    promotion_repository: PromotionRepository | None = None,
    model: BaseChatModel | None = None,
    promo_judge: PromoJudge | None = None,
):
    """Build the Iteration 3 tool loop, identity gate, and promo branch."""

    order_status = create_order_status_tool(repository)
    returns = create_returns_tool(repository)
    billing = create_billing_tool(repository)
    faq_lookup = create_faq_lookup_tool(faq_repository or default_faq_repository())
    tools = [order_status, returns, billing, faq_lookup]
    assistant_model = model or create_chat_model("assistant", settings)
    promotion_repository = promotion_repository or default_promotion_repository()
    judge = promo_judge or create_structured_judge(create_chat_model("promo_judge", settings))

    def route_after_assistant(state: AgentState) -> str:
        if requires_identity_verification(state):
            return "verify_identity"
        tool_route = tools_condition(state)
        if tool_route == "tools":
            return "tools"
        if has_unverified_account_intent(state):
            return END
        return "promo_prefilter"

    def route_after_verification(state: AgentState) -> str:
        return "assistant" if state.verified else END

    def route_after_prefilter(state: AgentState) -> str:
        return "promo_judge" if state.promo_matches else "format_response"

    graph = StateGraph(AgentState)
    graph.add_node(
        "assistant",
        build_assistant_node(
            settings=settings,
            model=assistant_model,
            tools=tools,
        ),
    )
    graph.add_node("verify_identity", build_verify_identity_node(repository))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node(
        "promo_prefilter",
        build_promo_prefilter_node(
            promotion_repository=promotion_repository,
            order_repository=repository,
        ),
    )
    graph.add_node("promo_judge", build_promo_judge_node(settings=settings, judge=judge))
    graph.add_node("format_response", build_format_response_node())
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "verify_identity": "verify_identity",
            "tools": "tools",
            "promo_prefilter": "promo_prefilter",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "verify_identity",
        route_after_verification,
        {"assistant": "assistant", END: END},
    )
    graph.add_edge("tools", "assistant")
    graph.add_conditional_edges(
        "promo_prefilter",
        route_after_prefilter,
        {"promo_judge": "promo_judge", "format_response": "format_response"},
    )
    graph.add_edge("promo_judge", "format_response")
    graph.add_edge("format_response", END)
    return graph.compile(checkpointer=local_checkpointer())
