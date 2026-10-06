"""The sole assembly point for graph nodes and routing."""

from __future__ import annotations

from datetime import timedelta

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from customer_service.caching.policy import decide_cache_policy, record_cache_decision
from customer_service.config.settings import Settings
from customer_service.infrastructure.checkpoints import local_checkpointer
from customer_service.infrastructure.faqs import FaqRepository, UnavailableFaqRepository
from customer_service.infrastructure.orders import OrderRepository
from customer_service.infrastructure.promotions import PromotionRepository, default_promotion_repository
from customer_service.llm.factory import create_chat_model
from customer_service.nodes.assistant.node import build_node as build_assistant_node
from customer_service.nodes.format_response.node import build_node as build_format_response_node
from customer_service.nodes.handoff_summary.node import build_node as build_handoff_summary_node
from customer_service.nodes.human_handoff.node import build_node as build_human_handoff_node
from customer_service.nodes.promo_judge.node import (
    PromoJudge,
    build_node as build_promo_judge_node,
    create_structured_judge,
)
from customer_service.nodes.promo_prefilter.node import build_node as build_promo_prefilter_node
from customer_service.nodes.safety_check.gemini import (
    SafetyJudge,
    create_structured_judge as create_structured_safety_judge,
)
from customer_service.nodes.safety_check.jev import JevJudge
from customer_service.nodes.safety_check.node import Classifier, build_node as build_safety_check_node
from customer_service.nodes.tool_dispatch.node import build_node as build_tool_dispatch_node
from customer_service.nodes.verify_identity.gate import (
    ACCOUNT_SPECIFIC_TOOLS,
    build_node as build_account_gate_node,
    pending_tool_calls,
)
from customer_service.nodes.verify_identity.node import (
    build_node as build_verify_identity_node,
    has_unused_claim,
)
from customer_service.state.models import AgentState
from customer_service.tools.billing import create_billing_tool
from customer_service.tools.faq_lookup import create_faq_lookup_tool
from customer_service.tools.order_status import create_order_status_tool
from customer_service.tools.returns import create_returns_tool
from customer_service.verification.attempts import AttemptLimiter, InMemoryAttemptLimiter
from customer_service.verification.grant import VerificationPolicy


def build_graph(
    *,
    settings: Settings,
    repository: OrderRepository,
    faq_repository: FaqRepository | None = None,
    promotion_repository: PromotionRepository | None = None,
    model: BaseChatModel | None = None,
    promo_judge: PromoJudge | None = None,
    safety_judge: SafetyJudge | None = None,
    jev_judge: Classifier | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    attempt_limiter: AttemptLimiter | None = None,
    verification_policy: VerificationPolicy | None = None,
):
    """Build the safety, verification, support, promotion, and handoff graph."""

    # One policy (TTL + clock) and one limiter are shared by every component, so
    # the claim node, the gate, the tools, and the prompts always agree.
    policy = verification_policy or VerificationPolicy(
        ttl=timedelta(seconds=settings.verification_ttl_seconds)
    )
    limiter = attempt_limiter or InMemoryAttemptLimiter(limit=settings.verification_key_failure_limit)
    max_failures = settings.verification_max_thread_failures

    order_status = create_order_status_tool(repository, policy=policy)
    returns = create_returns_tool(repository, policy=policy)
    billing = create_billing_tool(repository, policy=policy)
    faq_lookup = create_faq_lookup_tool(faq_repository or UnavailableFaqRepository())
    tools = [order_status, returns, billing, faq_lookup]
    record_cache_decision(
        decide_cache_policy(
            node_name="assistant",
            prompt_revision="assistant/v1",
            model=settings.model_for("assistant").model,
            tool_schemas=[{"name": tool.name} for tool in tools],
        )
    )
    for node_name in ("promo_judge", "safety_check"):
        record_cache_decision(
            decide_cache_policy(
                node_name=node_name,
                prompt_revision=f"{node_name}/v1",
                model=settings.model_for(node_name).model,
            )
        )
    assistant_model = model or create_chat_model("assistant", settings)
    promotion_repository = promotion_repository or default_promotion_repository()
    judge = promo_judge or create_structured_judge(create_chat_model("promo_judge", settings))
    # Injected assistant models are offline deterministic test doubles. Keep
    # their unrelated graph tests independent of live safety providers.
    live_safety = safety_judge is None and model is None
    safety = safety_judge or (
        create_structured_safety_judge(create_chat_model("safety_check", settings))
        if live_safety
        else _AllowSafetyJudge()
    )
    # Jev is the fast first pass and Gemini the fallback. It is built only for
    # a live graph, or when a test injects one explicitly.
    jev = jev_judge or (
        JevJudge.from_settings(settings) if settings.jev_enabled and live_safety else None
    )
    # Do not turn a backend error into a customer-visible ToolMessage: the
    # wrapper below must fail closed into a handoff.
    raw_tool_node = ToolNode(tools, handle_tool_errors=False)

    def route_after_safety(state: AgentState) -> str:
        if state.safety_action == "refuse":
            return END
        if state.safety_action == "escalate":
            return "handoff_summary"
        # Verification runs early, but only when the customer just supplied a
        # complete, unused order ID + email pair. FAQ-only chat never gets here.
        return "verify_identity" if has_unused_claim(state) else "assistant"

    def route_after_verification(state: AgentState) -> str:
        if state.escalation_reason is not None:
            return "handoff_summary"
        # Success appends nothing, so the customer's message is still last. A
        # failed check appends its fixed reply, which ends the turn.
        return "assistant" if isinstance(state.messages[-1], HumanMessage) else END

    def route_after_assistant(state: AgentState) -> str:
        if state.escalation_reason is not None:
            return "handoff_summary"
        calls = pending_tool_calls(state)
        if not calls:
            return "promo_prefilter"
        # Every account tool call is authorized in code before it can run.
        if any(call["name"] in ACCOUNT_SPECIFIC_TOOLS for call in calls):
            return "account_gate"
        return "tools"

    def route_after_gate(state: AgentState) -> str:
        if state.escalation_reason is not None:
            return "handoff_summary"
        # Allowed calls are still pending; a denial appended a reply instead.
        return "tools" if pending_tool_calls(state) else END

    def route_after_tools(state: AgentState) -> str:
        return "handoff_summary" if state.escalation_reason is not None else "assistant"

    def route_after_prefilter(state: AgentState) -> str:
        return "promo_judge" if state.promo_matches else "format_response"

    graph = StateGraph(AgentState)
    graph.add_node("safety_check", build_safety_check_node(settings=settings, judge=safety, jev=jev))
    graph.add_node(
        "assistant",
        build_assistant_node(settings=settings, model=assistant_model, tools=tools, policy=policy),
    )
    graph.add_node(
        "verify_identity",
        build_verify_identity_node(
            repository, policy=policy, limiter=limiter, max_failures=max_failures
        ),
    )
    graph.add_node(
        "account_gate",
        build_account_gate_node(
            repository, policy=policy, limiter=limiter, max_failures=max_failures
        ),
    )
    graph.add_node("tools", build_tool_dispatch_node(raw_tool_node))
    graph.add_node(
        "promo_prefilter",
        build_promo_prefilter_node(
            promotion_repository=promotion_repository,
            order_repository=repository,
            policy=policy,
        ),
    )
    graph.add_node("promo_judge", build_promo_judge_node(settings=settings, judge=judge))
    graph.add_node("format_response", build_format_response_node())
    graph.add_node("handoff_summary", build_handoff_summary_node(policy=policy))
    graph.add_node("human_handoff", build_human_handoff_node())

    graph.add_edge(START, "safety_check")
    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "verify_identity": "verify_identity",
            "assistant": "assistant",
            "handoff_summary": "handoff_summary",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "verify_identity",
        route_after_verification,
        {"assistant": "assistant", "handoff_summary": "handoff_summary", END: END},
    )
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "account_gate": "account_gate",
            "tools": "tools",
            "promo_prefilter": "promo_prefilter",
            "handoff_summary": "handoff_summary",
        },
    )
    graph.add_conditional_edges(
        "account_gate",
        route_after_gate,
        {"tools": "tools", "handoff_summary": "handoff_summary", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"assistant": "assistant", "handoff_summary": "handoff_summary"},
    )
    graph.add_conditional_edges(
        "promo_prefilter",
        route_after_prefilter,
        {"promo_judge": "promo_judge", "format_response": "format_response"},
    )
    graph.add_edge("promo_judge", "format_response")
    graph.add_edge("format_response", END)
    graph.add_edge("handoff_summary", "human_handoff")
    graph.add_edge("human_handoff", END)
    return graph.compile(checkpointer=checkpointer or local_checkpointer())


class _AllowSafetyJudge:
    def invoke(self, input: object) -> dict[str, str]:
        return {"action": "allow"}
