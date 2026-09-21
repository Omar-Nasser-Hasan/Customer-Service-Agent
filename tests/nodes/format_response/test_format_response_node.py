from __future__ import annotations

from customer_service.nodes.format_response.node import build_node
from customer_service.state.models import AgentState


def test_formatter_preserves_draft_and_reuses_its_message_id() -> None:
    update = build_node()(
        AgentState(
            draft_response="Your order has shipped.",
            draft_message_id="draft-message",
            promo_line="Catalog-owned promotion.",
        )
    )

    message = update["messages"][0]
    assert message.id == "draft-message"
    assert message.content == "Your order has shipped.\n\nCatalog-owned promotion."
    assert update["already_promoted"] is True
