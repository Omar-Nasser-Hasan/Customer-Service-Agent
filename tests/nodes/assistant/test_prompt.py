from customer_service.config.settings import Settings
from customer_service.nodes.assistant.prompt import build_system_prompt


def test_assistant_prompt_pins_language_scope_and_promo_boundaries() -> None:
    prompt = build_system_prompt(Settings(persona_name="Layla", store_name="Souqly"))

    assert "Layla" in prompt and "Souqly" in prompt
    assert "language of the customer's latest message" in prompt
    assert "answer the general policy from faq_lookup first" in prompt
    assert "Never offer or mention discounts" in prompt


def test_assistant_prompt_scopes_verified_customers_to_their_own_orders() -> None:
    prompt = build_system_prompt(Settings())

    assert "may ask about any of their own orders" in prompt
    assert "never the email again" in prompt
    # The old single-order rule is gone; ownership is enforced in code instead.
    assert "that order only" not in prompt
    assert "Use account tools only for that order" not in prompt
    # A failed lookup is never explained and never hints at other orders.
    assert "never hint that other orders exist" in prompt
