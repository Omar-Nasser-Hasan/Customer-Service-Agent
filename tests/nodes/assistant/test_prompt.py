from customer_service.config.settings import Settings
from customer_service.nodes.assistant.prompt import build_system_prompt


def test_assistant_prompt_pins_language_scope_and_promo_boundaries() -> None:
    prompt = build_system_prompt(Settings(persona_name="Layla", store_name="Souqly"))

    assert "Layla" in prompt and "Souqly" in prompt
    assert "language of the customer's latest message" in prompt
    assert "answer the general policy from faq_lookup first" in prompt
    assert "Never offer or mention discounts" in prompt
    assert "Use account tools only for that order" in prompt