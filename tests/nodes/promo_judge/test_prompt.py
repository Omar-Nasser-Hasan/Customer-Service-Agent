from customer_service.config.settings import Settings
from customer_service.nodes.promo_judge.prompt import build_system_prompt


def test_promo_judge_prompt_constrains_offer_selection_and_support_integrity() -> None:
    prompt = build_system_prompt(Settings(persona_name="Layla", store_name="Souqly"))

    assert "Layla" in prompt
    assert "Souqly" in prompt
    assert "must never be rewritten" in prompt
    assert "choose exactly one supplied" in prompt
    assert "Never invent offers" in prompt
