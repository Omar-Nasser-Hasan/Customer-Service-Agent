from __future__ import annotations

from datetime import date

from customer_service.infrastructure.promotions import Promotion, PromotionRepository


def test_catalog_ranks_topic_and_derived_order_state_matches() -> None:
    repository = PromotionRepository(
        [
            Promotion(
                promo_id="TOPIC-AND-STATE",
                text="A",
                topic_tags=["shipping"],
                order_state_tags=["shipped"],
                active=True,
                starts_on=date(2026, 9, 1),
                ends_on=date(2026, 9, 30),
                is_synthetic=True,
            ),
            Promotion(
                promo_id="TOPIC-ONLY",
                text="B",
                topic_tags=["shipping"],
                order_state_tags=[],
                active=True,
                starts_on=date(2026, 9, 1),
                ends_on=date(2026, 9, 30),
                is_synthetic=True,
            ),
        ],
        today=lambda: date(2026, 9, 21),
    )

    matches = repository.active_matches(
        conversation_tokens={"shipping"},
        verified=True,
        order_status="shipped",
        delivered=False,
    )

    assert [match.promo_id for match in matches] == ["TOPIC-AND-STATE", "TOPIC-ONLY"]
    assert [match.relevance_score for match in matches] == [2.0, 1.0]


def test_catalog_excludes_inactive_and_expired_promotions() -> None:
    repository = PromotionRepository(
        [
            Promotion(
                promo_id="EXPIRED",
                text="Expired",
                topic_tags=["shipping"],
                order_state_tags=[],
                active=True,
                starts_on=date(2026, 1, 1),
                ends_on=date(2026, 9, 20),
                is_synthetic=True,
            ),
            Promotion(
                promo_id="INACTIVE",
                text="Inactive",
                topic_tags=["shipping"],
                order_state_tags=[],
                active=False,
                starts_on=date(2026, 9, 1),
                ends_on=date(2026, 9, 30),
                is_synthetic=True,
            ),
        ],
        today=lambda: date(2026, 9, 21),
    )

    assert repository.active_matches(
        conversation_tokens={"shipping"},
        verified=False,
        order_status=None,
        delivered=False,
    ) == []
