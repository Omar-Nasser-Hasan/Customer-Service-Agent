"""Synthetic promotion catalog adapter for the Iteration 3 prototype."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, model_validator

from customer_service.state.models import PromoMatch


class Promotion(BaseModel):
    promo_id: str
    text: str
    topic_tags: list[str]
    order_state_tags: list[str]
    active: bool
    starts_on: date
    ends_on: date
    is_synthetic: bool

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_dates(self) -> "Promotion":
        if self.ends_on < self.starts_on:
            raise ValueError("Promotion end date must not precede its start date")
        return self


class PromotionRepository:
    """Ranks active catalog entries without exposing sensitive customer fields."""

    def __init__(self, promotions: list[Promotion], *, today: Callable[[], date] = date.today) -> None:
        self._promotions = promotions
        self._today = today

    @classmethod
    def from_path(
        cls, path: Path, *, today: Callable[[], date] = date.today
    ) -> "PromotionRepository":
        raw_promotions = json.loads(path.read_text(encoding="utf-8"))
        return cls([Promotion.model_validate(item) for item in raw_promotions], today=today)

    def active_matches(
        self,
        *,
        conversation_tokens: set[str],
        verified: bool,
        order_status: str | None,
        delivered: bool,
    ) -> list[PromoMatch]:
        """Return active matches ranked by topic and minimized order-state context."""

        state_tags = {"verified"} if verified else set()
        if order_status:
            state_tags.add(order_status.casefold())
        if delivered:
            state_tags.add("delivered")

        today = self._today()
        matches: list[PromoMatch] = []
        for promotion in self._promotions:
            if not promotion.active or not promotion.starts_on <= today <= promotion.ends_on:
                continue
            topic_score = len(conversation_tokens.intersection(set(promotion.topic_tags)))
            state_score = len(state_tags.intersection(set(promotion.order_state_tags)))
            score = topic_score + state_score
            if score:
                matches.append(
                    PromoMatch(
                        promo_id=promotion.promo_id,
                        text=promotion.text,
                        relevance_score=float(score),
                    )
                )
        return sorted(matches, key=lambda match: (-match.relevance_score, match.promo_id))


def default_promotion_repository() -> PromotionRepository:
    return PromotionRepository.from_path(Path(__file__).parents[1] / "data" / "promotions.json")
