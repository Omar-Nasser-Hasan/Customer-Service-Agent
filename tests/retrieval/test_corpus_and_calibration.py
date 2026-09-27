from __future__ import annotations

import pytest

from customer_service.retrieval.calibration import calibrate
from customer_service.retrieval.corpus import load_corpus


def test_bilingual_corpus_has_one_document_per_faq_language_pair() -> None:
    documents = load_corpus()
    assert len(documents) == 6
    assert {(document.id, document.language) for document in documents} == {
        ("shipping-times", "en"), ("shipping-times", "ar"),
        ("return-window", "en"), ("return-window", "ar"),
        ("payment-methods", "en"), ("payment-methods", "ar"),
    }
    assert all(document.searchable_text and len(document.content_hash) == 64 for document in documents)


class RankedRepository:
    def __init__(self, scores: dict[str, tuple[str | None, float | None]]) -> None:
        self.scores = scores

    async def best_similarities(self, queries: list[str]):
        from customer_service.infrastructure.faqs import FaqEntry
        return [(None, score) if faq_id is None else (FaqEntry(id=faq_id, title="title", answer="answer"), score) for faq_id, score in (self.scores[query] for query in queries)]


async def test_calibration_selects_threshold_only_with_separation() -> None:
    fixtures = {"positive": [{"query": "yes", "faq_id": "shipping-times"}], "negative": [{"query": "no"}]}
    result = await calibrate(RankedRepository({"yes": ("shipping-times", 0.8), "no": ("shipping-times", 0.2)}), fixtures=fixtures)  # type: ignore[arg-type]
    assert result.threshold == 0.5


async def test_calibration_rejects_overlapping_positive_and_negative_scores() -> None:
    fixtures = {"positive": [{"query": "yes", "faq_id": "shipping-times"}], "negative": [{"query": "no"}]}
    with pytest.raises(RuntimeError, match="overlap"):
        await calibrate(RankedRepository({"yes": ("shipping-times", 0.4), "no": ("shipping-times", 0.4)}), fixtures=fixtures)  # type: ignore[arg-type]
