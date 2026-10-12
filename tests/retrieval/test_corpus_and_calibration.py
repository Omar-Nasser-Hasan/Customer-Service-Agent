from __future__ import annotations

import json
from pathlib import Path
import pytest

from customer_service.retrieval.calibration import calibrate
from customer_service.retrieval.corpus import load_corpus, load_sink, load_variants
from customer_service.tools.returns import RETURN_WINDOW_DAYS
from customer_service.infrastructure.orders import OrderState


def test_bilingual_corpus_has_one_document_per_faq_language_pair() -> None:
    documents = load_corpus()
    assert len(documents) == 22
    assert {(document.id, document.language) for document in documents} == {
        (faq_id, language)
        for faq_id in {
            "shipping-times", "shipping-cost", "shipping-coverage", "order-tracking", "order-statuses",
            "return-window", "return-process", "refund-timing", "payment-methods", "payment-charging", "support-hours",
        }
        for language in {"en", "ar"}
    }
    assert all("{store_name}" not in document.answer for document in documents)
    assert all(document.searchable_text and len(document.content_hash) == 64 for document in documents)


def test_return_window_matches_returns_policy_constant() -> None:
    documents = {document.language: document for document in load_corpus() if document.id == "return-window"}
    assert str(RETURN_WINDOW_DAYS) == "14"
    assert all("14" in document.answer for document in documents.values())


def test_order_statuses_faq_covers_every_order_state() -> None:
    answer = next(document.answer for document in load_corpus() if document.id == "order-statuses" and document.language == "en")
    labels = {
        "processing": "Processing",
        "packed": "Packed",
        "shipped": "Shipped",
        "out_for_delivery": "Out for delivery",
        "delivered": "Delivered",
        "cancelled": "Cancelled",
    }
    assert set(labels) == set(OrderState.__args__)
    assert all(label in answer for label in labels.values())


def test_variant_and_sink_corpora_are_complete_and_synthetic() -> None:
    variants = load_variants()
    sink = load_sink()
    faq_ids = {document.id for document in load_corpus()}
    assert len(variants) == 330
    assert len(sink) == 68
    assert {variant.faq_id for variant in variants} == faq_ids
    assert {variant.faq_id for variant in sink} == {"__no_answer__"}
    assert all(variant.review_status == "draft" and variant.synthetic for variant in [*variants, *sink])
    eval_cases = json.loads((Path(__file__).with_name("faq_eval_cases.json")).read_text(encoding="utf-8"))
    assert not ({variant.text for variant in variants} & {case["text"] for case in eval_cases})


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
