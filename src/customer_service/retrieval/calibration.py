"""Reproducible threshold selection for committed bilingual fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from customer_service.retrieval.repository import PostgresFaqRepository


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    lowest_positive: float
    highest_negative: float


async def calibrate(
    repository: PostgresFaqRepository,
    path: Path | None = None,
    fixtures: dict[str, list[dict[str, str]]] | None = None,
) -> CalibrationResult:
    if fixtures is None:
        source = path or Path(__file__).parents[1] / "data" / "faq_calibration.json"
        fixtures = json.loads(source.read_text(encoding="utf-8"))
    cases = [*fixtures["positive"], *fixtures["negative"]]
    ranked = await repository.best_similarities([case["query"] for case in cases])
    positives: list[float] = []
    negatives: list[float] = []
    for case, (entry, score) in zip(fixtures["positive"], ranked[: len(fixtures["positive"])], strict=True):
        if entry is None or score is None or entry.id != case["faq_id"]:
            raise RuntimeError(f"Calibration positive did not retrieve {case['faq_id']}: {case['query']!r}")
        positives.append(score)
    for _, score in ranked[len(fixtures["positive"]) :]:
        negatives.append(score if score is not None else -1.0)
    low, high = min(positives), max(negatives)
    if low <= high:
        raise RuntimeError("Calibration failed: intended and no-match similarities overlap; review corpus or fixtures")
    return CalibrationResult(threshold=round((low + high) / 2, 6), lowest_positive=low, highest_negative=high)
