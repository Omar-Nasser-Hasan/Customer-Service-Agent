"""Local FAQ corpus and deterministic retrieval baseline for Iteration 2."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter


class FaqEntry(BaseModel):
    id: str
    title: str
    keywords: list[str]
    answer: str

    model_config = ConfigDict(extra="forbid")


class FaqRepository:
    def __init__(self, entries: list[FaqEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_path(cls, path: Path) -> "FaqRepository":
        raw_entries = json.loads(path.read_text(encoding="utf-8"))
        return cls(TypeAdapter(list[FaqEntry]).validate_python(raw_entries))

    def search(self, query: str) -> FaqEntry | None:
        tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
        if not tokens:
            return None
        scored = [
            (
                len(tokens.intersection({keyword.casefold() for keyword in entry.keywords})),
                entry,
            )
            for entry in self._entries
        ]
        score, entry = max(scored, key=lambda item: item[0])
        return entry if score else None


def default_faq_repository() -> FaqRepository:
    return FaqRepository.from_path(Path(__file__).parents[1] / "data" / "faqs.json")
