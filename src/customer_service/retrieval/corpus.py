"""Version-controlled, reviewed FAQ source content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter


class FaqDocument(BaseModel):
    id: str
    language: str
    title: str
    answer: str

    model_config = ConfigDict(extra="forbid")

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n\n{self.answer}".strip()

    @property
    def content_hash(self) -> str:
        payload = "\n".join((self.id, self.language, self.searchable_text))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_corpus(path: Path | None = None) -> list[FaqDocument]:
    source = path or Path(__file__).parents[1] / "data" / "faqs.json"
    return TypeAdapter(list[FaqDocument]).validate_python(json.loads(source.read_text(encoding="utf-8")))
