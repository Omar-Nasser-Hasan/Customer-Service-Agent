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


class FaqVariant(BaseModel):
    faq_id: str
    text: str
    dialect: str
    review_status: str = "draft"
    synthetic: bool = True

    model_config = ConfigDict(extra="forbid")

    @property
    def content_hash(self) -> str:
        payload = "\n".join((self.faq_id, self.dialect, self.text, self.review_status, str(self.synthetic)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_corpus(path: Path | None = None, *, store_name: str = "Souqly") -> list[FaqDocument]:
    source = path or Path(__file__).parents[1] / "data" / "faqs.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    for item in raw:
        item["answer"] = item["answer"].replace("{store_name}", store_name)
    return TypeAdapter(list[FaqDocument]).validate_python(raw)


def load_variants(path: Path | None = None) -> list[FaqVariant]:
    source = path or Path(__file__).parents[1] / "data" / "faq_variants.json"
    return TypeAdapter(list[FaqVariant]).validate_json(source.read_bytes())


def load_sink(path: Path | None = None) -> list[FaqVariant]:
    source = path or Path(__file__).parents[1] / "data" / "faq_sink.json"
    return TypeAdapter(list[FaqVariant]).validate_json(source.read_bytes())
