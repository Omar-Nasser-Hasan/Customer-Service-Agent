"""Public FAQ lookup tool; identity verification is never required."""

from __future__ import annotations

import json
import inspect
from typing import Literal

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.faqs import FaqRepository


def create_faq_lookup_tool(repository: FaqRepository) -> BaseTool:
    @tool("faq_lookup")
    async def faq_lookup(query: str, language: Literal["en", "ar"] = "en") -> str:
        """Find one exact public FAQ answer.

        Use a short standalone question about one topic and pass ``ar`` for
        Arabic or Arabizi input. The language selects the stored answer;
        Egyptian-vs-MSA reply register is handled by the assistant.
        """

        search = repository.search
        if "language" in inspect.signature(search).parameters:
            entry = await search(query, language=language)
        else:  # compatibility with small offline test doubles
            entry = await search(query)
        if entry is None:
            return json.dumps(
                {
                    "found": False,
                    "message": "No matching FAQ was found.",
                }
            )
        return json.dumps(
            {
                "found": True,
                "faq_id": entry.id,
                "title": entry.title,
                "answer": entry.answer,
            }
        )

    return faq_lookup
