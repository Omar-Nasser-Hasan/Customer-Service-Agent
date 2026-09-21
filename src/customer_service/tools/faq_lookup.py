"""Public FAQ lookup tool; identity verification is never required."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.faqs import FaqRepository


def create_faq_lookup_tool(repository: FaqRepository) -> BaseTool:
    @tool("faq_lookup")
    def faq_lookup(query: str) -> str:
        """Find a policy, shipping, payment, or general-support FAQ answer."""

        entry = repository.search(query)
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
