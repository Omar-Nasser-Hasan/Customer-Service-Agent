"""Voyage embedding boundary; tests inject this protocol rather than call Voyage."""

from __future__ import annotations

import asyncio
from typing import Literal, Protocol

from customer_service.config.settings import Settings


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"]) -> list[list[float]]: ...


class VoyageEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        import voyageai

        self._model = Settings.voyage_model
        self._dimensions = Settings.voyage_embedding_dimensions
        self._retries = Settings.faq_embedding_retries
        self._retry_delay = Settings.faq_embedding_retry_delay_seconds

    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"]) -> list[list[float]]:
        for attempt in range(self._retries + 1):
            try:
                result = await self._client.embed(
                    texts, model=self._model, input_type=input_type, output_dimension=self._dimensions
                )
                return [list(vector) for vector in result.embeddings]
            except Exception:
                if attempt >= self._retries:
                    # The caller turns this bounded failure into a handoff; never
                    # return a guessed or partial FAQ result.
                    raise
                await asyncio.sleep(self._retry_delay * (2**attempt))
