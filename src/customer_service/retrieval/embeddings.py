"""Voyage embedding boundary; tests inject this protocol rather than call Voyage."""

from __future__ import annotations

from typing import Literal, Protocol

from customer_service.config.settings import Settings


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"]) -> list[list[float]]: ...


class VoyageEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(api_key=settings.require_secret("VOYAGE_API_KEY", settings.voyage_api_key))
        self._model = settings.voyage_model
        self._dimensions = settings.voyage_embedding_dimensions

    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model, input_type=input_type, output_dimension=self._dimensions)
        return [list(vector) for vector in result.embeddings]
