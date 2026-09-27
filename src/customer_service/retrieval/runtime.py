"""Application-lifetime resources for production FAQ retrieval."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg_pool import AsyncConnectionPool

from customer_service.config.settings import Settings
from customer_service.retrieval.corpus import load_corpus
from customer_service.retrieval.embeddings import VoyageEmbeddingProvider
from customer_service.retrieval.repository import PostgresFaqRepository


@asynccontextmanager
async def faq_repository_runtime(settings: Settings) -> AsyncIterator[PostgresFaqRepository]:
    pool = AsyncConnectionPool(conninfo=settings.require_secret("POSTGRES_DSN", settings.postgres_dsn), open=False, min_size=1, max_size=4)
    await pool.open()
    try:
        repository = PostgresFaqRepository(pool=pool, embeddings=VoyageEmbeddingProvider(settings), settings=settings)
        await repository.setup()
        if settings.faq_sync_on_startup:
            await repository.sync(load_corpus())
        yield repository
    finally:
        await pool.close()
