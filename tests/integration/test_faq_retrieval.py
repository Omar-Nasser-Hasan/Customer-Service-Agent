from __future__ import annotations

import pytest
from docker.errors import DockerException
from psycopg_pool import AsyncConnectionPool
from testcontainers.community.postgres import PostgresContainer

from customer_service.config.settings import Settings
from customer_service.retrieval.corpus import FaqDocument
from customer_service.retrieval.repository import PostgresFaqRepository


class DeterministicEmbeddings:
    """Offline vectors let this validate pgvector mechanics without Voyage."""

    async def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            token = "shipping" if any(word in text.casefold() for word in ("shipping", "package", "يصل", "الشحن")) else "returns" if any(word in text.casefold() for word in ("return", "إرجاع")) else "payment"
            position = {"shipping": 0, "returns": 1, "payment": 2}[token]
            vector = [0.0] * 1024
            vector[position] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
async def faq_pool():
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as postgres:
            dsn = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
            pool = AsyncConnectionPool(conninfo=dsn, open=False)
            await pool.open()
            try:
                yield pool
            finally:
                await pool.close()
    except DockerException as error:
        pytest.skip(f"Docker daemon is unavailable: {error}")


@pytest.mark.integration
async def test_pgvector_sync_is_idempotent_and_search_keeps_safe_contract(faq_pool: AsyncConnectionPool) -> None:
    settings = Settings(voyage_api_key="test", faq_min_similarity=0.8)
    repository = PostgresFaqRepository(pool=faq_pool, embeddings=DeterministicEmbeddings(), settings=settings)
    documents = [
        FaqDocument(id="shipping-times", language="en", title="Shipping times", answer="Packages arrive soon."),
        FaqDocument(id="shipping-times", language="ar", title="أوقات الشحن", answer="متى يصل طلبي؟"),
        FaqDocument(id="return-window", language="en", title="Returns", answer="Return an item."),
    ]
    await repository.setup()
    assert (await repository.sync(documents)).inserted == 3
    assert (await repository.sync(documents)).unchanged == 3
    revised = [
        documents[0].model_copy(update={"answer": "Packages arrive in three to five business days."}),
        *documents[1:],
    ]
    assert (await repository.sync(revised)).updated == 1
    model_changed = PostgresFaqRepository(
        pool=faq_pool,
        embeddings=DeterministicEmbeddings(),
        settings=Settings(voyage_api_key="test", voyage_model="test-model-v2", faq_min_similarity=0.8),
    )
    assert (await model_changed.sync(revised)).updated == 3
    match = await repository.search("when will my package arrive?")
    assert match is not None and match.id == "shipping-times"
    assert await repository.search("cryptocurrency mining") is None
    changed = revised[:-1]
    assert (await repository.sync(changed)).deleted == 1
