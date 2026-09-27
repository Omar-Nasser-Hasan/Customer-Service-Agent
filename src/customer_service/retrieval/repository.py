"""Postgres/pgvector repository for customer-safe FAQ search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from customer_service.config.settings import Settings
from customer_service.infrastructure.faqs import FaqEntry
from customer_service.retrieval.corpus import FaqDocument
from customer_service.retrieval.embeddings import EmbeddingProvider


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


@dataclass(frozen=True)
class SyncResult:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0


class PostgresFaqRepository:
    def __init__(self, *, pool: AsyncConnectionPool, embeddings: EmbeddingProvider, settings: Settings) -> None:
        self._pool, self._embeddings, self._settings = pool, embeddings, settings

    async def setup(self) -> None:
        dimensions = self._settings.voyage_embedding_dimensions
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS faq_documents (
                        faq_id TEXT NOT NULL, language TEXT NOT NULL, title TEXT NOT NULL,
                        answer TEXT NOT NULL, searchable_text TEXT NOT NULL, content_hash TEXT NOT NULL,
                        corpus_version TEXT NOT NULL,
                        embedding_model TEXT NOT NULL, embedding_dimensions INTEGER NOT NULL,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (faq_id, language)
                    )""")
                await cursor.execute(
                    "ALTER TABLE faq_documents ADD COLUMN IF NOT EXISTS corpus_version TEXT NOT NULL DEFAULT 'v1'"
                )
                await cursor.execute("""
                    SELECT pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS type_name
                    FROM pg_attribute AS attribute
                    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                    WHERE relation.relname = 'faq_documents' AND attribute.attname = 'embedding'
                """)
                row = await cursor.fetchone()
                if row is not None and row[0] != f"vector({dimensions})":
                    # A vector column cannot hold two widths. This is an explicit
                    # corpus rebuild, not an unsafe partial conversion.
                    await cursor.execute("DROP INDEX IF EXISTS faq_documents_embedding_hnsw")
                    await cursor.execute("TRUNCATE faq_documents")
                    await cursor.execute(
                        f"ALTER TABLE faq_documents ALTER COLUMN embedding TYPE vector({dimensions})"
                    )
                await cursor.execute("""
                    CREATE INDEX IF NOT EXISTS faq_documents_embedding_hnsw
                    ON faq_documents USING hnsw (embedding vector_cosine_ops)
                """)
            await connection.commit()

    async def sync(self, documents: list[FaqDocument]) -> SyncResult:
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute("SELECT faq_id, language, content_hash, corpus_version, embedding_model, embedding_dimensions FROM faq_documents")
                existing = {(row["faq_id"], row["language"]): row for row in await cursor.fetchall()}
        inserted = updated = unchanged = 0
        changed: list[FaqDocument] = []
        for document in documents:
            row = existing.get((document.id, document.language))
            if row is None:
                inserted += 1; changed.append(document)
            elif row["content_hash"] != document.content_hash or row["corpus_version"] != self._settings.faq_corpus_version or row["embedding_model"] != self._settings.voyage_model or row["embedding_dimensions"] != self._settings.voyage_embedding_dimensions:
                updated += 1; changed.append(document)
            else:
                unchanged += 1
        vectors = await self._embeddings.embed([item.searchable_text for item in changed], input_type="document") if changed else []
        if len(vectors) != len(changed):
            raise RuntimeError("Embedding provider returned an unexpected number of vectors")
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                for document, embedding in zip(changed, vectors, strict=True):
                    if len(embedding) != self._settings.voyage_embedding_dimensions:
                        raise RuntimeError("Embedding provider returned an unexpected vector dimension")
                    await cursor.execute("""
                        INSERT INTO faq_documents (faq_id, language, title, answer, searchable_text, content_hash,
                          corpus_version, embedding_model, embedding_dimensions, embedding, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,NOW())
                        ON CONFLICT (faq_id, language) DO UPDATE SET title=EXCLUDED.title, answer=EXCLUDED.answer,
                          searchable_text=EXCLUDED.searchable_text, content_hash=EXCLUDED.content_hash,
                          corpus_version=EXCLUDED.corpus_version,
                          embedding_model=EXCLUDED.embedding_model, embedding_dimensions=EXCLUDED.embedding_dimensions,
                          embedding=EXCLUDED.embedding, updated_at=NOW()
                    """, (document.id, document.language, document.title, document.answer, document.searchable_text,
                           document.content_hash, self._settings.faq_corpus_version, self._settings.voyage_model, self._settings.voyage_embedding_dimensions,
                           _vector(embedding)))
                keys = [f"{document.id}:{document.language}" for document in documents]
                await cursor.execute("DELETE FROM faq_documents WHERE (faq_id || ':' || language) <> ALL(%s)", (keys,))
                deleted = cursor.rowcount
            await connection.commit()
        return SyncResult(inserted=inserted, updated=updated, unchanged=unchanged, deleted=max(deleted, 0))

    async def _best(self, query: str) -> tuple[FaqEntry | None, float | None]:
        vectors = await self._embeddings.embed([query], input_type="query")
        if len(vectors) != 1 or len(vectors[0]) != self._settings.voyage_embedding_dimensions:
            raise RuntimeError("Embedding provider returned an unexpected query vector")
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute("""
                    SELECT faq_id, title, answer, 1 - (embedding <=> %s::vector) AS similarity
                    FROM faq_documents ORDER BY embedding <=> %s::vector LIMIT 1
                """, (_vector(vectors[0]), _vector(vectors[0])))
                row = await cursor.fetchone()
        if row is None:
            return None, None
        return FaqEntry(id=row["faq_id"], title=row["title"], answer=row["answer"]), float(row["similarity"])

    async def search(self, query: str) -> FaqEntry | None:
        if self._settings.faq_min_similarity is None:
            raise RuntimeError("FAQ_MIN_SIMILARITY must be calibrated before semantic FAQ retrieval is enabled")
        entry, score = await self._best(query)
        return entry if score is not None and score >= self._settings.faq_min_similarity else None

    async def best_similarity(self, query: str) -> tuple[FaqEntry | None, float | None]:
        """Internal calibration helper; scores never leave the indexing workflow."""
        return await self._best(query)

    async def best_similarities(self, queries: list[str]) -> list[tuple[FaqEntry | None, float | None]]:
        """Batch query embeddings for calibration to respect provider rate limits."""
        vectors = await self._embeddings.embed(queries, input_type="query")
        if len(vectors) != len(queries) or any(len(vector) != self._settings.voyage_embedding_dimensions for vector in vectors):
            raise RuntimeError("Embedding provider returned unexpected calibration vectors")
        results: list[tuple[FaqEntry | None, float | None]] = []
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                for vector in vectors:
                    await cursor.execute("""
                        SELECT faq_id, title, answer, 1 - (embedding <=> %s::vector) AS similarity
                        FROM faq_documents ORDER BY embedding <=> %s::vector LIMIT 1
                    """, (_vector(vector), _vector(vector)))
                    row = await cursor.fetchone()
                    results.append((None, None) if row is None else (FaqEntry(id=row["faq_id"], title=row["title"], answer=row["answer"]), float(row["similarity"])))
        return results
