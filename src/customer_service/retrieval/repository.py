"""Postgres/pgvector repository for customer-safe FAQ search."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import unicodedata
from typing import Sequence

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from customer_service.config.settings import Settings
from customer_service.infrastructure.faqs import FaqEntry
from customer_service.retrieval.corpus import FaqDocument, FaqVariant
from customer_service.retrieval.embeddings import EmbeddingProvider
from customer_service.privacy.redaction import redact_text

LOGGER = logging.getLogger(__name__)


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def normalize_query(text: str) -> str:
    """Apply the measured light normalization used by the winning retrieval arm."""
    text = unicodedata.normalize("NFKC", text).replace("\u0640", "")
    text = "".join(
        char for char in text
        if not (0x0610 <= ord(char) <= 0x061A or 0x064B <= ord(char) <= 0x065F or ord(char) == 0x0670 or 0x06D6 <= ord(char) <= 0x06ED)
    )
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
    return re.sub(r"\s+", " ", text).strip()


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
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faq_entries (
                        faq_id TEXT NOT NULL, language TEXT NOT NULL, title TEXT NOT NULL,
                        answer TEXT NOT NULL, content_hash TEXT NOT NULL, corpus_version TEXT NOT NULL,
                        PRIMARY KEY (faq_id, language)
                    )
                """)
                await cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS faq_variants (
                        variant_key TEXT PRIMARY KEY, faq_id TEXT NOT NULL, text TEXT NOT NULL,
                        dialect TEXT NOT NULL, review_status TEXT NOT NULL, synthetic BOOLEAN NOT NULL,
                        content_hash TEXT NOT NULL, corpus_version TEXT NOT NULL,
                        embedding_model TEXT NOT NULL, embedding_dimensions INTEGER NOT NULL,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                await cursor.execute("""
                    SELECT pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS type_name
                    FROM pg_attribute AS attribute
                    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                    WHERE relation.relname = 'faq_variants' AND attribute.attname = 'embedding'
                """)
                variant_row = await cursor.fetchone()
                if variant_row is not None and variant_row[0] != f"vector({dimensions})":
                    await cursor.execute("DROP INDEX IF EXISTS faq_variants_embedding_hnsw")
                    await cursor.execute("TRUNCATE faq_variants")
                    await cursor.execute(f"ALTER TABLE faq_variants ALTER COLUMN embedding TYPE vector({dimensions})")
                await cursor.execute("""
                    CREATE INDEX IF NOT EXISTS faq_variants_embedding_hnsw
                    ON faq_variants USING hnsw (embedding vector_cosine_ops)
                """)
            await connection.commit()

    async def sync(self, documents: list[FaqDocument]) -> SyncResult:
        if not documents:
            raise RuntimeError("Refusing to sync an empty FAQ corpus")
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
                for document in documents:
                    await cursor.execute("""
                        INSERT INTO faq_entries (faq_id, language, title, answer, content_hash, corpus_version)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (faq_id, language) DO UPDATE SET title=EXCLUDED.title,
                          answer=EXCLUDED.answer, content_hash=EXCLUDED.content_hash,
                          corpus_version=EXCLUDED.corpus_version
                    """, (document.id, document.language, document.title, document.answer,
                           document.content_hash, self._settings.faq_corpus_version))
                keys = [f"{document.id}:{document.language}" for document in documents]
                await cursor.execute("DELETE FROM faq_documents WHERE (faq_id || ':' || language) <> ALL(%s)", (keys,))
                deleted = cursor.rowcount
                await cursor.execute("DELETE FROM faq_entries WHERE (faq_id || ':' || language) <> ALL(%s)", (keys,))
            await connection.commit()
        return SyncResult(inserted=inserted, updated=updated, unchanged=unchanged, deleted=max(deleted, 0))

    async def sync_variants(self, variants: list[FaqVariant]) -> SyncResult:
        """Synchronize question variants and the reserved no-answer sink."""
        if not variants:
            raise RuntimeError("Refusing to sync an empty FAQ variant corpus")
        dimensions = self._settings.voyage_embedding_dimensions
        desired = {f"{item.faq_id}:{item.content_hash}": item for item in variants}
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute("SELECT variant_key, content_hash, corpus_version, embedding_model, embedding_dimensions FROM faq_variants")
                existing = {row["variant_key"]: row for row in await cursor.fetchall()}
        changed: list[tuple[str, FaqVariant]] = []
        inserted = updated = unchanged = 0
        for key, item in desired.items():
            row = existing.get(key)
            if row is None:
                inserted += 1; changed.append((key, item))
            elif row["content_hash"] != item.content_hash or row["corpus_version"] != self._settings.faq_corpus_version or row["embedding_model"] != self._settings.voyage_model or row["embedding_dimensions"] != dimensions:
                updated += 1; changed.append((key, item))
            else:
                unchanged += 1
        vectors = await self._embeddings.embed([item.text for _, item in changed], input_type="document") if changed else []
        if len(vectors) != len(changed):
            raise RuntimeError("Embedding provider returned an unexpected number of FAQ variant vectors")
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                for (key, item), embedding in zip(changed, vectors, strict=True):
                    if len(embedding) != dimensions:
                        raise RuntimeError("Embedding provider returned an unexpected FAQ variant dimension")
                    await cursor.execute("""
                        INSERT INTO faq_variants (variant_key, faq_id, text, dialect, review_status, synthetic,
                          content_hash, corpus_version, embedding_model, embedding_dimensions, embedding, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,NOW())
                        ON CONFLICT (variant_key) DO UPDATE SET faq_id=EXCLUDED.faq_id, text=EXCLUDED.text,
                          dialect=EXCLUDED.dialect, review_status=EXCLUDED.review_status, synthetic=EXCLUDED.synthetic,
                          content_hash=EXCLUDED.content_hash, corpus_version=EXCLUDED.corpus_version,
                          embedding_model=EXCLUDED.embedding_model, embedding_dimensions=EXCLUDED.embedding_dimensions,
                          embedding=EXCLUDED.embedding, updated_at=NOW()
                    """, (key, item.faq_id, item.text, item.dialect, item.review_status, item.synthetic,
                           item.content_hash, self._settings.faq_corpus_version, self._settings.voyage_model,
                           dimensions, _vector(embedding)))
                await cursor.execute("DELETE FROM faq_variants WHERE variant_key <> ALL(%s)", (list(desired),))
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

    async def _variant_ranked(self, query: str) -> list[tuple[str, float]]:
        vector = (await self._embeddings.embed([normalize_query(query)], input_type="query"))[0]
        if len(vector) != self._settings.voyage_embedding_dimensions:
            raise RuntimeError("Embedding provider returned an unexpected query vector")
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute("""
                    SELECT faq_id, MAX(1 - (embedding <=> %s::vector)) AS similarity
                    FROM faq_variants
                    GROUP BY faq_id
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (_vector(vector), self._settings.faq_candidate_limit))
                rows = await cursor.fetchall()
        return [(row["faq_id"], float(row["similarity"])) for row in rows]

    async def _entry(self, faq_id: str, language: str) -> FaqEntry | None:
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    "SELECT faq_id, title, answer FROM faq_documents WHERE faq_id=%s AND language=%s",
                    (faq_id, language),
                )
                row = await cursor.fetchone()
        return None if row is None else FaqEntry(id=row["faq_id"], title=row["title"], answer=row["answer"])

    async def search_candidates(self, query: str, *, language: str = "en", limit: int = 3) -> list[FaqEntry]:
        """Return top real FAQ candidates without scores for the Gate C control."""
        ranked = [item for item in await self._variant_ranked(query) if item[0] != "__no_answer__"][:limit]
        entries = []
        for faq_id, _ in ranked:
            entry = await self._entry(faq_id, language)
            if entry is not None:
                entries.append(entry)
        return entries

    async def search(self, query: str, *, language: str = "en") -> FaqEntry | None:
        if self._settings.faq_log_queries:
            LOGGER.info("faq_query query=%s language=%s", redact_text(query), language)
        if self._settings.faq_min_similarity is None:
            raise RuntimeError("FAQ_MIN_SIMILARITY must be calibrated before semantic FAQ retrieval is enabled")
        # The variants table is authoritative once indexed. Legacy answer search remains
        # available for rollback and for tests that intentionally sync only faq_documents.
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT EXISTS (SELECT 1 FROM faq_variants LIMIT 1)")
                has_variants = bool((await cursor.fetchone())[0])
        if has_variants:
            ranked = await self._variant_ranked(query)
            if not ranked:
                return None
            top_id, top_score = ranked[0]
            if top_id == "__no_answer__" or top_score < self._settings.faq_min_similarity:
                return None
            if self._settings.faq_min_margin is not None and len(ranked) > 1 and top_score - ranked[1][1] < self._settings.faq_min_margin:
                return None
            return await self._entry(top_id, language)
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

    async def best_variant_similarities(self, queries: list[str], *, language: str = "en") -> list[tuple[FaqEntry | None, float | None]]:
        """Calibration helper for the variant index; scores never leave indexing."""
        results: list[tuple[FaqEntry | None, float | None]] = []
        for query in queries:
            ranked = await self._variant_ranked(query)
            if not ranked:
                results.append((None, None))
                continue
            faq_id, score = ranked[0]
            if faq_id == "__no_answer__":
                results.append((None, score))
            else:
                results.append((await self._entry(faq_id, language), score))
        return results
