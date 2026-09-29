"""Schema self-migration for the pgvector backend.

pgvector.sql only runs when Postgres first initializes an empty data
volume, so it can never migrate an already-running deployment. These
functions run idempotently at app startup instead.
"""
import logging

logger = logging.getLogger(__name__)

HNSW_INDEX = "markdown_chunks_embedding_hnsw_index"


def ensure_markdown_chunks_schema(conn, dim: int, embedding_model: str) -> None:
    """Create markdown_chunks if missing, and resize its embedding column
    to `dim` while the table is still empty.

    This is what lets a from-scratch deployment work without pgvector.sql
    and an existing deployment survive an EMBEDDING_MODEL/EMBEDDING_DIM
    change (e.g. nomic-embed-text's 768 dims -> qwen3-embedding:0.6b's
    1024) without every insert/query failing on a dimension mismatch.

    A *non-empty* table with the wrong dimension is an error, not something
    to fix silently: its rows were embedded by a different model and cannot
    be mixed with `embedding_model` embeddings, so they must be re-indexed.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                chunk TEXT NOT NULL,
                embedding vector(%s) NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            (dim,),
        )
        cursor.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'markdown_chunks'::regclass "
            "AND attname = 'embedding'"
        )
        typmod = cursor.fetchone()[0]
        # pgvector stores dim + 4 (the varlena header) in atttypmod;
        # -1 means the unbounded `vector` type.
        existing_dim = typmod - 4 if typmod > 0 else None
        if existing_dim == dim:
            _create_hnsw_index(cursor)
            conn.commit()
            return
        cursor.execute("SELECT EXISTS (SELECT 1 FROM markdown_chunks)")
        if cursor.fetchone()[0]:
            conn.rollback()
            raise RuntimeError(
                f"markdown_chunks.embedding has "
                f"{existing_dim if existing_dim else 'no'} dimensions but "
                f"EMBEDDING_DIM is {dim}, and the table is not empty. Its rows "
                f"were embedded by a different model and cannot be mixed with "
                f"'{embedding_model}' embeddings. TRUNCATE markdown_chunks "
                f"and re-index (restart the app, then re-run the Drive/"
                f"markdown indexer), or switch EMBEDDING_MODEL back."
            )
        # Still empty, so nothing of value is lost: drop the hnsw index
        # (ALTER TYPE can't run under one), resize, recreate.
        cursor.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")
        cursor.execute(
            "ALTER TABLE markdown_chunks ALTER COLUMN embedding TYPE vector(%s)",
            (dim,),
        )
        _create_hnsw_index(cursor)
        conn.commit()
        logger.info(
            "Resized markdown_chunks.embedding to %d dims for '%s'",
            dim, embedding_model,
        )


def _create_hnsw_index(cursor) -> None:
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} "
        "ON markdown_chunks USING hnsw (embedding vector_l2_ops)"
    )
