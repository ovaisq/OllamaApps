import re
from unittest.mock import MagicMock

import pytest

import pgv_schema

# pgvector stores dim + 4 (the varlena header) in pg_attribute.atttypmod.
def typmod_for(dim):
    return (dim + 4,)


@pytest.fixture
def conn():
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def executed_sql(cursor):
    return [call.args[0] for call in cursor.execute.call_args_list]


def test_creates_table_with_configured_dimension_and_commits(conn):
    """A missing table must be created at EMBEDDING_DIM (not a hardcoded
    old value like 768), and a matching dimension must not trigger a
    resize."""
    conn, cursor = conn
    cursor.fetchone.return_value = typmod_for(1024)

    pgv_schema.ensure_markdown_chunks_schema(conn, 1024, "qwen3-embedding:0.6b")

    sql = executed_sql(cursor)
    assert any("CREATE TABLE IF NOT EXISTS markdown_chunks" in s for s in sql)
    assert cursor.execute.call_args_list[0].args[1] == (1024,)
    assert any("CREATE INDEX IF NOT EXISTS" in s for s in sql)
    assert not any("ALTER TABLE" in s for s in sql)
    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()


def test_resizes_empty_column_when_dimension_changes(conn):
    """An empty 768-dim table (left over from nomic-embed-text) must be
    resized to the new model's dimension: drop the hnsw index (ALTER TYPE
    can't run under one), ALTER, recreate the index."""
    conn, cursor = conn
    cursor.fetchone.side_effect = [typmod_for(768), (False,)]

    pgv_schema.ensure_markdown_chunks_schema(conn, 1024, "qwen3-embedding:0.6b")

    sql = executed_sql(cursor)
    assert any("DROP INDEX IF EXISTS" in s for s in sql)
    alter_calls = [c for c in cursor.execute.call_args_list if "ALTER TABLE" in c.args[0]]
    assert len(alter_calls) == 1
    assert alter_calls[0].args == ("ALTER TABLE markdown_chunks ALTER COLUMN embedding TYPE vector(%s)", (1024,))
    assert any("CREATE INDEX IF NOT EXISTS" in s for s in sql)
    conn.commit.assert_called_once()


def test_raises_instead_of_resizing_a_populated_table(conn):
    """A non-empty table's rows were embedded by a different model; they
    cannot be mixed with new-model embeddings, so a resize must be refused
    with actionable guidance, not silently corrupt retrieval."""
    conn, cursor = conn
    cursor.fetchone.side_effect = [typmod_for(768), (True,)]

    with pytest.raises(RuntimeError) as exc_info:
        pgv_schema.ensure_markdown_chunks_schema(conn, 1024, "qwen3-embedding:0.6b")

    assert "re-index" in str(exc_info.value) or "re-run" in str(exc_info.value)
    assert "768" in str(exc_info.value)
    assert not any("ALTER TABLE" in s for s in executed_sql(cursor))
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_resizes_unbounded_empty_column(conn):
    """A plain `vector` column (atttypmod -1) accepts any dimension, which
    would make distance queries between mixed-dim rows fail at runtime --
    it must be resized like a bounded mismatch."""
    conn, cursor = conn
    cursor.fetchone.side_effect = [(-1,), (False,)]

    pgv_schema.ensure_markdown_chunks_schema(conn, 1024, "qwen3-embedding:0.6b")

    assert any("ALTER TABLE" in s for s in executed_sql(cursor))
    conn.commit.assert_called_once()


def test_pgvector_sql_declared_dimension_matches_config_default():
    """pgvector.sql (fresh-install DDL) and pgv_config's EMBEDDING_DIM
    default must agree -- they live in different files and nothing at
    runtime would catch them drifting apart."""
    sql = open("pgvector.sql").read()
    template = open("pgv_config.py.template").read()

    sql_dims = re.findall(r"embedding vector\((\d+)\)", sql)
    config_dim = re.search(r"EMBEDDING_DIM', (\d+)\)", template)

    assert sql_dims, "pgvector.sql no longer declares embedding vector(N)"
    assert config_dim, "pgv_config.py.template no longer sets EMBEDDING_DIM"
    assert set(sql_dims) == {config_dim.group(1)}
