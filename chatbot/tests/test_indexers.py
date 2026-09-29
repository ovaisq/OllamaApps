from unittest.mock import MagicMock, patch

import pgv_indexer
import chromadb_indexer


def test_pgv_index_markdown_file_skips_existing_chunks(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("hello world, this is a short markdown file.")

    fake_conn = MagicMock()
    select_cursor = MagicMock()
    select_cursor.__enter__.return_value = select_cursor
    select_cursor.fetchall.return_value = [("hello world, this is a short markdown file.",)]

    insert_cursor = MagicMock()
    insert_cursor.__enter__.return_value = insert_cursor

    fake_conn.cursor.side_effect = [select_cursor, insert_cursor]

    with patch("pgv_indexer.ollama.Client") as mock_client_cls, \
         patch("pgv_indexer.create_chunks", return_value=["hello world, this is a short markdown file."]):
        mock_client_cls.return_value.embeddings.return_value = {"embedding": [0.1]}
        inserted = pgv_indexer.index_markdown_file(str(md_file), fake_conn)

    assert inserted == 0
    insert_cursor.executemany.assert_not_called()


def test_chromadb_index_markdown_skips_existing_ids(tmp_path):
    md_file = tmp_path / "doc.md"
    text = "hello world, this is a short markdown file."
    md_file.write_text(text)

    from rag_common import chunk_hash, normalize_text

    collection = MagicMock()
    collection.get.return_value = {"ids": [chunk_hash(normalize_text(text))]}

    client = MagicMock()

    with patch("chromadb_indexer.create_chunks", return_value=[text]):
        inserted = chromadb_indexer.index_markdown(str(md_file), collection, client)

    assert inserted == 0
    collection.add.assert_not_called()


def test_chromadb_index_markdown_adds_new_chunks(tmp_path):
    md_file = tmp_path / "doc.md"
    text = "brand new content never indexed before."
    md_file.write_text(text)

    collection = MagicMock()
    collection.get.return_value = {"ids": []}

    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.5]}

    with patch("chromadb_indexer.create_chunks", return_value=[text]):
        inserted = chromadb_indexer.index_markdown(str(md_file), collection, client)

    assert inserted == 1
    collection.add.assert_called_once()
