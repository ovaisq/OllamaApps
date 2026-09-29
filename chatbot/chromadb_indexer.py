#!/usr/bin/env python3
"""Incremental RAG Indexer for Markdown Files using Ollama + ChromaDB.

This script indexes Markdown files into a ChromaDB vector store, allowing
semantic search over their content. It supports incremental indexing by
hashing chunks and skipping already indexed ones.
"""
import logging
import os

import chromadb
import ollama

from chroma_config import CHROMA_CONFIG, INDEXER_CONFIG, OLLAMA_CONFIG
from rag_common import chunk_hash, create_chunks, embed_text, normalize_text, read_markdown

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def index_text(text: str, source: str, collection, client) -> int:
    """Chunk, embed, and add new (deduped) content to the ChromaDB collection.

    Shared by the markdown-file indexer and the Google Drive indexer so both
    sources dedupe/embed/insert identically.
    """
    chunks = create_chunks(text, INDEXER_CONFIG["chunk_size"], INDEXER_CONFIG["chunk_overlap"])

    existing_ids = set(collection.get(limit=None).get("ids", []))

    unique_data = {}
    for chunk in chunks:
        cid = chunk_hash(chunk)
        if cid not in existing_ids and cid not in unique_data:
            embedding = embed_text(client, chunk, OLLAMA_CONFIG["embedding_model"], title=source)
            unique_data[cid] = (normalize_text(chunk), embedding, {"source": source})

    if not unique_data:
        logger.info("No new chunks to add from %s. Index is up to date.", source)
        return 0

    logger.info("Adding %d new chunks from %s...", len(unique_data), source)
    try:
        collection.add(
            documents=[v[0] for v in unique_data.values()],
            ids=list(unique_data.keys()),
            embeddings=[v[1] for v in unique_data.values()],
            metadatas=[v[2] for v in unique_data.values()],
        )
        logger.info("Chunks added successfully.")
    except chromadb.errors.DuplicateIDError:
        logger.warning("Duplicate IDs detected and skipped.")
    return len(unique_data)


def index_markdown(file_path: str, collection, client) -> int:
    """Index a Markdown file into ChromaDB, skipping chunks already present."""
    logger.info("Indexing file: %s", file_path)
    text = read_markdown(file_path)
    return index_text(text, os.path.basename(file_path), collection, client)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Incremental indexer for Markdown files")
    parser.add_argument("markdown_file", help="Path to the Markdown file to index")
    args = parser.parse_args()

    client = ollama.Client(host=OLLAMA_CONFIG["host"], timeout=OLLAMA_CONFIG["timeout"])
    chroma_client = chromadb.PersistentClient(path=CHROMA_CONFIG["db_path"])
    collection = chroma_client.get_or_create_collection(name=CHROMA_CONFIG["collection"])

    index_markdown(args.markdown_file, collection, client)


if __name__ == "__main__":
    main()
