#!/usr/bin/env python3
"""
Incremental RAG Indexer for Markdown Files using Ollama + ChromaDB

This script indexes Markdown files into a ChromaDB vector store, allowing semantic search over their content.
It supports incremental indexing by hashing chunks and skipping already indexed ones.
"""

import ollama
import chromadb
import hashlib
import os
import logging
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== CONFIG =====
CHROMA_COLLECTION = "readme_rag"
CHROMA_DB_PATH = "./chroma_db"
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
CHUNK_SIZE = 800      # ~500-800 tokens recommended
CHUNK_OVERLAP = 100   # Overlap for context
# ==================

logging.getLogger("chromadb").setLevel(logging.ERROR)

client = ollama.Client(host=OLLAMA_HOST)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)


def normalize_text(text):
    """
    Clean text by replacing multiple whitespace with a single space.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Normalized text.
    """
    return re.sub(r'\s+', ' ', text).strip()


def chunk_hash(text):
    """
    Generate a stable SHA256 hash for the given text to use as an ID.

    Args:
        text (str): Text to hash.

    Returns:
        str: Hexadecimal hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_markdown(file_path):
    """
    Read a Markdown file and return its contents.

    Args:
        file_path (str): Path to the Markdown file.

    Returns:
        str: Content of the file as string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_chunks(text):
    """
    Split text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): Input text to chunk.

    Returns:
        list[str]: List of chunked texts.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def embed_text(chunk, title=None):
    """
    Generate embeddings for a text chunk using Ollama's snowflake-arctic-embed model.

    Args:
        chunk (str): The text chunk to embed.
        title (str, optional): Title of the document for context.

    Returns:
        list[float]: Embedding vector.
    """
    enriched = f"Title: {title}\nContent: {chunk}" if title else chunk
    enriched = normalize_text(enriched)
    return client.embeddings(model="snowflake-arctic-embed", prompt=enriched)["embedding"]


def index_markdown(file_path):
    """
    Index a Markdown file into ChromaDB.

    This function reads the markdown file, splits it into chunks, generates embeddings,
    and adds new chunks to the collection only if they are not already present.

    Args:
        file_path (str): Path to the Markdown file to index.
    """
    print(f"[INFO] Indexing file: {file_path}")
    text = read_markdown(file_path)
    chunks = create_chunks(text)

    # Fetch ALL existing IDs
    existing_data = collection.get(limit=None)
    existing_ids = set(existing_data.get('ids', []))

    new_chunks = []
    new_ids = []
    new_embeddings = []
    new_metadata = []

    for chunk in chunks:
        cid = chunk_hash(chunk)
        if cid not in existing_ids:
            embedding = embed_text(chunk, title=os.path.basename(file_path))
            new_chunks.append(normalize_text(chunk))
            new_ids.append(cid)
            new_embeddings.append(embedding)
            new_metadata.append({"source": os.path.basename(file_path)})

    # Deduplicate in case of unexpected collision within current batch
    unique_data = {}
    for i, cid in enumerate(new_ids):
        if cid not in unique_data:
            unique_data[cid] = (new_chunks[i], new_embeddings[i], new_metadata[i])

    if unique_data:
        final_ids = list(unique_data.keys())
        final_chunks = [v[0] for v in unique_data.values()]
        final_embeddings = [v[1] for v in unique_data.values()]
        final_metadata = [v[2] for v in unique_data.values()]

        print(f"[INFO] Adding {len(final_chunks)} new chunks...")
        try:
            collection.add(
                documents=final_chunks,
                ids=final_ids,
                embeddings=final_embeddings,
                metadatas=final_metadata
            )
            print("[INFO] Chunks added successfully.")
        except chromadb.errors.DuplicateIDError as e:
            print(f"[WARNING] Duplicate IDs still detected and skipped: {e}")
    else:
        print("[INFO] No new chunks to add. Index is up to date.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Incremental indexer for Markdown files")
    parser.add_argument("markdown_file", help="Path to the Markdown file to index")
    args = parser.parse_args()

    index_markdown(args.markdown_file)
