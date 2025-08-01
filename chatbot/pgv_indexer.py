#!/usr/bin/env python3
import ollama
import hashlib
import os
import logging
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from psycopg2.extras import Json

# ===== CONFIG =====
OLLAMA_HOST = "http://"
CHUNK_SIZE = 800      # ~500-800 tokens recommended
CHUNK_OVERLAP = 100   # Overlap for context

# PostgreSQL connection settings
DB_HOST = ""
DB_PORT = 5432
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""

# ==================

logging.getLogger("psycopg2").setLevel(logging.ERROR)

client = ollama.Client(host=OLLAMA_HOST)
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# --- Helpers ---
def normalize_text(text):
    """Clean text for consistent embeddings."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_hash(text):
    """Stable SHA256 hash for chunk IDs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def embed_text(chunk, title=None):
    """Add context before embedding to improve semantic quality."""
    enriched = f"Title: {title}\nContent: {chunk}" if title else chunk
    enriched = normalize_text(enriched)
    return client.embeddings(model="snowflake-arctic-embed", prompt=enriched)["embedding"]

# --- Main Indexer ---
def index_markdown(file_path):
    print(f"[INFO] Indexing file: {file_path}")
    text = read_markdown(file_path)
    chunks = create_chunks(text)

    # Fetch ALL existing IDs
    cursor = conn.cursor()
    cursor.execute("SELECT id, chunk FROM markdown_chunks")
    existing_data = cursor.fetchall()

    existing_ids = set(chunk_hash(row[1]) for row in existing_data)

    new_chunks = []
    new_embeddings = []
    new_metadata = []

    for chunk in chunks:
        cid = chunk_hash(chunk)
        if cid not in existing_ids:
            embedding = embed_text(chunk, title=os.path.basename(file_path))
            new_chunks.append(normalize_text(chunk))
            new_embeddings.append(embedding)
            new_metadata.append(Json({"source": os.path.basename(file_path)}))

    # Deduplicate in case of unexpected collision within current batch
    unique_data = {}
    for i, chunk in enumerate(new_chunks):
        cid = chunk_hash(chunk)
        if cid not in unique_data:
            unique_data[cid] = (chunk, new_embeddings[i], new_metadata[i])

    if unique_data:
        final_chunks = [v[0] for v in unique_data.values()]
        final_embeddings = [v[1] for v in unique_data.values()]
        final_metadata = [v[2] for v in unique_data.values()]

        print(f"[INFO] Adding {len(final_chunks)} new chunks...")

        try:
            cursor.executemany(
                """
                INSERT INTO markdown_chunks (chunk, embedding, metadata)
                VALUES (%s, %s, %s);
                """,
                zip(final_chunks, final_embeddings, final_metadata)
            )
            conn.commit()
            print("[INFO] Chunks added successfully.")
        except psycopg2.IntegrityError as e:
            print(f"[WARNING] Duplicate IDs still detected and skipped: {e}")
    else:
        print("[INFO] No new chunks to add. Index is up to date.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Incremental indexer for Markdown files")
    parser.add_argument("markdown_file", help="Path to the Markdown file to index")
    args = parser.parse_args()

    index_markdown(args.markdown_file)
