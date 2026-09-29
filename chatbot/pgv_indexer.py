#!/usr/bin/env python3

import ollama
import os
import argparse
import logging
import psycopg2
from psycopg2.extras import Json

from pgv_config import DB_CONFIG, INDEXER_CONFIG, OLLAMA_CONFIG
from pgv_utils import (
    read_markdown, create_chunks, embed_text, normalize_text
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

def get_db_connection() -> psycopg2.extensions.connection:
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)

def get_existing_chunks(cursor) -> set:
    """Get existing chunks from database."""
    cursor.execute("SELECT chunk FROM markdown_chunks")
    return {normalize_text(row[0]) for row in cursor.fetchall()}

def index_text(text: str, source: str, db_connection) -> int:
    """Chunk, embed, and insert new (deduped) content into the database.

    Shared by the markdown-file indexer and the Google Drive indexer so both
    sources dedupe/embed/insert identically.
    """
    chunks = create_chunks(
        text,
        INDEXER_CONFIG['chunk_size'],
        INDEXER_CONFIG['chunk_overlap']
    )

    client = ollama.Client(host=OLLAMA_CONFIG['host'])

    with db_connection.cursor() as cursor:
        existing_chunks = get_existing_chunks(cursor)

    new_chunks = []
    for chunk in chunks:
        norm_chunk = normalize_text(chunk)
        if norm_chunk not in existing_chunks:
            embedding = embed_text(client, chunk, OLLAMA_CONFIG['embedding_model'], source)
            new_chunks.append((norm_chunk, embedding))

    with db_connection.cursor() as cursor:
        if new_chunks:
            insert_query = """
                INSERT INTO markdown_chunks (chunk, embedding, metadata)
                VALUES (%s, %s, %s)
            """
            values = [
                (chunk, embedding, Json({"source": source}))
                for chunk, embedding in new_chunks
            ]
            cursor.executemany(insert_query, values)
            db_connection.commit()
            logging.info(f"Inserted {len(new_chunks)} new chunks from {source}")
            return len(new_chunks)
        else:
            logging.info(f"No new chunks to insert from {source}")
            return 0


def index_markdown_file(file_path: str, db_connection) -> int:
    """Index a markdown file into database."""
    logging.info(f"Indexing file: {file_path}")
    text = read_markdown(file_path)
    return index_text(text, os.path.basename(file_path), db_connection)

def main():
    """Main function for indexer."""
    setup_logging()

    parser = argparse.ArgumentParser(description='Index markdown files into database')
    parser.add_argument('files', nargs='+', help='Markdown files to index')
    args = parser.parse_args()

    db_connection = get_db_connection()

    try:
        total_inserted = 0
        for file_path in args.files:
            inserted = index_markdown_file(file_path, db_connection)
            total_inserted += inserted

        logging.info(f"Successfully indexed {total_inserted} new chunks")
    finally:
        db_connection.close()

if __name__ == "__main__":
    main()
