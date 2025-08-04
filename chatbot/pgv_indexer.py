#!/usr/bin/env python3

import ollama
import os
import sys
import argparse
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json

from pgv_config import DB_CONFIG, INDEXER_CONFIG, OLLAMA_CONFIG
from pgv_utils import (
    read_markdown, create_chunks, embed_text,
    normalize_text, chunk_hash
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

def index_markdown_file(file_path: str, db_connection) -> int:
    """Index a markdown file into database."""
    logging.info(f"Indexing file: {file_path}")
    
    # Read and split content
    text = read_markdown(file_path)
    chunks = create_chunks(
        text,
        INDEXER_CONFIG['chunk_size'],
        INDEXER_CONFIG['chunk_overlap']
    )
    
    # Connect to Ollama
    client = ollama.Client(host=OLLAMA_CONFIG['host'])
    
    # Get existing chunks
    with db_connection.cursor() as cursor:
        existing_chunks = get_existing_chunks(cursor)
    
    # Prepare new chunks
    new_chunks = []
    for chunk in chunks:
        norm_chunk = normalize_text(chunk)
        if norm_chunk not in existing_chunks:
            embedding = embed_text(
                client, 
                chunk, 
                OLLAMA_CONFIG['embedding_model'], 
                os.path.basename(file_path)
            )
            new_chunks.append((norm_chunk, embedding))
    
    # Insert new chunks
    with db_connection.cursor() as cursor:
        if new_chunks:
            insert_query = """
                INSERT INTO markdown_chunks (chunk, embedding, metadata)
                VALUES (%s, %s, %s)
            """
            values = [
                (chunk, embedding, Json({"source": os.path.basename(file_path)}))
                for chunk, embedding in new_chunks
            ]
            cursor.executemany(insert_query, values)
            db_connection.commit()
            logging.info(f"Inserted {len(new_chunks)} new chunks")
            return len(new_chunks)
        else:
            logging.info("No new chunks to insert")
            return 0

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
