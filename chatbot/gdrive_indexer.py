#!/usr/bin/env python3
"""Index Google Drive content (Google Docs, .md/.txt, PDFs) into the
markdown chatbot's vector store.

One-time setup: with the chatbot app running (pgv_chatty.py or
chromadb_chatty.py, which mount the routes in gdrive_oauth_routes.py), visit
GET /connect-drive in a browser to grant Drive read access. That persists a
refresh token to gdrive_config.DRIVE_CONFIG['token_store_path']. Then run this
indexer as a cron/manual job, same as pgv_indexer.py/chromadb_indexer.py.
"""
import argparse
import logging

from gdrive_config import DRIVE_CONFIG, GOOGLE_CONFIG
from gdrive_auth import refresh_access_token
from gdrive_client import fetch_file_text, list_files
from gdrive_token_store import load_refresh_token

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_access_token() -> str:
    refresh_token = load_refresh_token(DRIVE_CONFIG["token_store_path"])
    if not refresh_token:
        raise RuntimeError(
            "No Google Drive refresh token found. Run gdrive_authorize.py first."
        )
    return refresh_access_token(
        GOOGLE_CONFIG["client_id"], GOOGLE_CONFIG["client_secret"], refresh_token
    )


def run_pgvector_backend(access_token: str) -> int:
    from pgv_config import DB_CONFIG
    from pgv_indexer import index_text
    import psycopg2

    conn = psycopg2.connect(**DB_CONFIG)
    total = 0
    try:
        for f in list_files(access_token, DRIVE_CONFIG["folder_id"]):
            text = fetch_file_text(access_token, f["id"], f["mimeType"])
            if not text or not text.strip():
                continue
            total += index_text(text, f["name"], conn)
    finally:
        conn.close()
    return total


def run_chromadb_backend(access_token: str) -> int:
    import chromadb
    import ollama
    from chroma_config import CHROMA_CONFIG, OLLAMA_CONFIG
    from chromadb_indexer import index_text

    client = ollama.Client(host=OLLAMA_CONFIG["host"], timeout=OLLAMA_CONFIG["timeout"])
    chroma_client = chromadb.PersistentClient(path=CHROMA_CONFIG["db_path"])
    collection = chroma_client.get_or_create_collection(name=CHROMA_CONFIG["collection"])

    total = 0
    for f in list_files(access_token, DRIVE_CONFIG["folder_id"]):
        text = fetch_file_text(access_token, f["id"], f["mimeType"])
        if not text or not text.strip():
            continue
        total += index_text(text, f["name"], collection, client)
    return total


def main():
    parser = argparse.ArgumentParser(description="Index Google Drive content")
    parser.add_argument(
        "--backend", choices=["pgvector", "chromadb"], default="pgvector",
        help="Which vector store to index into (default: pgvector)",
    )
    args = parser.parse_args()

    access_token = get_access_token()
    runner = run_pgvector_backend if args.backend == "pgvector" else run_chromadb_backend
    total_inserted = runner(access_token)
    logger.info("Successfully indexed %d new chunks from Google Drive", total_inserted)


if __name__ == "__main__":
    main()
