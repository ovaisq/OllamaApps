# Markdown Embedding Indexer with PostgreSQL & pgvector and ChromaDB + Gradio Front-end

This project provides a pipeline for indexing markdown content into a PostgreSQL database with [pgvector](https://github.com/pgvector/pgvector) for efficient semantic search. It includes:

* A PostgreSQL schema for storing text chunks and their vector embeddings.
* A Python script for parsing markdown files, generating embeddings, and inserting them into the database.
* A deployment script for setting up and running the system.

---

## Features

* **Markdown Chunking**: Splits markdown documents into manageable chunks.
* **Vector Embeddings**: Generates vector representations of chunks for semantic similarity.
* **pgvector Integration**: Stores embeddings in PostgreSQL with HNSW indexing for fast nearest-neighbor search.
* **Metadata Storage**: Keeps track of additional context via JSONB fields.
* **Deployment Script**: Simplifies database setup and indexing process.

## Configuration

Set environment variables:

```bash
export OLLAMA_HOST="http://localhost:11434"
```
