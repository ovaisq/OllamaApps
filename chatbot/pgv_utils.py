import hashlib
import re
from typing import List, Tuple
import psycopg2
from psycopg2.extras import Json

def normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_hash(text: str) -> str:
    """Generate SHA256 hash of text chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def read_markdown(file_path: str) -> str:
    """Read markdown file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def embed_text(client, text: str, model: str = 'snowflake-arctic-embed', title: str = None) -> List[float]:
    """Generate embedding for text."""
    enriched = f"Title: {title}\nContent: {text}" if title else text
    enriched = normalize_text(enriched)
    return client.embeddings(model=model, prompt=enriched)['embedding']
