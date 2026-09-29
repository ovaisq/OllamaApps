"""Shared helpers used by both the pgvector and ChromaDB chatbot variants."""
import hashlib
import io
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def build_ollama_client(host: str, timeout: float) -> Any:
    """Build an ollama.Client with a short connect timeout (fail fast if the
    host is unreachable) but a long read timeout (`timeout`) -- chat
    streaming on a large/cold-loaded model can take far longer than a
    reasonable connect timeout without actually being stuck.
    """
    import httpx
    import ollama

    return ollama.Client(
        host=host,
        timeout=httpx.Timeout(connect=10.0, read=timeout, write=timeout, pool=timeout),
    )


def load_env_file(env_path: Path = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (existing env
    vars take precedence). Mirrors the no-dependency loader already used in
    livetranscription/server.py.
    """
    env_path = env_path or Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_hash(text: str) -> str:
    """Generate SHA256 hash of a text chunk, used as a stable dedupe key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_markdown(file_path: str) -> str:
    """Read markdown file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def embed_text(client: Any, text: str, model: str, title: str = None) -> List[float]:
    """Generate an embedding for a chunk of text (optionally titled) via Ollama."""
    enriched = f"Title: {title}\nContent: {text}" if title else text
    enriched = normalize_text(enriched)
    return client.embeddings(model=model, prompt=enriched)["embedding"]


def format_context_chunks(chunks: List[Tuple[str, Optional[str]]]) -> str:
    """Tag each retrieved chunk with its source document. Without this the
    model gets anonymous text blobs and can't say "Engineering Reports
    covers X" -- it can only pattern-match on wording, which reads as
    pedantic ("no specific mention of X") even when relevant content was
    actually retrieved.
    """
    if not chunks:
        return "(no relevant documents found)"
    return "\n\n".join(f"[Source: {source or 'unknown source'}]\n{text}" for text, source in chunks)


CHAT_SYSTEM_PROMPT = """You are a helpful research assistant with access to the user's indexed documents.

How to answer:
- If the question is a general or discovery-style query (e.g. "what do you have on X", "X reports?"), name the source document(s) you found and briefly summarize what each one covers.
- If it's a specific factual question, answer it directly using the context, and say which document it came from.
- Base your answer only on the provided context. If none of it is actually relevant to the question, say so plainly instead of guessing or claiming there's "no mention" when you've only seen a handful of excerpts, not the full document.
- Do not answer general-knowledge questions from your own training data as if they came from the context. If the context has nothing relevant, say you found nothing relevant in the indexed documents -- don't substitute your own (possibly wrong) general knowledge as if it were grounded.
"""


def build_chat_messages(
    context_chunks: List[Tuple[str, Optional[str]]], conversation_context: str, query: str
) -> List[Dict[str, str]]:
    """Build the RAG chat messages, split into a system role (instructions)
    and a user role (context + question). Bundling instructions into a
    single user-role message gives chat-tuned models much weaker
    instruction-following than a proper system message.
    """
    context_text = format_context_chunks(context_chunks)
    user_content = f"""Context (grouped by source document):
{context_text}

Previous conversation:
{conversation_context}

Question: {query}
Answer:
"""
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_pdf_text(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes, or None if extraction fails/empty."""
    if not pdf_bytes:
        return None
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        logger.exception("Failed to extract text from PDF")
        return None


def extract_xlsx_text(xlsx_bytes: bytes) -> Optional[str]:
    """Render every sheet's cell values as CSV-ish text, or None on failure.

    Only cell values are captured (no formulas/formatting) -- enough for
    embedding/retrieval, not a faithful spreadsheet reproduction. A Google
    Sheet with multiple tabs should go through Sheets export instead (Drive's
    /export only returns the first sheet as CSV); this path is for real
    .xlsx files, which openpyxl can read in full.
    """
    if not xlsx_bytes:
        return None
    from openpyxl import load_workbook

    try:
        workbook = load_workbook(io.BytesIO(xlsx_bytes), data_only=True, read_only=True)
        parts = []
        for sheet in workbook.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                # Trim trailing empty cells (openpyxl pads rows to the
                # sheet's widest row) so embeddings aren't drowned in commas.
                trimmed = list(row)
                while trimmed and trimmed[-1] is None:
                    trimmed.pop()
                if trimmed:
                    parts.append(",".join("" if cell is None else str(cell) for cell in trimmed))
        return "\n".join(parts)
    except Exception:
        logger.exception("Failed to extract text from .xlsx")
        return None


def extract_text_from_upload(file_path: str) -> Optional[str]:
    """Extract plain text from a locally-uploaded .md/.txt/.pdf/.xlsx file,
    or None if the extension isn't supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".md", ".txt"):
        return read_markdown(file_path)
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            return extract_pdf_text(f.read())
    if ext == ".xlsx":
        with open(file_path, "rb") as f:
            return extract_xlsx_text(f.read())
    return None


def validate_message(message: str, max_length: int) -> str:
    """Validate a user-supplied chat message.

    Raises ValueError if the message is empty or exceeds max_length, so
    callers can reject oversized input before it reaches the LLM/embedding
    calls (cheap guard against accidental or malicious resource exhaustion).
    """
    if message is None or not message.strip():
        raise ValueError("Message must not be empty.")
    if len(message) > max_length:
        raise ValueError(f"Message exceeds the {max_length} character limit.")
    return message


def with_retries(
    fn: Callable[[], Any],
    attempts: int = 3,
    backoff_seconds: float = 0.5,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """Call fn() with exponential backoff retries on the given exception types."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            last_exc = e
            if attempt == attempts:
                break
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt, attempts, e, sleep_for,
            )
            time.sleep(sleep_for)
    raise last_exc


def safe_error_message(exc: Exception, log: logging.Logger = None) -> str:
    """Log the full exception server-side and return a generic, correlated
    message safe to show to end users (avoids leaking internals/stack traces).
    """
    error_id = uuid.uuid4().hex[:8]
    (log or logger).error("error_id=%s request failed: %r", error_id, exc, exc_info=exc)
    return f"Sorry, something went wrong processing your request (error id: {error_id})."
