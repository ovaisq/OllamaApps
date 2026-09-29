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

# Shown immediately on submit, before the embedding call/model warm-up/first
# token -- without it the chat window shows nothing at all for however long
# retrieval + a cold model load takes, which reads as broken, not just slow.
TYPING_INDICATOR_HTML = (
    '<div class="typing-indicator"><span></span><span></span><span></span></div>'
)

# Paired with TYPING_INDICATOR_HTML: passed to gr.mount_gradio_app(css=...).
TYPING_INDICATOR_CSS = """
.typing-indicator { display: inline-flex; gap: 4px; padding: 4px 0; }
.typing-indicator span {
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor; opacity: 0.4;
    animation: chatty-typing-bounce 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes chatty-typing-bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}
"""


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


def format_source_label(metadata: Optional[Dict[str, Any]]) -> str:
    """Render a chunk's metadata (source filename, Drive owner/sharer) as a
    single citation label, e.g. "Engineering Reports | owner: Jane Doe |
    shared by: John Smith". Falls back gracefully for file-based content
    that only ever has a "source" key.
    """
    metadata = metadata or {}
    parts = [metadata.get("source") or "unknown source"]
    owner = metadata.get("owner")
    if owner:
        parts.append(f"owner: {owner}")
    shared_by = metadata.get("shared_by")
    if shared_by and shared_by != owner:
        parts.append(f"shared by: {shared_by}")
    return " | ".join(parts)


def format_context_chunks(chunks: List[Tuple[str, Optional[Dict[str, Any]]]]) -> str:
    """Tag each retrieved chunk with its source document (and Drive
    ownership/sharing info, when known). Without this the model gets
    anonymous text blobs and can't say "Engineering Reports covers X" -- it
    can only pattern-match on wording, which reads as pedantic ("no
    specific mention of X") even when relevant content was actually
    retrieved.
    """
    if not chunks:
        return "(no relevant documents found)"
    return "\n\n".join(f"[{format_source_label(meta)}]\n{text}" for text, meta in chunks)


# Maps a "category" keyword to the mimeTypes/extensions that belong to it, so
# "list all spreadsheets" can filter on stored metadata rather than needing a
# semantic match (embedding similarity has no notion of "this is a
# spreadsheet" -- that's a metadata fact, not something in the chunk text).
CATEGORY_MIME_TYPES: Dict[str, set] = {
    "pdf": {"application/pdf"},
    "spreadsheet": {
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "doc": {
        "application/vnd.google-apps.document",
        "application/pdf",
        "text/plain",
        "text/markdown",
    },
}

_CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "pdf": ("pdf", "pdfs"),
    "spreadsheet": ("spreadsheet", "spreadsheets", "sheet", "sheets", "xlsx", "excel"),
    # Deliberately excludes "document"/"documents" -- those are used
    # generically to mean "files" ("show me all documents shared by Jen"),
    # not specifically the Docs category, and would wrongly narrow a
    # person-based catalog query down to only Docs-category files.
    "doc": ("doc", "docs", "docx", "txt"),
}

_CATEGORY_KEYWORD_RE = {
    cat: re.compile(r"\b(?:" + "|".join(kws) + r")\b", re.IGNORECASE)
    for cat, kws in _CATEGORY_KEYWORDS.items()
}

_LISTING_VERBS = ("list", "show me", "show", "what", "which", "find me", "find", "any")
_PERSON_RE = re.compile(r"(?:shared by|owned by|from)\s+([A-Za-z][\w'-]*)", re.IGNORECASE)


def detect_catalog_intent(query: str) -> Optional[Dict[str, Any]]:
    """Detect "show me all documents shared by Jen" / "list all PDFs" style
    enumeration questions, which a vector similarity search can't answer
    reliably (that's a metadata filter, not a semantic content match).
    Returns {"person": str|None, "category": str|None, "shared_with_me": bool}
    or None when the query looks like a normal content question.
    """
    q = query.lower()

    person_match = _PERSON_RE.search(query)
    person = person_match.group(1) if person_match else None

    category = None
    for cat, pattern in _CATEGORY_KEYWORD_RE.items():
        if pattern.search(q):
            category = cat
            break

    shared_with_me = "shared with me" in q
    is_listing = any(v in q for v in _LISTING_VERBS)

    if person or shared_with_me or (category and is_listing):
        return {"person": person, "category": category, "shared_with_me": shared_with_me}
    return None


def format_document_catalog(docs: List[Dict[str, Any]]) -> str:
    """Format a list of {"source", "owner", "shared_by", "mime_type"} dicts
    (from a metadata catalog query, not a content search) into text the
    model can turn into a natural-language list.
    """
    if not docs:
        return "(no matching documents found)"
    lines = []
    for doc in docs:
        label = format_source_label(doc)
        lines.append(f"- {label}")
    return "\n".join(lines)


CHAT_SYSTEM_PROMPT = """You are a helpful research assistant with access to the user's indexed documents.

How to answer:
- If the question is a general or discovery-style query (e.g. "what do you have on X", "X reports?"), name the source document(s) you found and briefly summarize what each one covers.
- If it's a specific factual question, answer it directly using the context, and say which document it came from.
- Base your answer only on the provided context. If none of it is actually relevant to the question, say so plainly instead of guessing or claiming there's "no mention" when you've only seen a handful of excerpts, not the full document.
- Do not answer general-knowledge questions from your own training data as if they came from the context. If the context has nothing relevant, say you found nothing relevant in the indexed documents -- don't substitute your own (possibly wrong) general knowledge as if it were grounded.
"""


def build_chat_messages(
    context_chunks: List[Tuple[str, Optional[Dict[str, Any]]]], conversation_context: str, query: str
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


def build_catalog_chat_messages(
    docs: List[Dict[str, Any]], conversation_context: str, query: str
) -> List[Dict[str, str]]:
    """Build chat messages for a metadata catalog query ("show me all
    documents shared by Jen", "list all PDFs") -- a document listing, not a
    content search, so it's framed differently from build_chat_messages but
    shares the same grounding system prompt.
    """
    catalog_text = format_document_catalog(docs)
    user_content = f"""Matching documents found (from index metadata, not a content search):
{catalog_text}

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
