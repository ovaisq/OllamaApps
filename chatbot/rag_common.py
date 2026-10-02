"""Shared helpers used by both the pgvector and ChromaDB chatbot variants."""
import hashlib
import io
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# httpx logs one INFO line per HTTP request, which floods the logs during a
# full Drive re-index (one embed per chunk) and drowns the app's own
# progress/error lines. Failures still surface: httpx errors raise into our
# own log calls. Set before any app imports rag_common.
logging.getLogger("httpx").setLevel(logging.WARNING)


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


class DriveSyncGate:
    """Prevents overlapping Google Drive syncs.

    One sync already parallelizes across a worker pool, so a second trigger
    (double-click, impatient re-click, or a queued event replay) would spawn
    a second fleet re-embedding the same corpus -- a sync storm that hammers
    the embed endpoint and can double-insert chunks whose dedupe check ran
    before the first pass committed. A short cooldown after a sync finishes
    also swallows rapid re-triggers.

    Deliberately not a plain threading.Lock held by callers: the UI needs an
    immediate "busy" answer, not a pile of clicks waiting their turn.
    """

    def __init__(self, cooldown_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._cooldown = cooldown_seconds
        self._last_finished = 0.0

    def try_begin(self) -> bool:
        """True if this caller won the right to run a sync right now."""
        if not self._lock.acquire(blocking=False):
            return False
        if time.time() - self._last_finished < self._cooldown:
            self._lock.release()
            return False
        return True

    def finish(self) -> None:
        """Release the gate after a sync completes (success or failure)."""
        self._last_finished = time.time()
        self._lock.release()


class StopEvents:
    """Registry of per-user stop Events so one session's Stop can't cancel
    another user's in-flight stream. The Gradio variant kept a per-session
    `state` dict holding an Event; the API is stateless per request, so the
    Event is keyed by the (few, allowlisted) session user instead. Each
    stream clears its Event on begin, so a stale Stop left over from an
    already-finished answer is a no-op."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}

    def event_for(self, key: Optional[str]) -> threading.Event:
        key = key or "anonymous"
        with self._lock:
            event = self._events.get(key)
            if event is None:
                event = threading.Event()
                self._events[key] = event
            return event


def is_model_loaded(client: Any, model: str, min_ctx: Optional[int] = None) -> Optional[bool]:
    """Check whether `model` is already running on the Ollama server (/api/ps).

    Args:
        client: An ollama.Client instance.
        model: Model name as configured, to match against running models.
        min_ctx: If set, a running instance only counts as loaded when it
            carries at least this many context tokens -- one loaded with a
            smaller window has to be reloaded to serve the configured one.

    Returns:
        True if the model is running with a sufficient context window,
        False if it definitively isn't, and None if the check itself
        failed -- callers must treat "unknown" as "don't touch anything",
        since a blind load request could reload an already-good instance.
    """
    try:
        running = client.ps().models
    except Exception:
        logger.warning("Could not list running models on the Ollama server", exc_info=True)
        return None
    for entry in running:
        name = getattr(entry, "model", None) or getattr(entry, "name", "")
        if name != model:
            continue
        if min_ctx is None:
            return True
        ctx = getattr(entry, "context_length", None)
        if ctx is None:
            # Server doesn't report the loaded window size; assume the
            # running instance is fine rather than force a blind reload.
            return True
        if ctx >= min_ctx:
            return True
        logger.info(
            "%s is running with a %s-token window but %s is configured; "
            "it must be reloaded to serve the larger context",
            model, ctx, min_ctx,
        )
        return False
    return False


def ensure_model_loaded(
    client: Any, model: str, keep_alive: float = -1, num_ctx: Optional[int] = None
) -> bool:
    """Load `model` on the Ollama server unless it is already running there.

    Sends a load-only request (no prompt), so Ollama loads the model and
    returns immediately; keep_alive decides how long it stays resident.
    Checking /api/ps first saves the load round trip and, more
    importantly, avoids touching an already-running instance: a load
    request with different options would force a full model reload.

    Args:
        client: An ollama.Client instance.
        model: Model name to load.
        keep_alive: Seconds the model stays loaded after the call
            (-1 = keep it resident forever).
        num_ctx: Context window to load the model with; None uses the
            model's own default (correct for embedding models, whose
            window must not be inflated to the chat model's size).

    Returns:
        True if a load was issued, False if it was skipped (already
        running, the running-state check failed, or the load failed).
        Never raises -- prewarming must not take the app down.
    """
    loaded = is_model_loaded(client, model, min_ctx=num_ctx)
    if loaded is None:
        # Can't tell what's running -- never issue a blind load request:
        # it could reload (and momentarily stop) an already-good instance.
        logger.warning("Not preloading %s: the Ollama server's running-model list is unavailable", model)
        return False
    if loaded:
        logger.info("%s is already loaded; skipping preload", model)
        return False
    try:
        client.generate(
            model=model,
            keep_alive=keep_alive,
            options={"num_ctx": num_ctx} if num_ctx else None,
        )
        logger.info("Preloaded %s (num_ctx=%s, keep_alive=%s)", model, num_ctx, keep_alive)
        return True
    except Exception:
        logger.warning(
            "Preloading %s failed; the first request will pay the cold-load cost",
            model, exc_info=True,
        )
        return False


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


def embed_text(
    client: Any,
    text: str,
    model: str,
    title: str = None,
    keep_alive: Optional[float] = None,
) -> List[float]:
    """Generate an embedding for a chunk of text (optionally titled) via Ollama.

    keep_alive (None = server default, -1 = keep the model loaded forever)
    is passed through so an embed call can't silently reset the model's
    unload timer and evict a keep-forever instance.
    """
    enriched = f"Title: {title}\nContent: {text}" if title else text
    enriched = normalize_text(enriched)
    return client.embeddings(model=model, prompt=enriched, keep_alive=keep_alive)["embedding"]


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


def collect_source_labels(chunks: List[Tuple[str, Optional[Dict[str, Any]]]]) -> List[str]:
    """Unique citation labels (in retrieval order) for the documents behind
    a set of retrieved chunks."""
    labels: List[str] = []
    for _text, meta in chunks:
        label = format_source_label(meta)
        if label not in labels:
            labels.append(label)
    return labels


def content_sources_payload(chunks: List[Tuple[str, Optional[Dict[str, Any]]]]) -> Optional[Dict[str, Any]]:
    """Structured 'Sources' footer data for content-search answers, from the
    documents actually fed to the model. The user deserves to see *which*
    files answered; the model's self-narration is not reliable enough.
    Returns None when nothing was retrieved (an empty footer would be noise)
    -- the answer should say so itself.
    """
    labels = collect_source_labels(chunks)
    if not labels:
        return None
    return {"kind": "content", "labels": labels, "total": len(labels)}


def catalog_sources_payload(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Footer for metadata-catalog answers ("list all PDFs"). Those are
    answered from index metadata rather than a content search -- that's the
    fact users should see, or a list they can't reconcile with the corpus
    just looks wrong."""
    labels: List[str] = []
    for doc in docs:
        label = format_source_label(doc)
        if label not in labels:
            labels.append(label)
    return {"kind": "catalog", "labels": labels, "documents": len(docs)}


def build_suggestions(sources: Optional[List[str]], limit: int = 4) -> List[str]:
    """Example questions for the chat's welcome state. The first couple are
    generated from real indexed document names (clicking one asks about a
    file that actually exists instead of a placeholder), the rest are
    generic catalog questions that work on any corpus."""
    out = [f'What\u2019s in "{name}"?' for name in (sources or [])[:2]]
    out.append("List all PDFs")
    out.append("What documents are shared with me?")
    return out[:limit]


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
        # Word files count as "docs": "list all docs" must catch them, and
        # the catalog query filters on stored mime_type, so every Word flavor
        # has to be listed (Google Docs, .docx, legacy .doc).
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
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


def normalize_source_name(name: str) -> str:
    """Lowercase, whitespace-normalized form for source/query matching."""
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


_SOURCE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _source_tokens(name: str) -> List[str]:
    """Significant tokens of a document name: 3+ chars, file extension
    stripped ("2019_RESUME.pdf" -> ["2019", "resume"])."""
    stem = os.path.splitext(name or "")[0]
    return [t for t in _SOURCE_TOKEN_RE.findall(normalize_source_name(stem)) if len(t) >= 3]


def detect_mentioned_sources(query: str, sources: List[str]) -> List[str]:
    """Return indexed source names the query explicitly mentions.

    Matching is token-based, not substring: a source matches when every
    significant token of its name (3+ chars, extension stripped) appears
    somewhere in the query, in any order. Substring matching missed real
    questions -- "Show Ovais resume from 2019" never contains
    "2019_resume.pdf" verbatim, but the query carries both of its tokens
    (2019, resume) -- and pure vector ranking can bury a named document
    when the corpus is dominated by dense numeric chunks (the original
    "Kona April 2026" failure).

    Returns matched sources in their given order. Tiny names ("R") have
    no significant tokens and never match; a query naming several
    documents routes to all of them.
    """
    query_tokens = set(_SOURCE_TOKEN_RE.findall(normalize_source_name(query)))
    matched = []
    for source in sources:
        tokens = _source_tokens(source)
        if tokens and all(t in query_tokens for t in tokens):
            matched.append(source)
    return matched


def is_low_information(text: str, min_alpha_ratio: float = 0.3) -> bool:
    """True for chunks that are mostly digits/punctuation (spreadsheet
    cell dumps like "2020,4,29,4.39,202").

    Such chunks carry nothing for RAG answers and poison vector search:
    one 1400-chunk bq-results dump made up 60% of a real deployment's
    index and buried actual documents for numeric-flavored queries.
    They are skipped at indexing time.
    """
    if not text:
        return True
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / len(text) < min_alpha_ratio


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
- Context chunks labeled [user-corrections] are corrections a user reported after a wrong answer. Treat them as the authoritative truth about their question and let them override any conflicting document content.
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


def extract_docx_text(docx_bytes: bytes) -> Optional[str]:
    """Render a .docx (Office Open XML) document as plain text, stdlib-only.

    A .docx is a zip; the document text is the <w:t> runs of the <w:p>
    paragraphs in word/document.xml (table cells are <w:p> too, so their
    text comes along in document order). Enough for embedding/retrieval,
    not a faithful layout reproduction -- same bar as extract_xlsx_text.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    if not docx_bytes:
        return None
    _W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            with zf.open("word/document.xml") as f:
                tree = ET.parse(f)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError):
        logger.exception("Failed to parse .docx")
        return None

    paragraphs = []
    for para in tree.iter(f"{_W}p"):
        runs = [node.text or "" for node in para.iter(f"{_W}t")]
        if runs:
            paragraphs.append("".join(runs))
    text = "\n".join(paragraphs).strip()
    return text or None


def extract_doc_text(doc_bytes: bytes) -> Optional[str]:
    """Extract text from a legacy .doc (OLE2/Word 97 binary) via antiword
    or catdoc (antiword preferred; whichever is installed is used).

    There is no practical pure-Python extractor for the Word 97 binary
    format, so this shells out. When neither tool is installed it returns
    None and the file is skipped-and-named by the upload summary -- the
    same treatment any unextractable file gets. The Docker image installs
    both (see Dockerfile).
    """
    import shutil
    import subprocess
    import tempfile

    if not doc_bytes:
        return None
    tool = shutil.which("antiword") or shutil.which("catdoc")
    if not tool:
        logger.warning(
            "No .doc extractor available (antiword/catdoc not on PATH); "
            "skipping file"
        )
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".doc", delete=False)
    try:
        tmp.write(doc_bytes)
        tmp.close()
        proc = subprocess.run([tool, tmp.name], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        logger.exception("Failed to extract .doc text via %s", tool)
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if proc.returncode != 0:
        logger.warning(
            "%s failed on .doc input (exit %d): %s",
            tool, proc.returncode, proc.stderr[:200].decode("utf-8", "replace"),
        )
        return None
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    return text or None


def extract_text_from_upload(file_path: str) -> Optional[str]:
    """Extract plain text from a locally-uploaded .md/.txt/.pdf/.doc/.docx/.xlsx
    file, or None if the extension isn't supported."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".md", ".txt"):
        return read_markdown(file_path)
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            return extract_pdf_text(f.read())
    if ext == ".xlsx":
        with open(file_path, "rb") as f:
            return extract_xlsx_text(f.read())
    if ext == ".docx":
        with open(file_path, "rb") as f:
            return extract_docx_text(f.read())
    if ext == ".doc":
        with open(file_path, "rb") as f:
            return extract_doc_text(f.read())
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


def drive_sync_timestamp_path() -> str:
    """Sidecar file next to the Drive token store holding the last
    successful sync's completion time (float epoch), for the UI's
    'Drive synced 2 h ago' line. Imported lazily: gdrive_config imports
    this module, so a top-level import would be circular."""
    from gdrive_config import DRIVE_CONFIG

    return DRIVE_CONFIG["token_store_path"] + ".last_sync"


def record_drive_sync_timestamp() -> None:
    """Record that a Drive sync just succeeded. Never raises -- the sync
    itself already succeeded; losing the timestamp is a cosmetic problem."""
    try:
        with open(drive_sync_timestamp_path(), "w", encoding="utf-8") as f:
            f.write(repr(time.time()))
    except OSError:
        logger.warning("Could not record Drive sync timestamp", exc_info=True)


def read_drive_sync_timestamp() -> Optional[float]:
    try:
        with open(drive_sync_timestamp_path(), encoding="utf-8") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def safe_error_message(exc: Exception, log: logging.Logger = None) -> str:
    """Log the full exception server-side and return a generic, correlated
    message safe to show to end users (avoids leaking internals/stack traces).
    """
    error_id = uuid.uuid4().hex[:8]
    (log or logger).error("error_id=%s request failed: %r", error_id, exc, exc_info=exc)
    return f"Sorry, something went wrong processing your request (error id: {error_id})."
