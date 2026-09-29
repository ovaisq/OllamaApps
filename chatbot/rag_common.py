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
.typing-indicator .typing-timer {
    width: auto; height: auto; border-radius: 0;
    background: none; animation: none;
    margin-left: 6px; align-self: center;
    font-size: 12px; opacity: 0.6;
}
"""

# Client-side chat UI behavior, wired via the Blocks .load(js=...) event so
# it runs once per page load. Two jobs:
#
# 1. Smart scrolling (replaces Gradio's built-in autoscroll, which runs two
#    overlapping mechanisms with different "user scrolled up" thresholds --
#    a token arriving inside the gap yanks the view back down, which reads
#    as "a pending answer blocks the scroll"). One deterministic rule:
#    follow the newest content only while the user is already near the
#    bottom; the moment they scroll up, stop fighting them; pressing Enter
#    (sending a message) resumes following.
#
# 2. A visible elapsed timer next to the typing dots while the answer is
#    pending. Gradio's status tracker cannot serve this role: it hides
#    itself as soon as the event starts streaming (and the typing-indicator
#    yield makes the chat stream immediately), so no built-in progress
#    setting produces a visible chat timer. This one renders in exactly
#    one place -- inside the pending answer bubble -- counts up until the
#    first real text replaces the dots, and removes itself then.
#
# Targets the chatbot's scroll container (div.bubble-wrap, verified in
# gradio 6.28's rendered DOM).
CHAT_UI_JS = """(function () {
    var following = true;
    var NEAR_BOTTOM_PX = 80;
    function nearBottom(el) {
        return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    }
    function install(el) {
        el.addEventListener('scroll', function () {
            following = nearBottom(el);
        }, { passive: true });
        var box = document.querySelector('textarea[data-testid="textbox"]');
        if (box) {
            box.addEventListener('keydown', function (event) {
                if (event.key === 'Enter') { following = true; }
            });
        }
        var observer = new MutationObserver(function () {
            manageTimer(el);
            if (following) { el.scrollTop = el.scrollHeight; }
        });
        observer.observe(el, { childList: true, subtree: true, characterData: true });
        if (following) { el.scrollTop = el.scrollHeight; }
    }
    var timerInterval = null;
    var timerSeconds = 0;
    var timerSpan = null;
    function stopTimer() {
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        timerSpan = null;
        timerSeconds = 0;
    }
    function manageTimer(el) {
        var dots = el.querySelector('.typing-indicator');
        if (!dots) {
            if (timerInterval) { stopTimer(); }
            return;
        }
        if (!timerInterval) {
            timerSeconds = 0;
            timerSpan = document.createElement('span');
            timerSpan.className = 'typing-timer';
            timerSpan.textContent = '0s';
            dots.appendChild(timerSpan);
            timerInterval = setInterval(function () {
                timerSeconds += 1;
                var current = el.querySelector('.typing-indicator .typing-timer');
                if (!current) { stopTimer(); return; }
                current.textContent = timerSeconds + 's';
            }, 1000);
        }
    }
    function start() {
        var el = document.querySelector('div.bubble-wrap');
        if (el) { install(el); return; }
        // The chatbot hydrates after the initial render; poll briefly.
        var tries = 0;
        var timer = setInterval(function () {
            var found = document.querySelector('div.bubble-wrap');
            if (found) { clearInterval(timer); install(found); }
            else if (++tries > 40) { clearInterval(timer); }
        }, 250);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();"""


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
