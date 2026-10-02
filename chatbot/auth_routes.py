"""Gates the entire chatbot app (chat UI + admin tab) behind Google sign-in.

One login covers both: the same Google OAuth consent that authenticates the
operator also grants Drive read access, so a successful /oauth2callback
both issues a session cookie and (when a refresh_token comes back) persists
Drive access for gdrive_indexer.py.

Only emails in AUTH_CONFIG['allowed_emails'] can ever get a session --
completing Google's consent screen is necessary but not sufficient.

Temporary escape hatch: `AUTH_DISABLED=1` (see gdrive_config) makes the
SessionAuthMiddleware pass every request through -- no sign-in at all.
Intended for LAN/debug sessions; don't leave it on behind a public endpoint.

The login experience is branded HTML, never JSON: /login renders a Chatty
landing page with a "Sign in with Google" button that triggers the OAuth
flow via /oauth-start, and every callback failure (consent denied, state
mismatch, bad code, non-allowlisted account, network failure) renders a
styled error page. The old behavior returned raw JSON 400/403 bodies, so a
user who hit a broken consent flow saw `{"status":"error","detail":...}`.
"""
import html
import logging
import secrets
import uuid

import fastapi
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app_session import SESSION_COOKIE, create_session_token, verify_session_token
from gdrive_auth import exchange_code_for_tokens, fetch_userinfo, get_google_auth_url
from gdrive_config import AUTH_CONFIG, DRIVE_CONFIG, GOOGLE_CONFIG
from gdrive_token_store import save_refresh_token

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {"/login", "/oauth-start", "/oauth2callback", "/health"}

# Single-operator, single-process flow: one pending state between /login
# issuing it and /oauth2callback consuming it.
_pending_state = {"value": None}


def _is_allowed_email(email: str) -> bool:
    allowed = AUTH_CONFIG.get("allowed_emails") or set()
    return bool(email) and email.lower().strip() in allowed


# ---------------------------------------------------------------------------
# Branded auth pages (no frontend build/deps: one inline-CSS template)
# ---------------------------------------------------------------------------

_PAGE_CSS = """
body { margin: 0; min-height: 100vh; display: flex; align-items: center;
       justify-content: center; background: #eef1f6; color: #1f2937;
       font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
.card { background: #ffffff; border-radius: 12px; box-shadow: 0 8px 30px rgba(15,36,64,0.12);
        padding: 2.5rem 2.25rem; max-width: 440px; margin: 1rem; text-align: center; }
.brand { color: #2563a7; font-weight: 700; letter-spacing: 0.05em;
         text-transform: uppercase; font-size: 0.8rem; }
h1 { font-size: 1.4rem; margin: 0.75rem 0 0.25rem; }
.sub { color: #6b7280; margin: 0 0 1.25rem; font-size: 0.95rem; line-height: 1.45; }
.btn { display: inline-block; background: #2563a7; color: #fff; text-decoration: none;
       padding: 0.6rem 1.4rem; border-radius: 8px; font-weight: 600; margin-top: 0.5rem; }
.btn:hover { background: #1e528f; }
.hint { margin-top: 1.25rem; font-size: 0.8rem; color: #9ca3af; line-height: 1.5; }
.err { color: #b91c1c; }
"""


def _page(title: str, subtitle: str, body_html: str,
          status_code: int = 200) -> HTMLResponse:
    """title/subtitle are escaped; body_html is app-generated only."""
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)} — Chatty</title>"
        f"<style>{_PAGE_CSS}</style></head><body>"
        f"<main class='card'><div class='brand'>Chatty</div>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{html.escape(subtitle)}</p>"
        f"<div>{body_html}</div></main></body></html>"
    )
    return HTMLResponse(doc, status_code=status_code)


def _login_body():
    return (
        '<a class="btn" href="/oauth-start">Sign in with Google</a>'
        '<p class="hint">Restricted to authorized accounts. The same '
        'sign-in connects Google Drive (read-only) so Chatty can index '
        "it.</p>"
    )


def _error_page(title: str, subtitle: str, error_id: str = None,
                status_code: int = 400) -> HTMLResponse:
    hint = (
        f'<p class="hint err">error id: {html.escape(error_id)}</p>'
        if error_id else ""
    )
    return _page(title, subtitle,
                 '<a class="btn" href="/login">Back to sign-in</a>' + hint,
                 status_code=status_code)


class SessionAuthMiddleware(BaseHTTPMiddleware):
    def _unauthorized(self, request: Request):
        """Browsers get the branded login page; API clients (the web UI's
        fetch calls) get a 401 they can act on (JS redirects to /login)
        instead of a redirect whose JSON body they'd parse anyway."""
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return RedirectResponse("/login")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)

        if AUTH_CONFIG.get("auth_disabled"):
            # Temporary escape hatch (AUTH_DISABLED env, see gdrive_config):
            # skip the session gate entirely. Routes that key off the
            # signed-in email degrade gracefully without one (no persisted
            # history/feedback, shared stop event) -- fine for a LAN debug
            # session, never for a public endpoint.
            return await call_next(request)

        email = verify_session_token(
            request.cookies.get(SESSION_COOKIE), AUTH_CONFIG["session_secret"]
        )
        if not email or not _is_allowed_email(email):
            return self._unauthorized(request)

        request.state.user_email = email
        return await call_next(request)


def register_routes(app: fastapi.FastAPI) -> None:
    app.add_middleware(SessionAuthMiddleware)

    @app.get("/login")
    def login():
        # Branded landing page, not an immediate redirect: a broken
        # consent flow must still land the user on a Chatty screen with a
        # retry affordance, not on Google's or a JSON blob.
        return _page(
            "Sign in to Chatty",
            "Your documents, in your language, on your machines.",
            _login_body(),
        )

    @app.get("/oauth-start")
    def oauth_start():
        state = secrets.token_urlsafe(32)
        _pending_state["value"] = state
        auth_url = get_google_auth_url(
            GOOGLE_CONFIG["client_id"], GOOGLE_CONFIG["redirect_uri"], state=state
        )
        return RedirectResponse(auth_url)

    @app.get("/logout")
    def logout():
        resp = RedirectResponse("/login")
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/oauth2callback")
    async def oauth2callback(code: str = None, state: str = None, error: str = None):
        if error:
            # `error` is a Google error token ("access_denied", ...) -- log
            # it raw, never echo it back verbatim to the page.
            logger.error("Google OAuth consent denied/error: %s", error)
            return _error_page(
                "Sign-in was cancelled",
                "Google sign-in didn't complete, so nothing was signed in.",
                status_code=400,
            )
        if not code:
            return _error_page(
                "Sign-in didn't complete",
                "Google returned no authorization code. Please try again.",
                status_code=400,
            )

        expected_state = _pending_state["value"]
        if not expected_state or not state or not secrets.compare_digest(state, expected_state):
            logger.error("OAuth callback state mismatch or replay attempt")
            return _error_page(
                "Session expired",
                "The sign-in session expired (or was replayed). Sign in again.",
                status_code=403,
            )
        _pending_state["value"] = None  # one-shot: consumed regardless of outcome below

        try:
            tokens = await exchange_code_for_tokens(
                GOOGLE_CONFIG["client_id"],
                GOOGLE_CONFIG["client_secret"],
                code,
                GOOGLE_CONFIG["redirect_uri"],
            )
        except RuntimeError:
            # Details (URLs, tokens, network noise) stay server-side; the
            # page gets a generic message plus a correlatable error id.
            logger.exception("OAuth code exchange failed")
            return _error_page(
                "Sign-in didn't complete",
                "The token exchange with Google failed. Please try again.",
                error_id=uuid.uuid4().hex[:8],
                status_code=400,
            )

        try:
            userinfo = await fetch_userinfo(tokens.get("access_token"))
        except RuntimeError:
            logger.exception("Failed to fetch Google account info")
            return _error_page(
                "Sign-in didn't complete",
                "Could not fetch your Google account details. Please try again.",
                error_id=uuid.uuid4().hex[:8],
                status_code=400,
            )

        email = (userinfo.get("email") or "").lower().strip()
        if not _is_allowed_email(email):
            # Deliberately do not save any refresh_token here: an
            # unauthorized account must never be able to overwrite the
            # legitimate Drive connection. The page is generic on purpose:
            # echoing the attempted account name would let anyone probe the
            # allowlist.
            logger.warning("Rejected login attempt from non-allowlisted account")
            return _error_page(
                "Not authorized",
                "That Google account isn't authorized for this Chatty instance.",
                status_code=403,
            )

        refresh_token = tokens.get("refresh_token")
        if refresh_token:
            save_refresh_token(DRIVE_CONFIG["token_store_path"], refresh_token)

        session_token = create_session_token(
            email, AUTH_CONFIG["session_secret"], AUTH_CONFIG["session_max_age_seconds"]
        )
        resp = RedirectResponse("/")
        resp.set_cookie(
            SESSION_COOKIE,
            session_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=AUTH_CONFIG["session_max_age_seconds"],
        )
        return resp
