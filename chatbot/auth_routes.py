"""Gates the entire chatbot app (chat UI + admin tab) behind Google sign-in.

One login covers both: the same OAuth consent that authenticates the
operator also grants Drive read access, so a successful /oauth2callback
both issues a session cookie and (when a refresh_token comes back) persists
Drive access for gdrive_indexer.py.

Only emails in AUTH_CONFIG['allowed_emails'] can ever get a session --
completing Google's consent screen is necessary but not sufficient.
"""
import logging
import secrets

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app_session import SESSION_COOKIE, create_session_token, verify_session_token
from gdrive_auth import exchange_code_for_tokens, fetch_userinfo, get_google_auth_url
from gdrive_config import AUTH_CONFIG, DRIVE_CONFIG, GOOGLE_CONFIG
from gdrive_token_store import save_refresh_token

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {"/login", "/oauth2callback", "/health"}

# Single-operator, single-process flow: one pending state between /login
# issuing it and /oauth2callback consuming it.
_pending_state = {"value": None}


def _is_allowed_email(email: str) -> bool:
    allowed = AUTH_CONFIG.get("allowed_emails") or set()
    return bool(email) and email.lower().strip() in allowed


class SessionAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        email = verify_session_token(
            request.cookies.get(SESSION_COOKIE), AUTH_CONFIG["session_secret"]
        )
        if not email or not _is_allowed_email(email):
            return RedirectResponse("/login")

        request.state.user_email = email
        return await call_next(request)


def register_routes(app: fastapi.FastAPI) -> None:
    app.add_middleware(SessionAuthMiddleware)

    @app.get("/login")
    def login():
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
            logger.error("Google OAuth consent denied/error: %s", error)
            return JSONResponse({"status": "error", "detail": error}, status_code=400)
        if not code:
            return JSONResponse({"status": "error", "detail": "missing code"}, status_code=400)

        expected_state = _pending_state["value"]
        if not expected_state or not state or not secrets.compare_digest(state, expected_state):
            logger.error("OAuth callback state mismatch or replay attempt")
            return JSONResponse({"status": "error", "detail": "invalid state"}, status_code=403)
        _pending_state["value"] = None  # one-shot: consumed regardless of outcome below

        try:
            tokens = await exchange_code_for_tokens(
                GOOGLE_CONFIG["client_id"],
                GOOGLE_CONFIG["client_secret"],
                code,
                GOOGLE_CONFIG["redirect_uri"],
            )
        except RuntimeError as e:
            logger.exception("OAuth code exchange failed")
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

        try:
            userinfo = await fetch_userinfo(tokens.get("access_token"))
        except RuntimeError as e:
            logger.exception("Failed to fetch Google account info")
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

        email = (userinfo.get("email") or "").lower().strip()
        if not _is_allowed_email(email):
            # Deliberately do not save any refresh_token here: an
            # unauthorized account must never be able to overwrite the
            # legitimate Drive connection.
            logger.warning("Rejected login attempt from non-allowlisted account")
            return JSONResponse({"status": "error", "detail": "not authorized"}, status_code=403)

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
