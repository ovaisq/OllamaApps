"""FastAPI routes that let an operator grant this deployment Drive read access
via a browser, once. Mounted onto the same app/port that serves the chatbot UI
so the OAuth redirect_uri can point at a single publicly reachable host
(e.g. https://chatty.ifthenelse.net:7860/oauth2callback).

Both routes are gated: /connect-drive requires a pre-shared admin token (these
routes are reachable by anyone who can hit the app's port, and without this
gate anyone could bind the app's Drive integration to their own Google
account, poisoning the shared vector store). /oauth2callback verifies the
OAuth `state` it receives was the one this app itself issued, so a code
obtained through another channel can't be replayed here.
"""
import logging
import secrets

import fastapi
from fastapi.responses import RedirectResponse, JSONResponse

from gdrive_auth import exchange_code_for_tokens, get_google_auth_url
from gdrive_config import DRIVE_CONFIG, GOOGLE_CONFIG
from gdrive_token_store import save_refresh_token

logger = logging.getLogger(__name__)

# Single-operator, single-process flow: one pending state is all that's needed
# between /connect-drive issuing it and /oauth2callback consuming it.
_pending_state = {"value": None}


def _admin_token_valid(request: fastapi.Request) -> bool:
    expected = GOOGLE_CONFIG.get("admin_token")
    if not expected:
        logger.error("GDRIVE_ADMIN_TOKEN is not configured; refusing Drive OAuth connect")
        return False
    supplied = request.query_params.get("token", "")
    return secrets.compare_digest(supplied, expected)


def register_routes(app: fastapi.FastAPI) -> None:
    @app.get("/connect-drive")
    def connect_drive(request: fastapi.Request):
        if not _admin_token_valid(request):
            return JSONResponse({"status": "error", "detail": "forbidden"}, status_code=403)

        state = secrets.token_urlsafe(32)
        _pending_state["value"] = state
        auth_url = get_google_auth_url(
            GOOGLE_CONFIG["client_id"], GOOGLE_CONFIG["redirect_uri"], state=state
        )
        return RedirectResponse(auth_url)

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

        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            # Google omits refresh_token on repeat consents unless prompt=consent
            # forces a new one, which get_google_auth_url already sets.
            logger.error("No refresh_token in Google OAuth response")
            return JSONResponse(
                {"status": "error", "detail": "no refresh_token returned"}, status_code=400
            )

        save_refresh_token(DRIVE_CONFIG["token_store_path"], refresh_token)
        return JSONResponse({"status": "ok", "detail": "Google Drive connected"})
