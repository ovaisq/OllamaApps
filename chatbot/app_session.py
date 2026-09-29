"""Stateless signed session cookies for gating the chatbot UI.

Uses PyJWT (same library as ~/CODE/ai/almanac/auth.py) rather than a
DB-backed session table, since this app only needs to recognize "is this
request from one of the allowlisted operator emails" -- no per-user state.
"""
import logging
import time
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

SESSION_COOKIE = "chatty_session"
_ALGORITHM = "HS256"


def create_session_token(email: str, secret: str, max_age_seconds: int) -> str:
    now = int(time.time())
    payload = {"email": email, "iat": now, "exp": now + max_age_seconds}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def verify_session_token(token: str, secret: str) -> Optional[str]:
    """Return the email embedded in a valid, unexpired token, or None."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError as e:
        logger.debug("Rejecting session token: %s", e)
        return None
    return payload.get("email")


def get_email_from_request(request, secret: str) -> Optional[str]:
    """Extract and verify the session cookie's email from a gr.Request
    inside a Gradio event handler -- outside FastAPI route/middleware
    context, so this can't rely on request.state.user_email being set.
    Used for per-user chat history (whose messages are these?).
    """
    if request is None:
        return None
    cookies = getattr(request, "cookies", None) or {}
    if hasattr(cookies, "get"):
        token = cookies.get(SESSION_COOKIE)
    else:
        token = getattr(cookies, SESSION_COOKIE, None)
    return verify_session_token(token, secret)
