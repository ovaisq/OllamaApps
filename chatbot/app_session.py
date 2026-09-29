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
