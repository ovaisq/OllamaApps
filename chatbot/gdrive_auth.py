"""Google OAuth helpers for Drive read access.

Mirrors the httpx-against-raw-REST-endpoints pattern already used in
~/CODE/ai/almanac/auth.py, rather than pulling in google-api-python-client.
"""
import logging
import secrets
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def get_google_auth_url(client_id: str, redirect_uri: str, state: str = None) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": DRIVE_READONLY_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state or secrets.token_urlsafe(32),
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


async def exchange_code_for_tokens(
    client_id: str, client_secret: str, code: str, redirect_uri: str
) -> dict:
    """Exchange an authorization code for an access_token/refresh_token pair."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if resp.status_code != 200:
        logger.error("Google code exchange failed: %s %s", resp.status_code, resp.text[:200])
        raise RuntimeError("Failed to exchange Google authorization code")
    return resp.json()


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Mint a fresh access_token from a stored refresh_token."""
    resp = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=10.0,
    )
    if resp.status_code != 200:
        logger.error("Google token refresh failed: %s %s", resp.status_code, resp.text[:200])
        raise RuntimeError("Failed to refresh Google access token")
    access_token = resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Google token refresh response missing access_token")
    return access_token
