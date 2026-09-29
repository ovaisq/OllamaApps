"""Persists the long-lived Google OAuth refresh token between indexer runs.

Only the refresh_token needs to survive across process runs; access tokens are
short-lived and re-minted from it each time via gdrive_auth.refresh_access_token.
"""
import json
import logging
import os
import stat
from typing import Optional

logger = logging.getLogger(__name__)


def save_refresh_token(path: str, refresh_token: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"refresh_token": refresh_token}, f)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600: refresh token is a bearer credential


def load_refresh_token(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("refresh_token")
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read token store at %s", path)
        return None
