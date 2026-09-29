"""Minimal Google Drive v3 REST client: list files and pull plain text out of
Google Docs/Sheets, .md/.txt/.xlsx files, and PDFs. No google-api-python-client
dependency (matches the httpx-against-raw-endpoints pattern used elsewhere in
this repo).
"""
import logging
from typing import Dict, Iterator, Optional

import httpx

from rag_common import extract_pdf_text, extract_xlsx_text

logger = logging.getLogger(__name__)

_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_SUPPORTED_MIMES = {
    _GOOGLE_DOC_MIME,
    _GOOGLE_SHEET_MIME,
    _XLSX_MIME,
    "text/plain",
    "text/markdown",
    "application/pdf",
}


def list_files(access_token: str, folder_id: str = "") -> Iterator[Dict]:
    """Yield {id, name, mimeType} for every supported file visible to the
    account (or within folder_id, non-recursively, when given). Files seen
    but skipped (unsupported type) are logged so a sync that "finds nothing"
    is diagnosable without manually querying the Drive API.
    """
    query_parts = ["trashed = false"]
    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")
    params = {
        "q": " and ".join(query_parts),
        "fields": "nextPageToken, files(id, name, mimeType)",
        "pageSize": 100,
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    with httpx.Client(timeout=15.0) as client:
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            resp = client.get(
                "https://www.googleapis.com/drive/v3/files",
                params=params,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                logger.error("Drive list failed: %s %s", resp.status_code, resp.text[:200])
                raise RuntimeError("Failed to list Drive files")
            data = resp.json()
            for f in data.get("files", []):
                if f.get("mimeType") in _SUPPORTED_MIMES:
                    yield f
                else:
                    logger.info(
                        "Skipping Drive file %r: unsupported mimeType %s",
                        f.get("name"), f.get("mimeType"),
                    )
            page_token = data.get("nextPageToken")
            if not page_token:
                break


def fetch_file_text(access_token: str, file_id: str, mime_type: str) -> Optional[str]:
    """Return plain text content for a supported Drive file, or None if the
    file type isn't one we can extract text from.
    """
    if mime_type == _GOOGLE_DOC_MIME:
        return _export_google_doc(access_token, file_id)
    if mime_type == _GOOGLE_SHEET_MIME:
        # Drive's /export only returns the first sheet as CSV; a multi-tab
        # Google Sheet loses everything past tab 1 through this path.
        return _export_google_sheet(access_token, file_id)
    if mime_type in ("text/plain", "text/markdown"):
        return _download_raw(access_token, file_id).decode("utf-8", errors="replace")
    if mime_type == "application/pdf":
        return extract_pdf_text(_download_raw(access_token, file_id))
    if mime_type == _XLSX_MIME:
        return extract_xlsx_text(_download_raw(access_token, file_id))
    return None


def _export_google_doc(access_token: str, file_id: str) -> Optional[str]:
    resp = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
        params={"mimeType": "text/plain"},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    )
    if resp.status_code != 200:
        logger.warning("Doc export %s failed: %s %s", file_id, resp.status_code, resp.text[:200])
        return None
    return resp.text


def _export_google_sheet(access_token: str, file_id: str) -> Optional[str]:
    resp = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
        params={"mimeType": "text/csv"},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    )
    if resp.status_code != 200:
        logger.warning(
            "Sheet export %s failed: %s %s", file_id, resp.status_code, resp.text[:200]
        )
        return None
    return resp.text


def _download_raw(access_token: str, file_id: str) -> bytes:
    resp = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        params={"alt": "media"},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    )
    if resp.status_code != 200:
        logger.warning("Download %s failed: %s %s", file_id, resp.status_code, resp.text[:200])
        return b""
    return resp.content
