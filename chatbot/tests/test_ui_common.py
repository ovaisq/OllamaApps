"""Tests for the shared UI design system + pure helpers (theme/CSS are
exercised by the real gr.Blocks build in test_app_build / test_*_chatty)."""
import time
from unittest.mock import MagicMock, patch

from ui_common import (
    feedback_prefill,
    format_index_chip,
    format_index_summary,
    humanize_age,
    load_js,
    refresh_feedback_rows,
)

SAMPLE_SUMMARY = {
    "chunks": 412,
    "documents": 37,
    "top_sources": [("a.md", 54), ("b.pdf", 30)],
    "last_sync": None,
}


def test_index_chip_is_one_line_with_counts_and_sync_age():
    chip = format_index_chip(SAMPLE_SUMMARY)
    assert "412" in chip
    assert "37" in chip
    assert "Drive synced never" in chip


def test_index_summary_lists_top_sources():
    summary_md = format_index_summary(SAMPLE_SUMMARY)
    assert "**Documents:** 37" in summary_md
    assert "**Indexed chunks:** 412" in summary_md
    assert "a.md (54)" in summary_md
    assert "b.pdf (30)" in summary_md


def test_format_index_summary_never_raises_on_garbage():
    """A stats hiccup must not take the Library tab down with it."""
    assert format_index_summary(object()) == ""


def test_humanize_age_buckets():
    now = time.time()
    assert humanize_age(None) == "never"
    assert humanize_age(now - 30) == "just now"
    assert humanize_age(now - 600) == "10 min ago"
    assert humanize_age(now - 7200) == "2 h ago"
    assert humanize_age(now - 172800) == "2 d ago"
    assert humanize_age(now + 5000) == "just now"  # future clock skew: clamp


def test_refresh_feedback_rows_empty_state():
    choices, preview, rows = refresh_feedback_rows(lambda email: [])
    assert choices == []
    assert rows == []
    assert "No dislikes recorded yet" in preview


def test_refresh_feedback_rows_shapes_choices_rows_and_preview():
    rows_in = [
        {"id": 1, "question": "q one", "answer": "a one", "created": "2026-01-02 03:04"},
        {"id": 2, "question": "q two", "answer": "a two", "created": "2026-01-01 02:03"},
    ]
    with patch("app_session.get_email_from_request", return_value="u@example.com"):
        choices, preview, rows = refresh_feedback_rows(lambda email: rows_in)

    assert [c[0] for c in choices] == ["1", "2"]
    assert rows is rows_in  # carried as state for the prefill lookup
    assert "q one" in preview  # the newest dislike's question leads the preview


def test_refresh_feedback_rows_unauthenticated_sees_nothing():
    fn = MagicMock()
    with patch("app_session.get_email_from_request", return_value=None):
        choices, _preview, rows = refresh_feedback_rows(fn)

    fn.assert_not_called()
    assert choices == []
    assert rows == []


def test_refresh_feedback_rows_survives_backend_errors():
    def boom(email):
        raise RuntimeError("db down")

    with patch("app_session.get_email_from_request", return_value="u@example.com"):
        choices, _preview, rows = refresh_feedback_rows(boom)

    assert choices == []
    assert rows == []


def test_feedback_prefill_matches_by_row_id():
    rows = [{"id": 7, "question": "the question"}]
    assert feedback_prefill("7", rows) == "the question"
    assert feedback_prefill(7, rows) == "the question"  # ids may be int or str


def test_feedback_prefill_unresolvable_selection_is_a_noop():
    """Returning None tells gradio to leave the target untouched (a cleared
    dropdown must not wipe whatever the user typed)."""
    assert feedback_prefill(None, [{"id": 7, "question": "q"}]) is None
    assert feedback_prefill("99", [{"id": 7, "question": "q"}]) is None
    assert feedback_prefill("7", None) is None


def test_load_js_includes_scroll_and_composer_behavior():
    js = load_js()
    assert "bubble-wrap" in js  # smart scrolling (rag_common CHAT_UI_JS)
    assert "chatty-composer" in js  # composer -> Stop disabled-state mirror
