from admin_ui import get_index_stats, sync_drive_now, upload_and_index


def test_upload_and_index_rejects_missing_file():
    assert upload_and_index(None, index_text_fn=lambda t, s: 1) == "No file selected."


def test_upload_and_index_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_text("hi")
    result = upload_and_index(str(f), index_text_fn=lambda t, s: 1)
    assert "No extractable text" in result


def test_upload_and_index_indexes_markdown_and_reports_count(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello world")

    calls = {}

    def index_text_fn(text, source):
        calls["text"] = text
        calls["source"] = source
        return 3

    result = upload_and_index(str(f), index_text_fn)

    assert calls["source"] == "notes.md"
    assert "hello world" in calls["text"]
    assert result == "Indexed 3 new chunk(s) from notes.md."


def test_upload_and_index_hides_internal_errors(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hello")

    def index_text_fn(text, source):
        raise ConnectionError("db-password=hunter2")

    result = upload_and_index(str(f), index_text_fn)
    assert "hunter2" not in result


def test_sync_drive_now_reports_chunk_count():
    assert sync_drive_now(lambda: 5) == "Synced Google Drive: 5 new chunk(s) indexed."


def test_sync_drive_now_tells_user_to_connect_drive_when_not_authorized():
    def drive_sync_fn():
        raise RuntimeError("No Google Drive refresh token found.")

    result = sync_drive_now(drive_sync_fn)
    assert "/login" in result


def test_get_index_stats_reports_total():
    assert get_index_stats(lambda: 42) == "Total indexed chunks: 42"


def test_get_index_stats_hides_internal_errors():
    def count_fn():
        raise ConnectionError("db-password=hunter2")

    result = get_index_stats(count_fn)
    assert "hunter2" not in result
