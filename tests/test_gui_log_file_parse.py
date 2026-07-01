import logging
from pathlib import Path


def test_log_item_from_file_line_parses_standard_log_line(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1] / "src")
    from gui.process import _log_item_from_file_line

    line = "[18:53:28] [INFO] [status] Running"

    assert _log_item_from_file_line(line) == (logging.INFO, "status", "18:53:28", line)


def test_log_item_from_file_line_ignores_non_log_line(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1] / "src")
    from gui.process import _log_item_from_file_line

    assert _log_item_from_file_line("[18:53:28] [diag] process started") is None
