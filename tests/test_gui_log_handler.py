import logging

from gui.log_handler import GuiLogHandler


def test_status_cache_survives_noisy_child_log_eviction():
    handler = GuiLogHandler(maxlen=2)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"))

    handler.emit(logging.LogRecord("status", logging.INFO, "", 1, "Running", (), None))
    handler.emit(logging.LogRecord("child", logging.INFO, "", 2, "line 1", (), None))
    handler.emit(logging.LogRecord("child", logging.INFO, "", 3, "line 2", (), None))

    assert all(item[1] != "status" for item in handler.cache)
    assert [item[1] for item in handler.status_cache] == ["status"]
    assert "Running" in handler.status_cache[-1][3]
