import logging
from pathlib import Path


def test_debug_file_logging_writes_debug_without_console_handler(tmp_path):
    from utils.logging_setup import configure_debug_file_logging

    log_file = tmp_path / "desktop2stereo.log"
    logger = logging.getLogger("d2s_test_debug_file")
    flet_logger = logging.getLogger("flet_controls")
    pil_logger = logging.getLogger("PIL.PngImagePlugin")
    transport_logger = logging.getLogger("flet_transport")
    root = logging.getLogger()
    before = list(root.handlers)

    configure_debug_file_logging(log_file)
    try:
        logger.debug("debug detail")
        flet_logger.debug("Container(1 - 2).did_mount()")
        flet_logger.debug("Text(1 - 2).will_unmount()")
        flet_logger.debug("Trigger event Page(1 - 2).on_resize PageResizeEvent(name='resize')")
        flet_logger.warning("important flet warning")
        pil_logger.debug("STREAM b'IHDR' 16 13")
        pil_logger.warning("important png warning")
        transport_logger.debug("send_message: ClientMessage(action=<ClientActions.UPDATE>)")
        transport_logger.debug("_on_message: ServerMessage(action=<ServerActions.PAGE_EVENT>)")
        transport_logger.warning("important transport warning")

        for handler in root.handlers:
            handler.flush()

        text = log_file.read_text(encoding="utf-8")
        assert "debug detail" in text
        assert "important flet warning" in text
        assert "important png warning" in text
        assert "important transport warning" in text
        assert "send_message:" not in text
        assert "_on_message:" not in text
        assert "did_mount" not in text
        assert "will_unmount" not in text
        assert "Trigger event" not in text
        assert "STREAM b'IHDR'" not in text
        assert len(root.handlers) == len(before) + 1
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_main_process_configures_shared_debug_log():
    main_text = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")

    assert "from utils.logging_setup import configure_debug_file_logging" in main_text
    assert 'LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "desktop2stereo.log")' in main_text
    assert "configure_debug_file_logging(LOG_FILE)" in main_text
