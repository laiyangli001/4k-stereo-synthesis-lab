from stereo_runtime.progress import file_size_progress, make_rich_progress, progress_write, write_bytes_with_progress
from stereo_runtime.progress import supports_live_progress


def test_write_bytes_with_progress_writes_file_and_reports_total(tmp_path, capsys):
    target = tmp_path / "model.trt"

    write_bytes_with_progress(target, b"trt-bytes", "Saving TensorRT engine: model.trt", chunk_size=3)

    assert target.read_bytes() == b"trt-bytes"
    out = capsys.readouterr().out
    assert "Saving TensorRT engine: model.trt" in out
    assert "100.00%" in out


def test_file_size_progress_uses_file_growth_as_approximation(tmp_path, capsys):
    target = tmp_path / "model.trt"

    with file_size_progress("Building TensorRT engine: model.trt", target, total_bytes=9, interval_s=999):
        target.write_bytes(b"123456789")

    out = capsys.readouterr().out
    assert "Building TensorRT engine: model.trt" in out
    assert "100.00%" in out


def test_supports_live_progress_can_be_forced_by_env(monkeypatch):
    monkeypatch.setenv("D2S_FORCE_RICH_PROGRESS", "1")

    assert supports_live_progress(object()) is True


def test_rich_progress_renders_single_line(monkeypatch, capsys):
    monkeypatch.setenv("D2S_FORCE_RICH_PROGRESS", "1")

    progress = make_rich_progress(total=100, desc="model.safetensors", leave=True)
    progress.update(100)
    progress.close()

    out = capsys.readouterr().out
    assert "model.safetensors" in out
    assert "100%" in out


def test_progress_write_uses_rich_console(monkeypatch, capsys):
    monkeypatch.setenv("D2S_FORCE_RICH_PROGRESS", "1")

    progress_write("[Main] Stopped")

    assert "[Main] Stopped" in capsys.readouterr().out
