from stereo_runtime.progress import DownloadProgress, file_size_progress, progress_write, write_bytes_with_progress


def test_write_bytes_with_progress_writes_file_and_reports_total(tmp_path, capsys):
    target = tmp_path / "model.trt"

    write_bytes_with_progress(target, b"trt-bytes", "Saving TensorRT engine: model.trt", chunk_size=3)

    assert target.read_bytes() == b"trt-bytes"
    out = capsys.readouterr().out
    assert "Saving TensorRT engine: model.trt" in out
    assert "[D2S_PROGRESS]" in out
    assert '"percent":100.0' in out


def test_file_size_progress_uses_file_growth_as_approximation(tmp_path, capsys):
    target = tmp_path / "model.trt"

    with file_size_progress("Building TensorRT engine: model.trt", target, total_bytes=9, interval_s=999):
        target.write_bytes(b"123456789")

    out = capsys.readouterr().out
    assert "Building TensorRT engine: model.trt" in out
    assert "[D2S_PROGRESS]" in out
    assert '"percent":100.0' in out


def test_download_progress_emits_structured_progress(capsys):
    progress = DownloadProgress(total=100, desc="model.safetensors", mininterval=0)
    progress.update(50)
    progress.close()

    out = capsys.readouterr().out
    assert "[D2S_PROGRESS]" in out
    assert '"desc":"model.safetensors"' in out
    assert '"percent":50.0' in out


def test_progress_write_keeps_long_messages_single_line(capsys):
    message = "[Main] Preparing depth model download: lc700x/InfiniDepth-Large/model.safetensors to models. First download may take several minutes."

    progress_write(message)

    assert capsys.readouterr().out == message + "\n"
