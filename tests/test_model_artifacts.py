from pathlib import Path

import pytest

from stereo_runtime import (
    artifact_paths_for_model,
    prepare_model_artifacts,
)
from stereo_runtime.model_artifacts import select_existing_onnx


def test_artifact_paths_follow_d2s_naming():
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir="models", export_height=294, export_width=518)

    assert paths.model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(paths.model_dir).replace("\\", "/") == "models/models--lc700x--Distill-Any-Depth-Base-hf"
    assert paths.onnx_fp16_path.name == "model_fp16_294x518.onnx"
    assert paths.onnx_fp32_path.name == "model_fp32_294x518.onnx"
    assert paths.trt_fp16_path.name == "model_fp16_294x518.trt"


def test_select_existing_onnx_prefers_fp16_for_auto(tmp_path: Path):
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp16_path.write_bytes(b"fp16")
    paths.onnx_fp32_path.write_bytes(b"fp32")

    assert select_existing_onnx(paths, "auto") == paths.onnx_fp16_path
    assert select_existing_onnx(paths, "fp32") == paths.onnx_fp32_path


def test_prepare_model_artifacts_reports_missing_without_side_effects(tmp_path: Path):
    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=False,
        download_if_missing=False,
        export_onnx_if_missing=False,
        build_trt_if_missing=False,
    )

    assert result.downloaded is False
    assert result.onnx_ready is False
    assert result.trt_ready is False
    assert "onnx artifact missing" in result.notes
    assert "TensorRT engine missing" in result.notes


def test_prepare_model_artifacts_local_files_only_requires_model_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prepare_model_artifacts("Distill-Any-Depth-Base", cache_dir=tmp_path, local_files_only=True)


def test_prepare_model_artifacts_builds_trt_from_existing_onnx(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_build(onnx_path, engine_path, *, workspace_gb=4, force=False):
        calls["onnx_path"] = Path(onnx_path)
        calls["engine_path"] = Path(engine_path)
        calls["workspace_gb"] = workspace_gb
        calls["force"] = force
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_path).write_bytes(b"trt")
        return Path(engine_path)

    import stereo_runtime.depth_trt_native_provider as native_provider

    monkeypatch.setattr(native_provider, "build_native_tensorrt_engine", fake_build)
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp16_path.write_bytes(b"onnx")

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        build_trt_if_missing=True,
        trt_workspace_gb=7,
    )

    assert result.trt_ready is True
    assert calls["onnx_path"] == paths.onnx_fp16_path
    assert calls["engine_path"] == paths.trt_fp16_path
    assert calls["workspace_gb"] == 7
    assert calls["force"] is False
