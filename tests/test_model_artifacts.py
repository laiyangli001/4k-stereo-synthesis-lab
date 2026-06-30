from pathlib import Path
from types import SimpleNamespace

import pytest

from stereo_runtime import (
    artifact_paths_for_model,
    prepare_model_artifacts,
)
from stereo_runtime.model_artifacts import ensure_model_downloaded, select_existing_onnx
from stereo_runtime.model_registry import ModelRegistry


def test_artifact_paths_follow_d2s_naming():
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir="models", export_height=294, export_width=518)

    assert paths.model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(paths.model_dir).replace("\\", "/") == "models/models--lc700x--Distill-Any-Depth-Base-hf"
    assert paths.onnx_fp16_path.name == "model_fp16_294x518.onnx"
    assert paths.onnx_fp32_path.name == "model_fp32_294x518.onnx"
    assert paths.trt_fp16_path.name == "model_fp16_294x518.trt"
    assert paths.migraphx_fp16_path.name == "model_fp16_294x518.mgx"


def test_infinidepth_artifact_paths_use_patch_16_export_size():
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir="models", export_height=294, export_width=518)

    assert paths.model_id == "lc700x/InfiniDepth-Base"
    assert paths.onnx_fp16_path.name == "model_fp16_288x512.onnx"
    assert paths.onnx_fp32_path.name == "model_fp32_288x512.onnx"
    assert paths.trt_fp16_path.name == "model_fp16_288x512.trt"


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


def test_prepare_model_artifacts_passes_local_files_only_to_onnx_export(monkeypatch, tmp_path: Path):
    calls = {}
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)

    def fake_export(**kwargs):
        calls.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(paths.model_dir / "snapshots" / "fake"))
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=True,
        export_onnx_if_missing=True,
    )

    assert result.onnx_ready is True
    assert calls["local_files_only"] is True
    assert calls["force_download"] is False


def test_prepare_model_artifacts_confirms_model_before_onnx_export(monkeypatch, tmp_path: Path):
    calls = []
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)

    def fake_snapshot_download(**kwargs):
        calls.append("download")
        paths.model_dir.mkdir(parents=True, exist_ok=True)
        return str(paths.model_dir / "snapshots" / "fake")

    def fake_export(**kwargs):
        calls.append("onnx")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=False,
        download_if_missing=False,
        export_onnx_if_missing=True,
    )

    assert result.onnx_ready is True
    assert calls == ["download", "onnx"]


def test_infinidepth_download_requires_resolved_weight_file(monkeypatch, tmp_path: Path):
    calls = {}
    spec = ModelRegistry.default().get("InfiniDepth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)

    def fake_resolve(model_id, cache_dir, *, local_files_only=False, force_download=False):
        calls["model_id"] = model_id
        calls["cache_dir"] = Path(cache_dir)
        calls["local_files_only"] = local_files_only
        calls["force_download"] = force_download
        return str(model_dir / "model.safetensors")

    import stereo_runtime.depth_provider as depth_provider

    monkeypatch.setattr(depth_provider, "_resolve_hf_model_file", fake_resolve)

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert calls == {
        "model_id": "lc700x/InfiniDepth-Base",
        "cache_dir": tmp_path,
        "local_files_only": False,
        "force_download": False,
    }


def test_generic_download_confirms_snapshot_even_when_cache_dir_exists(monkeypatch, tmp_path: Path):
    calls = {}
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(model_dir / "snapshots" / "fake")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert calls["repo_id"] == "lc700x/Distill-Any-Depth-Base-hf"
    assert calls["cache_dir"] == str(tmp_path)
    assert calls["local_files_only"] is False
    assert calls["force_download"] is False


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

    import stereo_runtime.providers.nvidia.tensorrt_native as native_provider

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
