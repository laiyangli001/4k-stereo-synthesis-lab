from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .model_registry import DepthModelSpec, ModelRegistry, resolve_model_dir

OnnxDtypeMode = Literal["auto", "fp16", "fp32"]


@dataclass(frozen=True)
class ModelArtifactPaths:
    model_id: str
    model_name: str
    model_dir: Path
    onnx_fp16_path: Path
    onnx_fp32_path: Path
    trt_fp16_path: Path
    export_height: int
    export_width: int

    def onnx_path_for_dtype(self, dtype_name: str) -> Path:
        if dtype_name == "fp16":
            return self.onnx_fp16_path
        if dtype_name == "fp32":
            return self.onnx_fp32_path
        raise ValueError(f"unknown ONNX dtype: {dtype_name!r}")

    def to_report(self) -> dict[str, str | int]:
        report = asdict(self)
        report["model_dir"] = str(self.model_dir)
        report["onnx_fp16_path"] = str(self.onnx_fp16_path)
        report["onnx_fp32_path"] = str(self.onnx_fp32_path)
        report["trt_fp16_path"] = str(self.trt_fp16_path)
        return report


@dataclass(frozen=True)
class PreparedModelArtifacts:
    paths: ModelArtifactPaths
    downloaded: bool
    onnx_ready: bool
    trt_ready: bool
    selected_onnx_path: Path | None = None
    notes: tuple[str, ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "paths": self.paths.to_report(),
            "downloaded": self.downloaded,
            "onnx_ready": self.onnx_ready,
            "trt_ready": self.trt_ready,
            "selected_onnx_path": str(self.selected_onnx_path) if self.selected_onnx_path else None,
            "notes": list(self.notes),
        }


def artifact_paths_for_model(
    model_name_or_id: str,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    export_height: int = 294,
    export_width: int = 518,
    registry: ModelRegistry | None = None,
) -> ModelArtifactPaths:
    registry = registry or ModelRegistry.default()
    spec = registry.get(model_name_or_id)
    root = Path(model_dir) if model_dir is not None else resolve_model_dir(spec.model_id, cache_dir)
    return ModelArtifactPaths(
        model_id=spec.model_id,
        model_name=spec.name,
        model_dir=root,
        onnx_fp16_path=root / f"model_fp16_{export_height}x{export_width}.onnx",
        onnx_fp32_path=root / f"model_fp32_{export_height}x{export_width}.onnx",
        trt_fp16_path=root / f"model_fp16_{export_height}x{export_width}.trt",
        export_height=int(export_height),
        export_width=int(export_width),
    )


def ensure_model_downloaded(
    model_spec: DepthModelSpec,
    *,
    cache_dir: str | Path = "./models",
    local_files_only: bool = False,
    force_download: bool = False,
) -> Path:
    """Download or validate a Hugging Face model without importing D2S settings."""
    model_dir = model_spec.model_dir(cache_dir)
    if local_files_only and not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if model_dir.exists() and not force_download:
        return model_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_spec.model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        force_download=force_download,
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"download completed but model directory was not found: {model_dir}")
    return model_dir


def select_existing_onnx(paths: ModelArtifactPaths, dtype: OnnxDtypeMode = "auto") -> Path | None:
    if dtype == "fp16":
        return paths.onnx_fp16_path if paths.onnx_fp16_path.exists() else None
    if dtype == "fp32":
        return paths.onnx_fp32_path if paths.onnx_fp32_path.exists() else None
    if paths.onnx_fp16_path.exists():
        return paths.onnx_fp16_path
    if paths.onnx_fp32_path.exists():
        return paths.onnx_fp32_path
    return None


def ensure_onnx_exported(
    model_spec: DepthModelSpec,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    height: int = 294,
    width: int = 518,
    dtype: OnnxDtypeMode = "auto",
    export_if_missing: bool = False,
) -> Path:
    paths = artifact_paths_for_model(
        model_spec.model_id,
        cache_dir=cache_dir,
        model_dir=model_dir,
        export_height=height,
        export_width=width,
    )
    existing = select_existing_onnx(paths, dtype)
    if existing is not None:
        return existing
    if not export_if_missing:
        raise FileNotFoundError(f"ONNX artifact not found for {model_spec.model_id}: {paths.onnx_fp16_path} or {paths.onnx_fp32_path}")

    from .onnx_export import export_depth_model_onnx

    output_path = paths.onnx_fp16_path if dtype in ("auto", "fp16") else paths.onnx_fp32_path
    result = export_depth_model_onnx(
        model_id=model_spec.model_id,
        output_path=output_path,
        cache_dir=cache_dir,
        height=height,
        width=width,
        dtype=dtype,
        force_download=False,
    )
    return result.output_path


def ensure_tensorrt_engine(
    model_spec: DepthModelSpec,
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    build_if_missing: bool = False,
    force_rebuild: bool = False,
    workspace_gb: int = 4,
) -> Path:
    engine_path = Path(engine_path)
    if engine_path.exists() and not force_rebuild:
        return engine_path
    if not build_if_missing and not force_rebuild:
        raise FileNotFoundError(f"TensorRT engine not found for {model_spec.model_id}: {engine_path}")

    from .depth_trt_native_provider import build_native_tensorrt_engine

    return build_native_tensorrt_engine(
        onnx_path,
        engine_path,
        workspace_gb=workspace_gb,
        force=force_rebuild,
    )


def prepare_model_artifacts(
    model_name_or_id: str,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    export_height: int = 294,
    export_width: int = 518,
    onnx_dtype: OnnxDtypeMode = "auto",
    local_files_only: bool = False,
    force_download: bool = False,
    download_if_missing: bool = False,
    export_onnx_if_missing: bool = False,
    build_trt_if_missing: bool = False,
    force_rebuild_trt: bool = False,
    trt_workspace_gb: int = 4,
) -> PreparedModelArtifacts:
    registry = ModelRegistry.default()
    spec = registry.get(model_name_or_id)
    paths = artifact_paths_for_model(
        spec.model_id,
        cache_dir=cache_dir,
        model_dir=model_dir,
        export_height=export_height,
        export_width=export_width,
        registry=registry,
    )

    notes: list[str] = []
    if download_if_missing or force_download:
        ensure_model_downloaded(
            spec,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    elif local_files_only and not paths.model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {paths.model_dir}")
    downloaded = paths.model_dir.exists()

    selected_onnx = select_existing_onnx(paths, onnx_dtype)
    if selected_onnx is None and export_onnx_if_missing:
        selected_onnx = ensure_onnx_exported(
            spec,
            cache_dir=cache_dir,
            model_dir=paths.model_dir,
            height=export_height,
            width=export_width,
            dtype=onnx_dtype,
            export_if_missing=True,
        )
    elif selected_onnx is None:
        notes.append("onnx artifact missing")

    trt_ready = paths.trt_fp16_path.exists()
    if not trt_ready and build_trt_if_missing and selected_onnx is not None:
        ensure_tensorrt_engine(
            spec,
            onnx_path=selected_onnx,
            engine_path=paths.trt_fp16_path,
            build_if_missing=True,
            force_rebuild=force_rebuild_trt,
            workspace_gb=trt_workspace_gb,
        )
        trt_ready = paths.trt_fp16_path.exists()
    elif not trt_ready:
        notes.append("TensorRT engine missing")

    return PreparedModelArtifacts(
        paths=paths,
        downloaded=downloaded,
        onnx_ready=selected_onnx is not None,
        trt_ready=trt_ready,
        selected_onnx_path=selected_onnx,
        notes=tuple(notes),
    )
