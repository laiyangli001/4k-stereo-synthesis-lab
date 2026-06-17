from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .depth_provider import DepthProviderConfig
from .output import OutputFormat
from .presets import normalize_preset, stereo_config_for_preset
from .synthesis import HoleFill, StereoConfig

RuntimeMode = Literal["auto", "movie", "game", "image"]
StereoQuality = Literal["fast", "quality_4k", "hq_4k"]
DepthBackend = Literal["auto", "tensorrt_native", "onnx_cuda", "pytorch_cuda"]
OnnxDtypeMode = Literal["auto", "fp16", "fp32"]


@dataclass(frozen=True)
class StereoLabRuntimeConfig:
    """Host-facing runtime config.

    The host owns model selection and download. It passes the downloaded
    model directory here; stereo_lab derives ONNX/TensorRT artifact paths
    inside the same directory and owns export/build/inference after that.
    """

    model_id: str
    model_dir: str | Path
    mode: RuntimeMode = "movie"
    stereo_quality: StereoQuality = "quality_4k"
    output_format: OutputFormat = "half_sbs"
    depth_backend: DepthBackend = "auto"
    device: str = "cuda"
    onnx_dtype: OnnxDtypeMode = "auto"
    export_height: int = 294
    export_width: int = 518
    depth_strength: float = 2.0
    convergence: float = 0.0
    ipd: float = 0.064
    max_shift_ratio: float = 0.05
    layers: int = 2
    occlusion: bool = True
    symmetric: bool = True
    hole_fill: HoleFill = "edge_aware"
    temporal: bool = True
    temporal_strength: float = 0.75
    auto_reset_temporal: bool = True
    edge_threshold: float = 0.04
    edge_dilation: int = 2
    fused: bool = True

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def onnx_path(self) -> Path:
        return self.model_path / f"model_fp16_{self.export_height}x{self.export_width}.onnx"

    @property
    def fp32_onnx_path(self) -> Path:
        return self.model_path / f"model_fp32_{self.export_height}x{self.export_width}.onnx"

    @property
    def trt_engine_path(self) -> Path:
        return self.model_path / f"model_fp16_{self.export_height}x{self.export_width}.trt"

    def artifact_paths(self) -> dict[str, str]:
        return {
            "model_dir": str(self.model_path),
            "onnx_path": str(self.onnx_path),
            "fp32_onnx_path": str(self.fp32_onnx_path),
            "trt_engine_path": str(self.trt_engine_path),
        }

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["model_dir"] = str(self.model_path)
        report.update(self.artifact_paths())
        return report


def preset_for_runtime_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    mapping = {
        "auto": "auto",
        "movie": "cinema",
        "cinema": "cinema",
        "video": "cinema",
        "game": "game_low_latency",
        "game_low_latency": "game_low_latency",
        "image": "still_image_hq",
        "still": "still_image_hq",
        "still_image": "still_image_hq",
        "still_image_hq": "still_image_hq",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unknown runtime mode: {mode!r}") from exc


def depth_provider_config_from_runtime(config: StereoLabRuntimeConfig) -> DepthProviderConfig:
    backend = config.depth_backend
    if backend == "auto":
        backend = "tensorrt_native"
    if backend == "onnx_cuda":
        backend = "onnx_cuda_iobinding"
    if backend == "pytorch_cuda":
        backend = "pytorch_cuda"

    return DepthProviderConfig(
        backend=backend,
        device=config.device,
        cache_dir=config.model_path.parent,
        onnx_path=config.onnx_path,
        engine_path=config.trt_engine_path,
        local_files_only=True,
        prefer_native_tensorrt=backend == "tensorrt_native",
        prefer_tensorrt=backend == "tensorrt_native",
        prefer_onnx=backend == "onnx_cuda_iobinding",
        use_iobinding=True,
        use_dlpack=backend == "onnx_cuda_iobinding",
    )


def stereo_config_from_runtime(config: StereoLabRuntimeConfig) -> StereoConfig:
    preset = normalize_preset(preset_for_runtime_mode(config.mode))
    layers = config.layers
    if config.stereo_quality == "hq_4k" and layers < 3:
        layers = 3

    return stereo_config_for_preset(
        preset,
        output_format=config.output_format,
        overrides={
            "backend": config.stereo_quality,
            "layers": layers,
            "occlusion": config.occlusion,
            "symmetric": config.symmetric,
            "hole_fill": config.hole_fill,
            "temporal": config.temporal,
            "depth_strength": config.depth_strength,
            "convergence": config.convergence,
            "ipd": config.ipd,
            "max_shift_ratio": config.max_shift_ratio,
            "temporal_strength": config.temporal_strength,
            "auto_reset_temporal": config.auto_reset_temporal,
            "edge_threshold": config.edge_threshold,
            "edge_dilation": config.edge_dilation,
            "fused": config.fused,
        },
    )
