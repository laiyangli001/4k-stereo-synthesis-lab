from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .model_artifacts import artifact_paths_for_model
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from .depth_provider import DepthProviderConfig
    from .synthesis import StereoConfig

RuntimeMode = Literal["auto", "movie", "game", "image", "debug"]
StereoQuality = Literal["fast", "fast_plus", "quality_4k", "hq_4k"]
DepthBackend = Literal["auto", "tensorrt_native", "onnx_cuda", "pytorch_cuda", "pytorch_rocm", "migraphx_rocm"]
OnnxDtypeMode = Literal["auto", "fp16", "fp32"]
DepthUpsampleMode = Literal["bilinear", "guided"]
OutputFormat = Literal["half_sbs", "full_sbs", "half_tab", "full_tab", "mono", "anaglyph", "interleaved", "leia", "depth_map"]
HoleFill = Literal["none", "fast", "edge_aware"]


@dataclass(frozen=True)
class StereoRuntimeConfig:
    """Host-facing runtime config.

    The host owns user selection and settings persistence. stereo_runtime owns
    model registry resolution, model_dir derivation, download, artifact paths,
    ONNX/TensorRT preparation, and depth/stereo inference.
    """

    model_id: str
    model_dir: str | Path | None = None
    cache_dir: str | Path = "./models"
    mode: RuntimeMode = "movie"
    stereo_preset: str | None = None
    stereo_quality: StereoQuality = "quality_4k"
    output_format: OutputFormat = "half_sbs"
    depth_backend: DepthBackend = "auto"
    device: str = "cuda"
    onnx_dtype: OnnxDtypeMode = "auto"
    export_height: int = 294
    export_width: int = 518
    build_trt_engine: bool = False
    force_rebuild_trt: bool = False
    build_migraphx_graph: bool = False
    force_rebuild_migraphx: bool = False
    trt_workspace_gb: int = 4
    use_cuda_graph: bool = False
    profile_sync: bool = False
    depth_upsample: DepthUpsampleMode = "bilinear"
    depth_upsample_edge_strength: float = 0.35
    depth_strength: float = 2.0
    convergence: float = 0.0
    ipd: float = 0.064
    max_shift_ratio: float = 0.05
    ipd_mm: float | None = 32.0
    stereo_scale: float = 0.4
    layers: int = 2
    occlusion: bool = True
    symmetric: bool = True
    hole_fill: HoleFill = "edge_aware"
    temporal: bool = True
    temporal_strength: float = 0.75
    auto_reset_temporal: bool = True
    scene_reset_threshold: float = 0.22
    reset_cooldown_frames: int = 3
    foreground_scale: float = 0.0
    depth_antialias_strength: float = 0.0
    edge_threshold: float = 0.04
    edge_dilation: int = 2
    mask_feather_radius: int = 3
    hole_fill_mode: str = "balanced"
    hole_fill_radius: int = 3
    hole_fill_strength: float = 1.0
    screen_edge_mask_suppression: int = 0
    cross_eyed: bool = False
    anaglyph_method: str = "red_cyan"
    debug_output: bool = False
    fused: bool = True

    @property
    def resolved_model_id(self) -> str:
        return ModelRegistry.default().resolve_model_id(self.model_id)

    @property
    def model_path(self) -> Path:
        if self.model_dir:
            return Path(self.model_dir)
        return self._artifact_paths().model_dir

    @property
    def onnx_path(self) -> Path:
        return self._artifact_paths().onnx_fp16_path

    @property
    def fp32_onnx_path(self) -> Path:
        return self._artifact_paths().onnx_fp32_path

    @property
    def trt_engine_path(self) -> Path:
        return self._artifact_paths().trt_fp16_path

    @property
    def migraphx_graph_path(self) -> Path:
        return self._artifact_paths().migraphx_fp16_path

    def _artifact_paths(self):
        return artifact_paths_for_model(
            self.resolved_model_id,
            cache_dir=self.cache_dir,
            model_dir=self.model_dir,
            export_height=self.export_height,
            export_width=self.export_width,
        )

    def artifact_paths(self) -> dict[str, str]:
        return {
            "model_dir": str(self.model_path),
            "onnx_path": str(self.onnx_path),
            "fp32_onnx_path": str(self.fp32_onnx_path),
            "trt_engine_path": str(self.trt_engine_path),
            "migraphx_graph_path": str(self.migraphx_graph_path),
        }

    def frame_contract(self) -> dict[str, str]:
        return runtime_frame_contract(self)

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["resolved_model_id"] = self.resolved_model_id
        report["model_dir"] = str(self.model_path)
        report.update(self.artifact_paths())
        report["frame_contract"] = self.frame_contract()
        return report


def runtime_frame_contract(config: StereoRuntimeConfig) -> dict[str, str]:
    """Return the host-facing RGB frame contract.

    The host/capture pipeline owns capture-side color preprocessing and passes
    an already-RGB image frame. stereo_runtime starts at depth-provider input
    preparation and does not own BGR/BGRA-to-RGB conversion.
    """

    return {
        "input": "rgb_frame",
        "host_responsibility": "capture current image frame, perform capture-side color preprocessing, and pass an RGB frame at source resolution",
        "stereo_runtime_responsibility": "prepare RGB frame for depth inference, run depth provider, and synthesize stereo output",
        "not_stereo_runtime_responsibility": "desktop capture, BGR/BGRA-to-RGB conversion, window/monitor source handling",
        "backend_detail": "TensorRT/ONNX/PyTorch/Triton packing remains internal to stereo_runtime",
        "quality_rule": "host must not downscale or alter depth inference resolution semantics",
    }


def runtime_config_from_d2s_settings(
    settings: dict[str, Any],
    *,
    cache_dir: str | Path = "./models",
    device: str = "cuda",
    depth_only: bool = True,
) -> StereoRuntimeConfig:
    """Build a runtime config from Desktop2Stereo-style settings.

    GUI/settings keep user-facing choices. Runtime owns model resolution,
    artifact paths, dtype probing, and backend/provider details.
    """

    model_name = settings.get("Depth Model") or settings.get("model_id")
    if not model_name:
        raise ValueError("D2S settings must include 'Depth Model' or 'model_id'")

    depth_backend: DepthBackend
    if settings.get("MIGraphX", False):
        depth_backend = "migraphx_rocm"
    elif settings.get("TensorRT", False):
        depth_backend = "tensorrt_native"
    elif settings.get("ONNX", False):
        depth_backend = "onnx_cuda"
    elif settings.get("Depth Backend"):
        depth_backend = _normalize_depth_backend(settings["Depth Backend"])
    else:
        depth_backend = "pytorch_cuda"

    onnx_dtype: OnnxDtypeMode = "auto"
    preset_value = settings.get("Stereo Preset", settings.get("Stereo Mode Preset"))
    mode_source = settings.get("Stereo Runtime Mode", settings.get("Run Mode"))
    if mode_source is None:
        mode_source = preset_value if preset_value is not None else "movie"
    mode = _normalize_runtime_mode(mode_source)
    output_format = _normalize_output_format(settings.get("Display Mode", "half_sbs"))
    stereo_quality = _normalize_stereo_quality(settings.get("Stereo Quality", settings.get("Synthetic View", "fast" if depth_only else "quality_4k")))
    ipd_mm = _normalize_ipd_mm(settings)

    has_hole_fill_mode = "Hole Fill Mode" in settings
    hole_fill_mode, hole_fill_radius, hole_fill_strength = _normalize_hole_fill_mode(
        settings.get("Hole Fill Mode", settings.get("Hole Fill", "balanced"))
    )
    if not has_hole_fill_mode:
        hole_fill_radius = int(settings.get("Hole Fill Radius", hole_fill_radius))
        hole_fill_strength = float(settings.get("Hole Fill Strength", hole_fill_strength))

    return StereoRuntimeConfig(
        model_id=str(model_name),
        cache_dir=cache_dir,
        mode=mode,
        stereo_preset=str(preset_value) if preset_value is not None else None,
        stereo_quality=stereo_quality,
        output_format=output_format,
        depth_backend=depth_backend,
        device=device,
        onnx_dtype=onnx_dtype,
        build_trt_engine=bool(settings.get("TensorRT", False)),
        force_rebuild_trt=bool(settings.get("Recompile TensorRT", False)),
        build_migraphx_graph=bool(settings.get("MIGraphX", False)),
        force_rebuild_migraphx=bool(settings.get("Recompile MIGraphX", False)),
        depth_strength=float(settings.get("Depth Strength", 2.0)),
        convergence=float(settings.get("Convergence", 0.0)),
        ipd=ipd_mm / 1000.0,
        max_shift_ratio=float(settings.get("Max Shift Ratio", 0.05)),
        ipd_mm=ipd_mm,
        stereo_scale=float(settings.get("Stereo Scale", settings.get("Stereo Strength Scale", 0.4))),
        temporal=_to_bool(settings.get("Temporal", True)),
        temporal_strength=float(settings.get("Temporal Strength", 0.75)),
        auto_reset_temporal=_to_bool(settings.get("Auto Scene Reset", settings.get("Auto Reset Temporal", True))),
        scene_reset_threshold=float(settings.get("Scene Reset Threshold", 0.22)),
        reset_cooldown_frames=int(settings.get("Reset Cooldown Frames", 3)),
        foreground_scale=float(settings.get("Foreground Scale", 0.0)),
        depth_antialias_strength=float(settings.get("Depth Antialias Strength", settings.get("Anti-aliasing", 0.0))),
        edge_threshold=float(settings.get("Edge Threshold", 0.04)),
        edge_dilation=int(settings.get("Edge Dilation", 2)),
        mask_feather_radius=int(settings.get("Mask Feather Radius", 3)),
        hole_fill_mode=hole_fill_mode,
        hole_fill_radius=hole_fill_radius,
        hole_fill_strength=hole_fill_strength,
        screen_edge_mask_suppression=int(settings.get("Screen Edge Mask Suppression", 0)),
        cross_eyed=_to_bool(settings.get("Cross Eyed", False)),
        anaglyph_method=str(settings.get("Anaglyph Method", "red_cyan")),
        debug_output=_to_bool(settings.get("Debug Stereo Output", False)),
        profile_sync=_to_bool(settings.get("Depth Profile Sync", settings.get("Profile Sync", False))),
    )


def _normalize_ipd_mm(settings: dict[str, Any]) -> float:
    raw = settings.get("IPD mm", settings.get("IPD (mm)", settings.get("IPD", 0.032)))
    value = float(raw)
    # Legacy Desktop2Stereo settings stored IPD in meters, e.g. 0.064.
    if value <= 1.0:
        value *= 1000.0
    return max(1.0, value)

def _normalize_depth_backend(value: Any) -> DepthBackend:
    key = str(value).strip().lower().replace("-", "_")
    mapping: dict[str, DepthBackend] = {
        "auto": "auto",
        "tensorrt": "tensorrt_native",
        "tensorrt_native": "tensorrt_native",
        "trt": "tensorrt_native",
        "onnx": "onnx_cuda",
        "onnx_cuda": "onnx_cuda",
        "onnx_cuda_iobinding": "onnx_cuda",
        "pytorch": "pytorch_cuda",
        "pytorch_cuda": "pytorch_cuda",
        "pytorch_rocm": "pytorch_rocm",
        "rocm": "pytorch_rocm",
        "amd_rocm": "pytorch_rocm",
        "migraphx": "migraphx_rocm",
        "migraphx_rocm": "migraphx_rocm",
        "rocm_migraphx": "migraphx_rocm",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unknown depth backend: {value!r}") from exc


def _normalize_runtime_mode(value: Any) -> RuntimeMode:
    key = "_".join(part for part in str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_").split("_") if part)
    if key in {"auto"}:
        return "auto"
    if key in {"game", "game_low_latency"}:
        return "game"
    if key in {"image", "still", "still_image", "still_image_hq"}:
        return "image"
    if key in {"debug", "debug_export"}:
        return "debug"
    return "movie"


def _normalize_stereo_quality(value: Any) -> StereoQuality:
    key = str(value).strip().lower().replace("-", "_").replace("+", "_plus").replace(" ", "_")
    mapping: dict[str, StereoQuality] = {
        "fast": "fast",
        "fast_plus": "fast_plus",
        "fastplus": "fast_plus",
        "quality": "quality_4k",
        "quality_4k": "quality_4k",
        "hq": "hq_4k",
        "hq_4k": "hq_4k",
    }
    return mapping.get(key, "quality_4k")


def _normalize_hole_fill_mode(value: Any) -> tuple[str, int, float]:
    key = _normalize_hole_fill_mode_key(value)
    mapping = {
        "balanced": ("balanced", 3, 1.0),
        "soft": ("soft_low_ghost", 1, 0.6),
        "low_ghost": ("soft_low_ghost", 1, 0.6),
        "soft_low_ghost": ("soft_low_ghost", 1, 0.6),
        "sharp": ("sharp_test", 1, 1.0),
        "sharp_test": ("sharp_test", 1, 1.0),
    }
    return mapping.get(key, ("balanced", 3, 1.0))


def _normalize_hole_fill_mode_key(value: Any) -> str:
    text = str(value or "balanced").strip().lower()
    for old, new in (
        ("低重影", "low ghost"),
        ("柔和", "soft"),
        ("均衡", "balanced"),
        ("锐利测试", "sharp test"),
    ):
        text = text.replace(old, new)
    for ch in ("-", "/", "+", "\\", "|", "(", ")", "[", "]"):
        text = text.replace(ch, " ")
    return "_".join(part for part in text.split() if part)


def _normalize_output_format(value: Any) -> OutputFormat:
    key = "_".join(
        part
        for part in str(value).strip().lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace("+", "_")
        .split("_")
        if part
    )
    key = "_".join(part for part in key.replace(" ", "_").split("_") if part)
    mapping: dict[str, OutputFormat] = {
        "half_sbs": "half_sbs",
        "half_side_by_side": "half_sbs",
        "sbs": "half_sbs",
        "side_by_side": "half_sbs",
        "full_sbs": "full_sbs",
        "full_side_by_side": "full_sbs",
        "half_tab": "half_tab",
        "half_top_bottom": "half_tab",
        "tab": "half_tab",
        "top_bottom": "half_tab",
        "full_tab": "full_tab",
        "full_top_bottom": "full_tab",
        "mono": "mono",
        "anaglyph": "anaglyph",
        "interleaved": "interleaved",
        "leia": "leia",
        "depth_map": "depth_map",
    }
    return mapping.get(key, "half_sbs")


def preset_for_runtime_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    mapping = {
        "auto": "auto",
        "traditional": "traditional_fastest",
        "traditional_fastest": "traditional_fastest",
        "movie": "cinema",
        "cinema": "cinema",
        "video": "cinema",
        "game": "game_low_latency",
        "game_low_latency": "game_low_latency",
        "image": "still_image_hq",
        "still": "still_image_hq",
        "still_image": "still_image_hq",
        "still_image_hq": "still_image_hq",
        "debug": "debug_export",
        "debug_export": "debug_export",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unknown runtime mode: {mode!r}") from exc


def depth_provider_config_from_runtime(config: StereoRuntimeConfig) -> "DepthProviderConfig":
    from .depth_provider import DepthProviderConfig

    backend = config.depth_backend
    if backend == "auto":
        backend = "tensorrt_native"
    if backend == "onnx_cuda":
        backend = "onnx_cuda_iobinding"
    if backend == "pytorch_cuda":
        backend = "pytorch_cuda"
    onnx_path = config.onnx_path if backend == "migraphx_rocm" else None
    engine_path = config.migraphx_graph_path if backend == "migraphx_rocm" else None
    build_engine = config.build_migraphx_graph if backend == "migraphx_rocm" else config.build_trt_engine
    force_rebuild = config.force_rebuild_migraphx if backend == "migraphx_rocm" else config.force_rebuild_trt

    return DepthProviderConfig(
        backend=backend,
        model_id=config.resolved_model_id,
        model_name=ModelRegistry.default().get(config.resolved_model_id).name,
        device=config.device,
        cache_dir=config.model_path.parent,
        onnx_path=onnx_path,
        engine_path=engine_path,
        local_files_only=False,
        prefer_native_tensorrt=backend == "tensorrt_native",
        prefer_tensorrt=backend == "tensorrt_native",
        prefer_onnx=backend == "onnx_cuda_iobinding",
        use_iobinding=True,
        use_dlpack=backend == "onnx_cuda_iobinding",
        build_engine=build_engine,
        force_rebuild=force_rebuild,
        use_cuda_graph=config.use_cuda_graph,
        profile_sync=config.profile_sync,
        depth_upsample=config.depth_upsample,
        depth_upsample_edge_strength=config.depth_upsample_edge_strength,
    )


def stereo_config_from_runtime(config: StereoRuntimeConfig) -> "StereoConfig":
    from .presets import normalize_preset, stereo_config_for_preset

    preset = normalize_preset(config.stereo_preset or preset_for_runtime_mode(config.mode))
    layers = config.layers
    if config.stereo_quality == "hq_4k" and layers < 3:
        layers = 3

    temporal = config.temporal
    temporal_strength = config.temporal_strength
    auto_reset_temporal = config.auto_reset_temporal
    scene_reset_threshold = config.scene_reset_threshold
    reset_cooldown_frames = config.reset_cooldown_frames
    foreground_scale = config.foreground_scale
    depth_antialias_strength = config.depth_antialias_strength
    if config.stereo_quality == "fast":
        temporal = False
        temporal_strength = 0.0
        auto_reset_temporal = False
        scene_reset_threshold = 0.0
        reset_cooldown_frames = 0
        foreground_scale = 0.0
        depth_antialias_strength = 0.0

    return stereo_config_for_preset(
        preset,
        output_format=config.output_format,
        overrides={
            "backend": config.stereo_quality,
            "layers": layers,
            "occlusion": config.occlusion,
            "symmetric": config.symmetric,
            "hole_fill": config.hole_fill,
            "temporal": temporal,
            "depth_strength": config.depth_strength,
            "convergence": config.convergence,
            "ipd": config.ipd,
            "max_shift_ratio": config.max_shift_ratio,
            "ipd_mm": config.ipd_mm,
            "stereo_scale": config.stereo_scale,
            "temporal_strength": temporal_strength,
            "auto_reset_temporal": auto_reset_temporal,
            "scene_reset_threshold": scene_reset_threshold,
            "reset_cooldown_frames": reset_cooldown_frames,
            "foreground_scale": foreground_scale,
            "depth_antialias_strength": depth_antialias_strength,
            "edge_threshold": config.edge_threshold,
            "edge_dilation": config.edge_dilation,
            "mask_feather_radius": config.mask_feather_radius,
            "hole_fill_mode": config.hole_fill_mode,
            "hole_fill_radius": config.hole_fill_radius,
            "hole_fill_strength": config.hole_fill_strength,
            "screen_edge_mask_suppression": config.screen_edge_mask_suppression,
            "cross_eyed": config.cross_eyed,
            "anaglyph_method": config.anaglyph_method,
            "debug_output": config.debug_output,
            "fused": config.fused,
        },
    )


StereoLabRuntimeConfig = StereoRuntimeConfig
DepthRuntimeConfig = StereoRuntimeConfig
StereoLabDepthRuntimeConfig = StereoRuntimeConfig

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"auto", "none", ""}:
        return None
    return key in {"1", "true", "yes", "on"}
