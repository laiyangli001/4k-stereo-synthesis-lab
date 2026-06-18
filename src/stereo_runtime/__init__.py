"""4K stereo synthesis research prototypes."""

from .adapter import (
    DepthRuntimeConfig,
    StereoLabDepthRuntimeConfig,
    StereoLabRuntimeConfig,
    StereoRuntimeConfig,
    depth_provider_config_from_runtime,
    preset_for_runtime_mode,
    runtime_frame_contract,
    runtime_config_from_d2s_settings,
    stereo_config_from_runtime,
)
from .model_registry import DepthModelSpec, ModelRegistry, resolve_model_dir
from .model_artifacts import (
    ModelArtifactPaths,
    PreparedModelArtifacts,
    artifact_paths_for_model,
    ensure_model_downloaded,
    ensure_onnx_exported,
    ensure_tensorrt_engine,
    prepare_model_artifacts,
)
from .onnx_export import (
    OnnxExportResult,
    choose_export_dtype,
    export_depth_model_onnx,
    probe_model_dtype,
)

_LAZY_EXPORTS = {
    "DepthRuntime": ("runtime", "DepthRuntime"),
    "DepthRuntimeResult": ("runtime", "DepthRuntimeResult"),
    "StereoLabDepthRuntime": ("runtime", "StereoLabDepthRuntime"),
    "StereoLabDepthRuntimeResult": ("runtime", "StereoLabDepthRuntimeResult"),
    "StereoLabRuntime": ("runtime", "StereoLabRuntime"),
    "StereoLabRuntimeResult": ("runtime", "StereoLabRuntimeResult"),
    "StereoRuntime": ("runtime", "StereoRuntime"),
    "StereoRuntimeResult": ("runtime", "StereoRuntimeResult"),
    "StereoConfig": ("synthesis", "StereoConfig"),
    "StereoResult": ("synthesis", "StereoResult"),
    "synthesize_stereo": ("synthesis", "synthesize_stereo"),
    "DepthSafetyConfig": ("depth_safety", "DepthSafetyConfig"),
    "DepthSafetyDecision": ("depth_safety", "DepthSafetyDecision"),
    "apply_depth_safety": ("depth_safety", "apply_depth_safety"),
    "evaluate_depth_safety": ("depth_safety", "evaluate_depth_safety"),
    "OpenXREyeView": ("openxr_render", "OpenXREyeView"),
    "OpenXRFov": ("openxr_render", "OpenXRFov"),
    "OpenXRRenderConfig": ("openxr_render", "OpenXRRenderConfig"),
    "OpenXRScreenPose": ("openxr_render", "OpenXRScreenPose"),
    "OpenXRStereoResult": ("openxr_render", "OpenXRStereoResult"),
    "OpenXRRuntimeResult": ("runtime", "OpenXRRuntimeResult"),
    "build_openxr_eye_mvp": ("openxr_render", "build_openxr_eye_mvp"),
    "is_pyopenxr_available": ("openxr_render", "is_pyopenxr_available"),
    "render_openxr_stereo": ("openxr_render", "render_openxr_stereo"),
    "AutoModeDecision": ("presets", "AutoModeDecision"),
    "AutoModeRuntime": ("presets", "AutoModeRuntime"),
    "AutoModeRuntimeState": ("presets", "AutoModeRuntimeState"),
    "AutoModeSignals": ("presets", "AutoModeSignals"),
    "PRESET_CHOICES": ("presets", "PRESET_CHOICES"),
    "StereoModePreset": ("presets", "StereoModePreset"),
    "auto_detection_required": ("presets", "auto_detection_required"),
    "auto_mode_scores": ("presets", "auto_mode_scores"),
    "classify_auto_mode": ("presets", "classify_auto_mode"),
    "openxr_config_for_auto_mode": ("presets", "openxr_config_for_auto_mode"),
    "openxr_config_for_preset": ("presets", "openxr_config_for_preset"),
    "preset_summary": ("presets", "preset_summary"),
    "stereo_config_for_auto_mode": ("presets", "stereo_config_for_auto_mode"),
    "stereo_config_for_preset": ("presets", "stereo_config_for_preset"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AutoModeDecision",
    "AutoModeRuntime",
    "AutoModeRuntimeState",
    "AutoModeSignals",
    "DepthModelSpec",
    "DepthRuntime",
    "DepthRuntimeConfig",
    "DepthRuntimeResult",
    "DepthSafetyConfig",
    "DepthSafetyDecision",
    "ModelRegistry",
    "ModelArtifactPaths",
    "OnnxExportResult",
    "OpenXREyeView",
    "OpenXRFov",
    "OpenXRRenderConfig",
    "OpenXRScreenPose",
    "OpenXRRuntimeResult",
    "OpenXRStereoResult",
    "PRESET_CHOICES",
    "PreparedModelArtifacts",
    "StereoConfig",
    "StereoLabDepthRuntime",
    "StereoLabDepthRuntimeConfig",
    "StereoLabDepthRuntimeResult",
    "StereoLabRuntime",
    "StereoLabRuntimeConfig",
    "StereoLabRuntimeResult",
    "StereoModePreset",
    "StereoResult",
    "StereoRuntime",
    "StereoRuntimeConfig",
    "StereoRuntimeResult",
    "apply_depth_safety",
    "artifact_paths_for_model",
    "auto_detection_required",
    "auto_mode_scores",
    "build_openxr_eye_mvp",
    "choose_export_dtype",
    "classify_auto_mode",
    "depth_provider_config_from_runtime",
    "ensure_model_downloaded",
    "ensure_onnx_exported",
    "ensure_tensorrt_engine",
    "evaluate_depth_safety",
    "export_depth_model_onnx",
    "is_pyopenxr_available",
    "openxr_config_for_auto_mode",
    "openxr_config_for_preset",
    "preset_for_runtime_mode",
    "preset_summary",
    "prepare_model_artifacts",
    "probe_model_dtype",
    "render_openxr_stereo",
    "resolve_model_dir",
    "runtime_frame_contract",
    "runtime_config_from_d2s_settings",
    "stereo_config_for_auto_mode",
    "stereo_config_for_preset",
    "stereo_config_from_runtime",
    "synthesize_stereo",
]
