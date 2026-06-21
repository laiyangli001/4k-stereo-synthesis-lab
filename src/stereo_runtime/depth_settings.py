from __future__ import annotations

from dataclasses import dataclass

from .model_capabilities import model_name_mapping


@dataclass(frozen=True)
class DepthSettings:
    model: str
    model_id: str
    all_models: dict
    cache_path: str
    depth_resolution: int
    device_id: int
    fp16: bool
    foreground_scale: float
    aa_strength: float
    use_torch_compile: bool
    use_tensorrt: bool
    recompile_trt: bool
    use_migraphx: bool
    recompile_migraphx: bool
    use_coreml: bool
    recompile_coreml: bool
    use_openvino: bool
    recompile_openvino: bool


def resolve_depth_settings(settings: dict, *, cache_path: str = "./models") -> DepthSettings:
    mapping = model_name_mapping()
    model = settings["Depth Model"]
    return DepthSettings(
        model=model,
        model_id=mapping[model],
        all_models=settings["Model List"],
        cache_path=cache_path,
        depth_resolution=settings["Depth Resolution"],
        device_id=settings["Computing Device"],
        fp16=settings["FP16"],
        foreground_scale=settings["Foreground Scale"] / 10,
        aa_strength=settings["Anti-aliasing"] * 2,
        use_torch_compile=settings["torch.compile"],
        use_tensorrt=settings["TensorRT"],
        recompile_trt=settings["Recompile TensorRT"],
        use_migraphx=settings.get("MIGraphX", False),
        recompile_migraphx=settings.get("Recompile MIGraphX", False),
        use_coreml=settings["CoreML"],
        recompile_coreml=settings["Recompile CoreML"],
        use_openvino=settings["OpenVINO"],
        recompile_openvino=settings["Recompile OpenVINO"],
    )
