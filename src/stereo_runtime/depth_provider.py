from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .depth_upsample import DepthUpsampleMode, upsample_depth
from .output import ensure_bchw, ensure_b1hw, match_depth

DISTILL_ANY_DEPTH_BASE_NAME = "Distill-Any-Depth-Base"
DISTILL_ANY_DEPTH_BASE_MODEL_ID = "lc700x/Distill-Any-Depth-Base-hf"
DISTILL_ANY_DEPTH_BASE_RESOLUTION = 518
DISTILL_ANY_DEPTH_PATCH_SIZE = 14
INFINIDEPTH_PATCH_SIZE = 16
INFINIDEPTH_ENCODERS = {
    "lc700x/infinidepth-small": "vits16",
    "lc700x/infinidepth-smallplus": "vits16plus",
    "lc700x/infinidepth-base": "vitb16",
    "lc700x/infinidepth-large": "vitl16",
}


@dataclass(frozen=True)
class DepthProviderInfo:
    provider: str
    model_name: str
    model_id: str
    depth_resolution: int
    cache_dir: str
    load_mode: str = "online"
    depth_backend: str = "pytorch_cuda"
    runtime: str = "transformers"
    onnx_path: str | None = None
    execution_provider: str | None = None
    fallback_reason: str | None = None
    io_binding: bool = False
    dlpack: bool = False
    output_device: str | None = None
    trt_lib_dirs: list[str] | None = None

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthProfileResult:
    depth: torch.Tensor
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.model_ms + self.postprocess_ms

    def to_report(self) -> dict[str, float]:
        return {
            "preprocess_ms": float(self.preprocess_ms),
            "model_ms": float(self.model_ms),
            "postprocess_ms": float(self.postprocess_ms),
            "total_ms": float(self.total_ms),
        }


@dataclass(frozen=True)
class DepthProviderConfig:
    backend: str = "distill_base_nvidia"
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID
    model_name: str = DISTILL_ANY_DEPTH_BASE_NAME
    depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION
    patch_size: int = DISTILL_ANY_DEPTH_PATCH_SIZE
    device: str | torch.device = "cuda"
    cache_dir: str | Path | None = None
    onnx_path: str | Path | None = None
    trt_cache_dir: str | Path | None = None
    engine_path: str | Path | None = None
    local_files_only: bool = True
    force_download: bool = False
    prefer_tensorrt: bool = True
    prefer_native_tensorrt: bool = False
    prefer_onnx: bool = True
    allow_pytorch_fallback: bool = True
    require_tensorrt: bool = False
    use_iobinding: bool = True
    use_dlpack: bool = False
    build_engine: bool = False
    force_rebuild: bool = False
    use_cuda_graph: bool = False
    profile_sync: bool = False
    depth_upsample: DepthUpsampleMode = "bilinear"
    depth_upsample_edge_strength: float = 0.35


def default_lab_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "models"


def distill_base_518_info(cache_dir: str | Path | None = None) -> DepthProviderInfo:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return DepthProviderInfo(
        provider="transformers.AutoModelForDepthEstimation",
        model_name=DISTILL_ANY_DEPTH_BASE_NAME,
        model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        cache_dir=str(cache),
    )


def _nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def _model_input_size(height: int, width: int, target: int, patch: int) -> tuple[int, int]:
    longest = max(height, width)
    scale = target / float(longest) if longest != target else 1.0
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    return _nearest_multiple(resized_h, patch), _nearest_multiple(resized_w, patch)


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    flat = depth.flatten(start_dim=2)
    amin = flat.amin(dim=-1).view(depth.shape[0], 1, 1, 1)
    amax = flat.amax(dim=-1).view(depth.shape[0], 1, 1, 1)
    return ((depth - amin) / (amax - amin).clamp_min(1e-6)).clamp(0, 1)


def _is_infinidepth_model(model_id: str) -> bool:
    return "infinidepth" in str(model_id).lower()


def _infinidepth_encoder_for_model(model_id: str) -> str:
    return INFINIDEPTH_ENCODERS.get(str(model_id).strip().lower(), "vitl16")


def _resolve_hf_model_file(
    model_id: str,
    cache_dir: str | Path,
    *,
    local_files_only: bool = False,
    force_download: bool = False,
) -> str:
    from huggingface_hub import hf_hub_download

    for filename in ("model.safetensors", "model.pt", "model.ckpt"):
        try:
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
                force_download=force_download,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"unable to resolve InfiniDepth weights for {model_id!r}") from last_error


class DistillAnyDepthBase518:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="transformers.AutoModelForDepthEstimation",
            model_name=DISTILL_ANY_DEPTH_BASE_NAME,
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="transformers",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForDepthEstimation

        kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": self.dtype,
            "weights_only": True,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        model = AutoModelForDepthEstimation.from_pretrained(
            DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            **kwargs,
        )

        self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            DISTILL_ANY_DEPTH_PATCH_SIZE,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        sync()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            predicted = model(pixel_values=tensor).predicted_depth
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted)
        depth = _normalize_depth(depth)
        depth = upsample_depth(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        sync()
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)


class GenericAutoDepthProvider:
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        patch_size: int | None = DISTILL_ANY_DEPTH_PATCH_SIZE,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id.rsplit("/", 1)[-1].replace("-hf", "")
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.depth_resolution = int(depth_resolution)
        self.patch_size = patch_size
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="transformers.AutoModelForDepthEstimation",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="transformers-generic",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForDepthEstimation

        kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": self.dtype,
            "weights_only": True,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            **kwargs,
        )
        self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            self.depth_resolution,
            self.patch_size or 1,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)

        mean, std = _normalization_tensors_for_model(self.model_id, self.device, self.dtype)
        tensor = (tensor - mean) / std
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        sync()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            output = model(pixel_values=tensor)
        predicted = _extract_depth_output(output)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted)
        depth = _postprocess_generic_depth(depth, self.model_id)
        depth = upsample_depth(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        sync()
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)


class InfiniDepthProvider:
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id.rsplit("/", 1)[-1]
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.depth_resolution = int(depth_resolution)
        self.encoder = _infinidepth_encoder_for_model(model_id)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="models.InfiniDepth.api.InfiniDepthModel",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="infinidepth",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from models.InfiniDepth.api import InfiniDepthModel

        model_path = _resolve_hf_model_file(
            self.model_id,
            self.cache_dir,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        model = InfiniDepthModel(model_path=model_path, encoder=self.encoder)
        self._model = model.to(self.device, dtype=self.dtype).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            self.depth_resolution,
            INFINIDEPTH_PATCH_SIZE,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        sync()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            predicted = model.predict_depth(tensor, fp32=self.dtype != torch.float16)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted)
        depth = _normalize_depth(depth)
        depth = upsample_depth(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        sync()
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)


TorchDepthProvider = DistillAnyDepthBase518
GenericTorchDepthProvider = GenericAutoDepthProvider
InfiniDepthTorchProvider = InfiniDepthProvider


def _normalization_tensors_for_model(model_id: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    model_lower = model_id.lower()
    if any(key in model_lower for key in ("depthpro", "zoedepth", "dpt")):
        mean_values = [0.5, 0.5, 0.5]
        std_values = [0.5, 0.5, 0.5]
    else:
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _extract_depth_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, dict) and "predicted_depth" in output:
        return output["predicted_depth"]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"unsupported model output type: {type(output).__name__}")


def _is_metric_model(model_id: str) -> bool:
    model_lower = model_id.lower()
    return any(key in model_lower for key in ("metric", "kitti", "nyu", "depth-ai", "da3"))


def _postprocess_generic_depth(depth: torch.Tensor, model_id: str) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if _is_metric_model(model_id):
        depth = depth.clamp_min(5e-3).reciprocal()
    return _normalize_depth(depth)


def _prepare_accelerated_artifacts(cfg: DepthProviderConfig, *, build_trt: bool = False):
    from .model_artifacts import prepare_model_artifacts

    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir is not None else default_lab_cache_dir()
    model_dir = Path(cfg.onnx_path).parent if cfg.onnx_path is not None else None
    result = prepare_model_artifacts(
        cfg.model_id,
        cache_dir=cache_dir,
        model_dir=model_dir,
        local_files_only=cfg.local_files_only,
        force_download=cfg.force_download,
        download_if_missing=not cfg.local_files_only,
        export_onnx_if_missing=True,
        build_trt_if_missing=build_trt,
        force_rebuild_trt=cfg.force_rebuild,
    )
    return result


def create_depth_provider(config: DepthProviderConfig | dict[str, Any] | None = None):
    cfg = config if isinstance(config, DepthProviderConfig) else DepthProviderConfig(**(config or {}))
    backend = cfg.backend
    device = torch.device(cfg.device)

    if backend in {"migraphx_rocm", "rocm_migraphx", "migraphx"}:
        from .providers.amd import create_migraphx_rocm_provider

        return create_migraphx_rocm_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=cfg.onnx_path,
            graph_path=cfg.engine_path,
            build_graph=cfg.build_engine,
            force_rebuild=cfg.force_rebuild,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            allow_pytorch_fallback=cfg.allow_pytorch_fallback,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_rocm", "rocm", "amd_rocm"}:
        from .providers.amd import create_pytorch_rocm_provider

        return create_pytorch_rocm_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_xpu", "xpu", "intel_xpu"}:
        from .providers.intel import create_pytorch_xpu_provider

        return create_pytorch_xpu_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_mps", "mps", "apple_mps"}:
        from .providers.apple import create_pytorch_mps_provider

        return create_pytorch_mps_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"tensorrt_native", "native_tensorrt", "tensorrt_native_graph"} or (
        backend in {"distill_base_nvidia", "nvidia_chain"} and cfg.prefer_native_tensorrt
    ):
        from .providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

        artifacts = _prepare_accelerated_artifacts(cfg, build_trt=cfg.build_engine or cfg.force_rebuild)
        onnx_path = artifacts.selected_onnx_path or cfg.onnx_path
        engine_path = artifacts.paths.trt_fp16_path if cfg.engine_path is None else cfg.engine_path
        return NativeTensorRtDepthProvider(
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=onnx_path,
            engine_path=engine_path,
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            build_engine=cfg.build_engine,
            force_rebuild=cfg.force_rebuild,
            use_cuda_graph=cfg.use_cuda_graph or backend == "tensorrt_native_graph",
            profile_sync=cfg.profile_sync,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"distill_base_nvidia", "nvidia_chain", "tensorrt", "tensorrt_ort"} and cfg.prefer_tensorrt:
        from .providers.nvidia.tensorrt_ort import TensorRtOrtDepthProvider

        artifacts = _prepare_accelerated_artifacts(cfg)
        return TensorRtOrtDepthProvider(
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=artifacts.selected_onnx_path or cfg.onnx_path,
            trt_cache_dir=cfg.trt_cache_dir,
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"distill_base_nvidia", "nvidia_chain", "onnx_cuda", "onnx_cuda_iobinding"} and cfg.prefer_onnx:
        from .providers.nvidia.onnx_cuda import OnnxCudaDepthProvider

        artifacts = _prepare_accelerated_artifacts(cfg)
        return OnnxCudaDepthProvider(
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=artifacts.selected_onnx_path or cfg.onnx_path,
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            use_iobinding=cfg.use_iobinding,
            use_dlpack=cfg.use_dlpack,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"distill_base_518", "pytorch_cuda", "pytorch"}:
        if cfg.model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
            if _is_infinidepth_model(cfg.model_id):
                return InfiniDepthProvider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                )
            return GenericAutoDepthProvider(
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                device=device,
                cache_dir=cfg.cache_dir,
                depth_resolution=cfg.depth_resolution,
                patch_size=cfg.patch_size,
                local_files_only=cfg.local_files_only,
                force_download=cfg.force_download,
                depth_upsample=cfg.depth_upsample,
                depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
            )
        return DistillAnyDepthBase518(
            device=device,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    raise ValueError(f"unknown depth backend: {backend}")


def estimate_depth(
    rgb: torch.Tensor,
    config: DepthProviderConfig | dict[str, Any] | None = None,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    provider = create_depth_provider(config)
    depth = provider.predict(rgb)
    return depth, provider.info


def estimate_distill_any_depth_base_518(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    provider = DistillAnyDepthBase518(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    return provider.predict(rgb), provider.info
