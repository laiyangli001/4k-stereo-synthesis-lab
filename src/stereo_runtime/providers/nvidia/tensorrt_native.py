from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

import torch

from ...depth_onnx_provider import ModelOnnxPreprocessor, default_onnx_path
from ...depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_NAME,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DepthProfileResult,
    DepthProviderInfo,
    _normalize_depth,
    default_lab_cache_dir,
)
from ...depth_upsample import DepthUpsampleMode, upsample_depth
from ...output import ensure_b1hw, ensure_bchw, match_depth
from .tensorrt_ort import ensure_tensorrt_dll_path


def default_distill_base_native_trt_path(cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return cache / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.trt"


def default_native_tensorrt_engine_path(cache_dir: str | Path | None = None) -> Path:
    return default_distill_base_native_trt_path(cache_dir)


def _infer_model_metadata_from_paths(
    onnx_path: Path,
    engine_path: Path,
    *,
    model_id: str,
    model_name: str,
) -> tuple[str, str]:
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID or model_name != DISTILL_ANY_DEPTH_BASE_NAME:
        return model_id, model_name

    for path in (onnx_path, engine_path):
        for part in path.parts:
            if part.startswith("models--") and "Distill-Any-Depth" in part:
                inferred_id = part.removeprefix("models--").replace("--", "/")
                inferred_name = inferred_id.rsplit("/", 1)[-1].replace("-hf", "")
                return inferred_id, inferred_name
    return model_id, model_name


class NativeTensorRtEngine:
    def __init__(self, engine_path: str | Path, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.engine_path = Path(engine_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self._output_buffers: dict[str, torch.Tensor] = {}
        self._graph_input: torch.Tensor | None = None
        self._graph_output_name: str | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        if self.device.type != "cuda":
            raise RuntimeError("Native TensorRT engine requires CUDA")
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        ensure_tensorrt_dll_path()
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with self.engine_path.open("rb") as file:
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if not self.input_names:
            raise RuntimeError("TensorRT engine has no input tensors")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors")

    @property
    def input_shape(self) -> tuple[int, ...]:
        input_name = self.input_names[0]
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(input_name))

    @property
    def input_image_size(self) -> tuple[int, int] | None:
        shape = self.input_shape
        if len(shape) != 4 or any(dim < 1 for dim in shape):
            return None
        return int(shape[-2]), int(shape[-1])

    @staticmethod
    def _torch_dtype_from_trt(dtype) -> torch.dtype:
        import tensorrt as trt

        mapping = {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            mapping[trt.bfloat16] = torch.bfloat16
        return mapping.get(dtype, torch.float32)

    def _bind_input_output(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = tensor.contiguous().to(device=self.device, dtype=self.dtype)
        input_name = self.input_names[0]
        self.context.set_input_shape(input_name, tuple(tensor.shape))
        self.context.set_tensor_address(input_name, tensor.data_ptr())

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            output_dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            output = self._output_buffers.get(name)
            if output is None or tuple(output.shape) != shape or output.dtype != output_dtype:
                output = torch.empty(shape, device=self.device, dtype=output_dtype)
                self._output_buffers[name] = output
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())
        return outputs

    def _execute(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def __call__(self, tensor: torch.Tensor, *, synchronize: bool = True) -> torch.Tensor:
        outputs = self._bind_input_output(tensor)
        self._execute()
        if synchronize:
            stream = torch.cuda.current_stream(self.device)
            stream.synchronize()

        if "predicted_depth" in outputs:
            return outputs["predicted_depth"]
        return outputs[self.output_names[0]]

    def capture_graph(self, input_shape: tuple[int, ...]) -> None:
        if self._graph is not None:
            return
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph requires CUDA")
        static_input = torch.empty(input_shape, device=self.device, dtype=self.dtype)
        outputs = self._bind_input_output(static_input)
        output_name = "predicted_depth" if "predicted_depth" in outputs else self.output_names[0]

        for _ in range(3):
            self._execute()
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._execute()
        torch.cuda.synchronize(self.device)

        self._graph_input = static_input
        self._graph_output_name = output_name
        self._graph = graph

    def run_graph(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._graph is None or self._graph_input is None or self._graph_output_name is None:
            self.capture_graph(tuple(tensor.shape))
        assert self._graph is not None
        assert self._graph_input is not None
        assert self._graph_output_name is not None
        tensor = tensor.contiguous().to(device=self.device, dtype=self.dtype)
        if tuple(tensor.shape) != tuple(self._graph_input.shape):
            raise RuntimeError(f"CUDA graph input shape mismatch: expected {tuple(self._graph_input.shape)}, got {tuple(tensor.shape)}")
        self._graph_input.copy_(tensor)
        self._graph.replay()
        return self._output_buffers[self._graph_output_name]

    def close(self) -> None:
        for attr in ("context", "engine", "runtime"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass


def build_native_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = True,
    workspace_gb: int = 4,
    force: bool = False,
) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    if engine_path.exists() and not force:
        return engine_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    ensure_tensorrt_dll_path()
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)

    input_tensor = network.get_input(0)
    input_shape = tuple(int(dim) for dim in input_tensor.shape)
    if any(dim < 1 for dim in input_shape):
        raise RuntimeError(f"native TensorRT build requires fixed input shape, got {input_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


class DistillAnyDepthBaseNativeTensorRt:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        model_name: str = DISTILL_ANY_DEPTH_BASE_NAME,
        build_engine: bool = False,
        force_rebuild: bool = False,
        use_cuda_graph: bool = False,
        profile_sync: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("Native TensorRT depth provider requires CUDA")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx_path(self.cache_dir)
        self.engine_path = Path(engine_path) if engine_path is not None else default_native_tensorrt_engine_path(self.cache_dir)
        self.model_id, self.model_name = _infer_model_metadata_from_paths(
            self.onnx_path,
            self.engine_path,
            model_id=model_id,
            model_name=model_name,
        )
        self.build_engine = bool(build_engine)
        self.force_rebuild = bool(force_rebuild)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.profile_sync = bool(profile_sync)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.dtype = torch.float16
        self.info = DepthProviderInfo(
            provider="tensorrt.Runtime",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_onnx_native_tensorrt",
            depth_backend="tensorrt_native_graph" if self.use_cuda_graph else "tensorrt_native",
            runtime=("tensorrt-native-cudagraph" if self.use_cuda_graph else "tensorrt-native") + ("-profile-sync" if self.profile_sync else "-async"),
            onnx_path=str(self.onnx_path),
            io_binding=False,
            output_device="cuda",
        )
        self._engine: NativeTensorRtEngine | None = None
        self._preprocessor = ModelOnnxPreprocessor(model_id=self.model_id, device=self.device, dtype=self.dtype)

    def load(self) -> NativeTensorRtEngine:
        if self._engine is not None:
            self._preprocessor.fixed_input_size = self._engine.input_image_size
            return self._engine
        if self.build_engine or self.force_rebuild or not self.engine_path.exists():
            build_native_tensorrt_engine(self.onnx_path, self.engine_path, force=self.force_rebuild)
        trt_lib_dirs = ensure_tensorrt_dll_path()
        self._engine = NativeTensorRtEngine(self.engine_path, device=self.device, dtype=self.dtype)
        self._preprocessor.fixed_input_size = self._engine.input_image_size
        self.info = replace(
            self.info,
            execution_provider="TensorRT native",
            trt_lib_dirs=trt_lib_dirs,
        )
        return self._engine

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        def sync() -> None:
            if self.profile_sync and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        engine = self.load()
        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        tensor = self._preprocessor(rgb)
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        if self.use_cuda_graph:
            predicted = engine.run_graph(tensor)
        else:
            try:
                predicted = engine(tensor, synchronize=self.profile_sync)
            except TypeError:
                predicted = engine(tensor)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted.float())
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


NativeTensorRtDepthProvider = DistillAnyDepthBaseNativeTensorRt
