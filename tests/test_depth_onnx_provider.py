import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
    TorchDepthProvider,
    create_depth_provider,
    estimate_depth,
)
from stereo_runtime.depth_onnx_provider import DistillPreprocessor, _preprocess_distill_rgb, estimate_depth_onnx_cuda
from stereo_runtime.providers.nvidia.tensorrt_ort import estimate_depth_nvidia_chain


class FakeTorchProvider:
    def __init__(self, **kwargs):
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="Distill-Any-Depth-Base",
            model_id="fake",
            depth_resolution=518,
            cache_dir=str(kwargs.get("cache_dir") or ""),
            load_mode="test",
            depth_backend="pytorch_cpu",
            runtime="fake",
        )

    def predict(self, rgb):
        return torch.zeros(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])


def test_nvidia_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_ort as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_nvidia_chain(
        rgb,
        device="cpu",
        onnx_path=missing,
        prefer_onnx=True,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_nvidia_provider_requires_tensorrt_when_requested(tmp_path):
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    try:
        estimate_depth_nvidia_chain(
            rgb,
            device="cpu",
            onnx_path=missing,
            require_tensorrt=True,
        )
    except RuntimeError as exc:
        assert "tensorrt" in str(exc)
    else:
        raise AssertionError("expected TensorRT requirement to fail")


def test_onnx_cuda_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_onnx_cuda(
        rgb,
        device="cpu",
        onnx_path=missing,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_create_depth_provider_supports_persistent_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_cuda",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBase518)
    assert isinstance(provider, TorchDepthProvider)
    assert provider.info.depth_backend == "pytorch_cpu"


def test_create_depth_provider_supports_generic_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch",
            model_id="Intel/dpt-large",
            model_name="dpt-large",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, GenericAutoDepthProvider)
    assert provider.info.model_id == "Intel/dpt-large"
    assert provider.info.runtime == "transformers-generic"


def test_generic_provider_predict_profile_with_fake_transformers_model(monkeypatch, tmp_path):
    class Output:
        predicted_depth = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForDepthEstimation", FakeAutoModel)
    provider = GenericAutoDepthProvider(
        model_id="Intel/dpt-large",
        model_name="dpt-large",
        device="cpu",
        cache_dir=tmp_path,
        local_files_only=True,
        depth_resolution=8,
        patch_size=1,
    )

    result = provider.predict_profile(torch.zeros(1, 3, 8, 8))

    assert result.depth.shape == (1, 1, 8, 8)
    assert float(result.depth.min()) >= 0.0
    assert float(result.depth.max()) <= 1.0


def test_create_depth_provider_supports_native_tensorrt(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    engine_path = tmp_path / "model.trt"
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="tensorrt_native",
            device="cuda",
            cache_dir=tmp_path,
            engine_path=engine_path,
            build_engine=True,
        )
    )

    assert isinstance(provider, NativeTensorRtDepthProvider)
    assert provider.info.depth_backend == "tensorrt_native"
    assert provider.engine_path == engine_path
    assert provider.build_engine is True


def test_native_tensorrt_infers_large_metadata_from_model_path(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    onnx_path = model_dir / "model_fp16_294x518.onnx"
    engine_path = model_dir / "model_fp16_294x518.trt"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=onnx_path,
        engine_path=engine_path,
    )

    assert provider.info.model_id == "xingyang1/Distill-Any-Depth-Large-hf"
    assert provider.info.model_name == "Distill-Any-Depth-Large"


def test_native_tensorrt_keeps_explicit_metadata(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=model_dir / "model_fp16_294x518.onnx",
        engine_path=model_dir / "model_fp16_294x518.trt",
        model_id="custom/model",
        model_name="Custom Model",
    )

    assert provider.info.model_id == "custom/model"
    assert provider.info.model_name == "Custom Model"


def test_estimate_depth_uses_configured_provider(monkeypatch):
    class Provider:
        info = DepthProviderInfo(
            provider="fake",
            model_name="fake",
            model_id="fake",
            depth_resolution=518,
            cache_dir="",
            load_mode="test",
            depth_backend="fake",
            runtime="test",
        )

        def predict(self, rgb):
            return torch.ones(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])

    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "create_depth_provider", lambda config=None: Provider())
    depth, info = estimate_depth(torch.zeros(1, 3, 4, 5), {"backend": "fake"})

    assert depth.shape == (1, 1, 4, 5)
    assert info.depth_backend == "fake"


def test_distill_preprocessor_matches_reference():
    rgb = torch.linspace(0, 1, steps=3 * 12 * 16, dtype=torch.float32).view(1, 3, 12, 16)
    device = torch.device("cpu")
    dtype = torch.float32

    expected = _preprocess_distill_rgb(rgb, device=device, dtype=dtype)
    actual = DistillPreprocessor(device=device, dtype=dtype)(rgb)

    assert torch.equal(actual, expected)

def test_distill_preprocessor_can_use_fixed_tensorrt_input_size():
    rgb = torch.zeros(1, 3, 2160, 1920)
    preprocessor = DistillPreprocessor(
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_input_size=(294, 518),
    )

    tensor = preprocessor(rgb)

    assert tensor.shape == (1, 3, 294, 518)
    assert preprocessor.input_size(2160, 1920) == (294, 518)


def test_native_tensorrt_provider_uses_engine_static_input_size(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    captured = {}

    class FakeEngine:
        input_image_size = (294, 518)

        def __call__(self, tensor):
            captured["tensor_shape"] = tuple(tensor.shape)
            return torch.zeros(1, 1, 294, 518, dtype=torch.float32)

    class FakePreprocessor:
        fixed_input_size = None

        def __call__(self, rgb):
            assert self.fixed_input_size == (294, 518)
            return torch.zeros(1, 3, *self.fixed_input_size, dtype=torch.float32)

    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        engine_path=tmp_path / "model.trt",
    )
    provider._engine = FakeEngine()
    provider._preprocessor = FakePreprocessor()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = provider.predict_profile(torch.zeros(1, 3, 2160, 1920))

    assert captured["tensor_shape"] == (1, 3, 294, 518)
    assert result.depth.shape == (1, 1, 2160, 1920)
