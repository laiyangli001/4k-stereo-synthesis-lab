from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_runtime.providers.amd import GenericTorchRocmDepthProvider, TorchRocmDepthProvider
import stereo_runtime.providers.amd.migraphx as migraphx_provider


def test_create_pytorch_rocm_provider_marks_backend():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_rocm",
            device="cuda",
            local_files_only=True,
            prefer_tensorrt=True,
            prefer_onnx=True,
        )
    )

    assert isinstance(provider, TorchRocmDepthProvider)
    assert provider.info.depth_backend == "pytorch_rocm"
    assert provider.info.runtime == "transformers-rocm"
    assert provider.info.execution_provider == "ROCm PyTorch"
    assert provider.info.output_device == "cuda"


def test_create_pytorch_rocm_provider_supports_generic_models():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="amd_rocm",
            model_id="apple/DepthPro-hf",
            model_name="DepthPro-Large",
            device="cuda",
            local_files_only=True,
            depth_resolution=518,
            patch_size=14,
        )
    )

    assert isinstance(provider, GenericTorchRocmDepthProvider)
    assert provider.info.model_id == "apple/DepthPro-hf"
    assert provider.info.model_name == "DepthPro-Large"
    assert provider.info.depth_backend == "pytorch_rocm"


def test_create_migraphx_rocm_provider_falls_back_to_pytorch_rocm(monkeypatch):
    monkeypatch.setattr(migraphx_provider, "is_rocm_torch_available", lambda: True)
    monkeypatch.setattr(migraphx_provider, "is_migraphx_available", lambda: False)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="migraphx_rocm",
            device="cuda",
            local_files_only=True,
            allow_pytorch_fallback=True,
            prefer_tensorrt=False,
            prefer_onnx=False,
        )
    )

    assert isinstance(provider, TorchRocmDepthProvider)
    assert provider.info.depth_backend == "pytorch_rocm"
    assert provider.info.fallback_reason == "migraphx is not installed"
