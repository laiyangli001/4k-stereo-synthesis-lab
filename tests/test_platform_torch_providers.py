from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_runtime.providers.apple import DistillAnyDepthBaseMps, GenericAutoDepthMpsProvider
from stereo_runtime.providers.intel import DistillAnyDepthBaseXpu, GenericAutoDepthXpuProvider


def test_create_pytorch_xpu_provider_marks_backend():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_xpu",
            device="xpu",
            local_files_only=True,
            prefer_tensorrt=True,
            prefer_onnx=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBaseXpu)
    assert provider.info.depth_backend == "pytorch_xpu"
    assert provider.info.runtime == "transformers-xpu"
    assert provider.info.execution_provider == "Intel XPU PyTorch"
    assert provider.info.output_device == "xpu"


def test_create_pytorch_xpu_provider_supports_generic_models():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="intel_xpu",
            model_id="apple/DepthPro-hf",
            model_name="DepthPro-Large",
            device="xpu",
            local_files_only=True,
            depth_resolution=518,
            patch_size=14,
        )
    )

    assert isinstance(provider, GenericAutoDepthXpuProvider)
    assert provider.info.model_id == "apple/DepthPro-hf"
    assert provider.info.model_name == "DepthPro-Large"
    assert provider.info.depth_backend == "pytorch_xpu"


def test_create_pytorch_mps_provider_marks_backend():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_mps",
            device="mps",
            local_files_only=True,
            prefer_tensorrt=True,
            prefer_onnx=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBaseMps)
    assert provider.info.depth_backend == "pytorch_mps"
    assert provider.info.runtime == "transformers-mps"
    assert provider.info.execution_provider == "Apple MPS PyTorch"
    assert provider.info.output_device == "mps"


def test_create_pytorch_mps_provider_supports_generic_models():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="apple_mps",
            model_id="apple/DepthPro-hf",
            model_name="DepthPro-Large",
            device="mps",
            local_files_only=True,
            depth_resolution=518,
            patch_size=14,
        )
    )

    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.info.model_id == "apple/DepthPro-hf"
    assert provider.info.model_name == "DepthPro-Large"
    assert provider.info.depth_backend == "pytorch_mps"
