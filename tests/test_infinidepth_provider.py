import pytest

from stereo_runtime.depth_provider import DepthProviderConfig, InfiniDepthProvider, create_depth_provider
from stereo_runtime.model_registry import ModelRegistry


@pytest.mark.parametrize(
    ("model_id", "encoder"),
    [
        ("lc700x/InfiniDepth-Small", "vits16"),
        ("lc700x/InfiniDepth-SmallPlus", "vits16plus"),
        ("lc700x/InfiniDepth-Base", "vitb16"),
        ("lc700x/InfiniDepth-Large", "vitl16"),
    ],
)
def test_infinidepth_models_use_specialized_provider(model_id, encoder):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_cuda",
            model_id=model_id,
            model_name=model_id.rsplit("/", 1)[-1],
            device="cpu",
            local_files_only=True,
        )
    )

    assert isinstance(provider, InfiniDepthProvider)
    assert provider.encoder == encoder
    assert provider.info.provider == "models.InfiniDepth.api.InfiniDepthModel"
    assert provider.info.runtime == "infinidepth"


def test_model_registry_contains_infinidepth_family():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("InfiniDepth-Small") == "lc700x/InfiniDepth-Small"
    assert registry.resolve_model_id("InfiniDepth-SmallPlus") == "lc700x/InfiniDepth-SmallPlus"
    assert registry.resolve_model_id("InfiniDepth-Base") == "lc700x/InfiniDepth-Base"
    assert registry.resolve_model_id("InfiniDepth-Large") == "lc700x/InfiniDepth-Large"


def test_infinidepth_onnx_provider_prepares_exported_artifact(monkeypatch, tmp_path):
    calls = {}
    selected = tmp_path / "models--lc700x--InfiniDepth-Base" / "model_fp16_288x512.onnx"

    class Paths:
        trt_fp16_path = tmp_path / "models--lc700x--InfiniDepth-Base" / "model_fp16_288x512.trt"

    class Artifacts:
        selected_onnx_path = selected
        paths = Paths()

    def fake_prepare(*args, **kwargs):
        calls["prepare_args"] = args
        calls.update(kwargs)
        return Artifacts()

    class FakeOnnxProvider:
        def __init__(self, **kwargs):
            calls["provider_kwargs"] = kwargs

    import stereo_runtime.depth_provider as provider_module
    import stereo_runtime.providers.nvidia.onnx_cuda as onnx_module

    monkeypatch.setattr(provider_module, "_prepare_accelerated_artifacts", fake_prepare)
    monkeypatch.setattr(onnx_module, "OnnxCudaDepthProvider", FakeOnnxProvider)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="onnx_cuda",
            model_id="lc700x/InfiniDepth-Base",
            model_name="InfiniDepth-Base",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=False,
            prefer_onnx=True,
        )
    )

    assert isinstance(provider, FakeOnnxProvider)
    assert calls.get("build_trt", False) is False
    assert calls["provider_kwargs"]["onnx_path"] == selected
    assert calls["provider_kwargs"]["model_id"] == "lc700x/InfiniDepth-Base"
    assert calls["provider_kwargs"]["model_name"] == "InfiniDepth-Base"
