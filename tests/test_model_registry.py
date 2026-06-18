from stereo_runtime import DepthRuntimeConfig, ModelRegistry, resolve_model_dir
from stereo_runtime.adapter import depth_provider_config_from_runtime


def test_default_registry_contains_d2s_models():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("Distill-Any-Depth-Base") == "lc700x/Distill-Any-Depth-Base-hf"
    assert registry.resolve_model_id("DepthPro-Large") == "apple/DepthPro-hf"
    assert registry.resolve_model_id("depth-anything/DA3-BASE") == "depth-anything/DA3-BASE"
    assert len(registry.names()) >= 40


def test_depth_runtime_config_resolves_model_dir_from_model_name():
    config = DepthRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="./models")

    assert config.resolved_model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(config.model_path).replace("\\", "/").endswith("models/models--lc700x--Distill-Any-Depth-Base-hf")
    assert config.onnx_path.name == "model_fp16_294x518.onnx"
    assert config.fp32_onnx_path.name == "model_fp32_294x518.onnx"
    assert config.trt_engine_path.name == "model_fp16_294x518.trt"


def test_resolve_model_dir_uses_huggingface_cache_name():
    path = resolve_model_dir("owner/repo-name", "cache-root")

    assert str(path).replace("\\", "/") == "cache-root/models--owner--repo-name"


def test_depth_runtime_config_passes_trt_build_options():
    config = DepthRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        depth_backend="tensorrt_native",
        build_trt_engine=True,
        force_rebuild_trt=True,
        use_cuda_graph=True,
    )

    provider_config = depth_provider_config_from_runtime(config)

    assert provider_config.build_engine is True
    assert provider_config.force_rebuild is True
    assert provider_config.use_cuda_graph is True
