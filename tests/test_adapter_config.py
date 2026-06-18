import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime import (
    StereoRuntimeConfig,
    depth_provider_config_from_runtime,
    runtime_frame_contract,
    runtime_config_from_d2s_settings,
    stereo_config_from_runtime,
)
from stereo_runtime.adapter import preset_for_runtime_mode


def test_runtime_config_derives_artifacts_from_model_dir():
    model_dir = Path(r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=model_dir,
    )

    assert config.model_path == model_dir
    assert config.onnx_path == model_dir / "model_fp16_294x518.onnx"
    assert config.fp32_onnx_path == model_dir / "model_fp32_294x518.onnx"
    assert config.trt_engine_path == model_dir / "model_fp16_294x518.trt"
    assert config.artifact_paths()["model_dir"] == str(model_dir)


def test_runtime_config_maps_depth_backend_auto_to_native_tensorrt():
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="auto",
        depth_upsample="guided",
        depth_upsample_edge_strength=0.5,
    )
    depth_config = depth_provider_config_from_runtime(config)

    assert depth_config.backend == "tensorrt_native"
    assert depth_config.cache_dir == config.model_path.parent
    assert depth_config.onnx_path == config.onnx_path
    assert depth_config.engine_path == config.trt_engine_path
    assert depth_config.local_files_only is True
    assert depth_config.depth_upsample == "guided"
    assert depth_config.depth_upsample_edge_strength == 0.5


def test_runtime_config_maps_modes_and_stereo_params():
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        mode="game",
        stereo_quality="quality_4k",
        output_format="full_sbs",
        depth_strength=1.6,
        temporal_strength=0.6,
        hole_fill="fast",
        layers=2,
    )
    stereo = stereo_config_from_runtime(config)

    assert preset_for_runtime_mode("movie") == "cinema"
    assert preset_for_runtime_mode("game") == "game_low_latency"
    assert preset_for_runtime_mode("image") == "still_image_hq"
    assert stereo.backend == "quality_4k"
    assert stereo.output_format == "full_sbs"
    assert stereo.depth_strength == 1.6
    assert stereo.temporal_strength == 0.6
    assert stereo.hole_fill == "fast"
    assert stereo.layers == 2


def test_hq_quality_raises_layers_to_at_least_three():
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        mode="image",
        stereo_quality="hq_4k",
        layers=2,
    )
    stereo = stereo_config_from_runtime(config)

    assert stereo.backend == "hq_4k"
    assert stereo.layers == 3


def test_runtime_config_defines_d2s_rgb_frame_contract():
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
    )

    contract = runtime_frame_contract(config)

    assert contract["input"] == "rgb_frame"
    assert "perform capture-side color preprocessing" in contract["host_responsibility"]
    assert "depth inference" in contract["stereo_runtime_responsibility"]
    assert "BGR/BGRA-to-RGB conversion" in contract["not_stereo_runtime_responsibility"]
    assert config.to_report()["frame_contract"] == contract


def test_runtime_config_from_d2s_settings_maps_legacy_model_and_trt_flags():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "TensorRT": True,
            "Recompile TensorRT": True,
            "FP16": True,
            "Display Mode": "Full-SBS",
            "Run Mode": "Game",
            "Depth Strength": 1.8,
            "Convergence": 0.1,
            "IPD": 0.07,
        },
        cache_dir="./models",
        device="cuda",
    )

    assert config.resolved_model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert config.depth_backend == "tensorrt_native"
    assert config.onnx_dtype == "fp16"
    assert config.build_trt_engine is True
    assert config.force_rebuild_trt is True
    assert config.output_format == "full_sbs"
    assert config.mode == "game"
    assert config.depth_strength == 1.8
    assert config.convergence == 0.1
    assert config.ipd == 0.07


def test_runtime_config_from_d2s_settings_keeps_gui_fp16_as_export_request_only():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "DepthPro-Large",
            "TensorRT": False,
            "FP16": False,
            "Display Mode": "Half-SBS",
        },
        device="cuda",
    )

    assert config.resolved_model_id == "apple/DepthPro-hf"
    assert config.depth_backend == "pytorch_cuda"
    assert config.onnx_dtype == "fp32"
    assert config.build_trt_engine is False
