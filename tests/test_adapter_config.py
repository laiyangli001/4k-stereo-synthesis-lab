import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab import StereoLabRuntimeConfig, depth_provider_config_from_runtime, stereo_config_from_runtime
from stereo_lab.adapter import preset_for_runtime_mode


def test_runtime_config_derives_artifacts_from_model_dir():
    model_dir = Path(r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf")
    config = StereoLabRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=model_dir,
    )

    assert config.model_path == model_dir
    assert config.onnx_path == model_dir / "model_fp16_294x518.onnx"
    assert config.fp32_onnx_path == model_dir / "model_fp32_294x518.onnx"
    assert config.trt_engine_path == model_dir / "model_fp16_294x518.trt"
    assert config.artifact_paths()["model_dir"] == str(model_dir)


def test_runtime_config_maps_depth_backend_auto_to_native_tensorrt():
    config = StereoLabRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="auto",
    )
    depth_config = depth_provider_config_from_runtime(config)

    assert depth_config.backend == "tensorrt_native"
    assert depth_config.cache_dir == config.model_path.parent
    assert depth_config.onnx_path == config.onnx_path
    assert depth_config.engine_path == config.trt_engine_path
    assert depth_config.local_files_only is True


def test_runtime_config_maps_modes_and_stereo_params():
    config = StereoLabRuntimeConfig(
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
    config = StereoLabRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        mode="image",
        stereo_quality="hq_4k",
        layers=2,
    )
    stereo = stereo_config_from_runtime(config)

    assert stereo.backend == "hq_4k"
    assert stereo.layers == 3
