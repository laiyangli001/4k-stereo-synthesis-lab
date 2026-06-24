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
    assert config.migraphx_graph_path == model_dir / "model_fp16_294x518.mgx"
    assert config.artifact_paths()["model_dir"] == str(model_dir)
    assert config.artifact_paths()["migraphx_graph_path"] == str(model_dir / "model_fp16_294x518.mgx")


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
    assert depth_config.onnx_path is None
    assert depth_config.engine_path is None
    assert depth_config.local_files_only is False
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
    assert config.onnx_dtype == "auto"
    assert config.build_trt_engine is True
    assert config.force_rebuild_trt is True
    assert config.output_format == "full_sbs"
    assert config.mode == "game"
    assert config.depth_strength == 1.8
    assert config.convergence == 0.1
    assert config.ipd == 0.07
    assert config.ipd_mm == 70.0
    assert config.stereo_scale == 0.4


def test_runtime_config_from_d2s_settings_maps_migraphx_flags_before_tensorrt():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "MIGraphX": True,
            "Recompile MIGraphX": True,
            "TensorRT": True,
            "Recompile TensorRT": True,
        },
        cache_dir="./models",
        device="cuda",
    )
    depth_config = depth_provider_config_from_runtime(config)

    assert config.depth_backend == "migraphx_rocm"
    assert config.build_migraphx_graph is True
    assert config.force_rebuild_migraphx is True
    assert config.build_trt_engine is True
    assert config.force_rebuild_trt is True
    assert depth_config.backend == "migraphx_rocm"
    assert depth_config.onnx_path == config.onnx_path
    assert depth_config.engine_path == config.migraphx_graph_path
    assert depth_config.build_engine is True
    assert depth_config.force_rebuild is True


def test_runtime_config_from_d2s_settings_defaults_convergence_to_zero():
    config = runtime_config_from_d2s_settings(
        {"Depth Model": "Distill-Any-Depth-Base"},
        cache_dir="./models",
        device="cuda",
    )

    assert config.convergence == 0.0


def test_runtime_config_from_d2s_settings_uses_dtype_auto_for_gui_fp16_flag():
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
    assert config.onnx_dtype == "auto"
    assert config.build_trt_engine is False

def test_runtime_config_from_d2s_settings_maps_realtime_stereo_options():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Stereo Preset": "Game / Low Latency",
            "Stereo Quality": "hq_4k",
            "Display Mode": "Anaglyph",
            "Max Shift Ratio": 0.08,
            "IPD mm": 63,
            "Stereo Scale": 0.35,
            "Temporal": False,
            "Temporal Strength": 0.4,
            "Auto Scene Reset": False,
            "Scene Reset Threshold": 0.18,
            "Reset Cooldown Frames": 2,
            "Foreground Scale": 0.2,
            "Depth Antialias Strength": 0.6,
            "Edge Threshold": 0.06,
            "Edge Dilation": 3,
            "Mask Feather Radius": 2,
            "Hole Fill Mode": "Soft / Low Ghost",
            "Cross Eyed": True,
            "Anaglyph Method": "green_magenta",
        },
        device="cuda",
    )
    stereo = stereo_config_from_runtime(config)

    assert config.mode == "game"
    assert config.stereo_preset == "Game / Low Latency"
    assert stereo.backend == "hq_4k"
    assert stereo.output_format == "anaglyph"
    assert stereo.max_shift_ratio == 0.08
    assert stereo.ipd == 0.063
    assert stereo.ipd_mm == 63.0
    assert stereo.stereo_scale == 0.35
    assert stereo.convergence == 0.0
    assert stereo.temporal is False
    assert stereo.temporal_strength == 0.4
    assert stereo.auto_reset_temporal is False
    assert stereo.scene_reset_threshold == 0.18
    assert stereo.reset_cooldown_frames == 2
    assert stereo.foreground_scale == 0.2
    assert stereo.depth_antialias_strength == 0.6
    assert stereo.edge_threshold == 0.06
    assert stereo.edge_dilation == 3
    assert stereo.mask_feather_radius == 2
    assert stereo.hole_fill_mode == "soft_low_ghost"
    assert stereo.hole_fill_radius == 1
    assert stereo.hole_fill_strength == 0.6
    assert stereo.cross_eyed is True
    assert stereo.anaglyph_method == "green_magenta"


def test_runtime_config_accepts_full_sbs_display_name_variants():
    base = {"Depth Model": "Distill-Any-Depth-Base"}
    for value in ("Full-SBS", "Full SBS", "full side by side", "Full/Side-by-Side"):
        config = runtime_config_from_d2s_settings({**base, "Display Mode": value})
        assert config.output_format == "full_sbs"
        assert stereo_config_from_runtime(config).output_format == "full_sbs"


def test_runtime_config_accepts_fast_plus_stereo_quality_variants():
    base = {"Depth Model": "Distill-Any-Depth-Base"}
    for value in ("fast_plus", "fastplus", "fast+"):
        config = runtime_config_from_d2s_settings({**base, "Stereo Quality": value})
        assert config.stereo_quality == "fast_plus"
        assert stereo_config_from_runtime(config).backend == "fast_plus"


def test_runtime_fast_quality_disables_temporal_and_postprocess_overrides():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Stereo Preset": "cinema",
            "Stereo Quality": "fast",
            "Temporal": True,
            "Temporal Strength": 0.7,
            "Auto Scene Reset": True,
            "Scene Reset Threshold": 0.22,
            "Reset Cooldown Frames": 3,
            "Foreground Scale": 0.5,
            "Depth Antialias Strength": 2.0,
        },
        device="cuda",
    )

    stereo = stereo_config_from_runtime(config)

    assert stereo.backend == "fast"
    assert stereo.temporal is False
    assert stereo.temporal_strength == 0.0
    assert stereo.auto_reset_temporal is False
    assert stereo.scene_reset_threshold == 0.0
    assert stereo.reset_cooldown_frames == 0
    assert stereo.foreground_scale == 0.0
    assert stereo.depth_antialias_strength == 0.0


def test_runtime_config_defaults_mask_feather_radius_for_hole_fill():
    config = runtime_config_from_d2s_settings({"Depth Model": "Distill-Any-Depth-Base"})
    stereo = stereo_config_from_runtime(config)

    assert config.mask_feather_radius == 3
    assert stereo.mask_feather_radius == 3


def test_runtime_config_hole_fill_mode_overrides_legacy_radius_strength_values():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Hole Fill Mode": "Sharp Test",
            "Hole Fill Radius": 5,
            "Hole Fill Strength": 0.4,
        }
    )
    stereo = stereo_config_from_runtime(config)

    assert config.hole_fill_mode == "sharp_test"
    assert config.hole_fill_radius == 1
    assert config.hole_fill_strength == 1.0
    assert stereo.hole_fill_mode == "sharp_test"
    assert stereo.hole_fill_radius == 1
    assert stereo.hole_fill_strength == 1.0


def test_runtime_config_profile_sync_defaults_off_and_maps_setting():
    base = {"Depth Model": "Distill-Any-Depth-Base", "TensorRT": True}

    default_config = runtime_config_from_d2s_settings(base, device="cuda")
    assert default_config.profile_sync is False
    assert depth_provider_config_from_runtime(default_config).profile_sync is False

    profiled_config = runtime_config_from_d2s_settings({**base, "Depth Profile Sync": True}, device="cuda")
    assert profiled_config.profile_sync is True
    assert depth_provider_config_from_runtime(profiled_config).profile_sync is True


def test_runtime_config_keeps_fixed_stereo_preset_separate_from_run_mode():
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Run Mode": "Auto",
            "Stereo Preset": "Still Image / HQ",
            "Stereo Quality": "quality_4k",
        },
        device="cuda",
    )

    assert config.mode == "auto"
    assert config.stereo_preset == "Still Image / HQ"
    stereo = stereo_config_from_runtime(config)
    assert stereo.backend == "quality_4k"
    assert stereo.temporal is True
