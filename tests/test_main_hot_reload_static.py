
from pathlib import Path


def test_main_has_limited_stereo_hot_reload_path():
    root = Path(__file__).resolve().parents[1]
    main_source = root / "src" / "main.py"
    main_text = main_source.read_text(encoding="utf-8")

    context_text = (root / "src" / "app_support" / "runtime_context.py").read_text(encoding="utf-8")
    assert "StereoHotReloader" in context_text
    callbacks_text = (root / "src" / "app_support" / "runtime_callbacks.py").read_text(encoding="utf-8")
    assert "def apply_stereo_hot_reload_if_needed" in callbacks_text
    assert "stereo_hot_reloader.apply_if_needed" in callbacks_text or "apply_if_needed" in callbacks_text
    assert "apply_stereo_hot_reload_if_needed" in callbacks_text
    assert "reset_temporal=False" not in main_text

    hot_reload_source = root / "src" / "stereo_runtime" / "hot_reload.py"
    text = hot_reload_source.read_text(encoding="utf-8")

    assert "def hot_reload_value_snapshot" in text
    assert '"depth_strength"' in text
    assert '"convergence"' in text
    assert '"ipd_mm"' in text
    assert '"stereo_scale"' in text
    assert '"max_shift_ratio"' in text
    assert '"cross_eyed"' in text
    assert '"temporal_strength"' in text
    assert '"scene_reset_threshold"' in text
    assert '"reset_cooldown_frames"' in text
    assert '"foreground_scale"' in text
    assert "def clamp_foreground_scale_hot_reload" in text
    assert "clamp_foreground_scale_hot_reload(" in text
    assert '"depth_antialias_strength"' in text
    assert '"edge_dilation"' in text
    assert '"edge_threshold"' in text
    assert '"anaglyph_method"' in text
    assert '"depth_safety"' not in text
    assert "def to_bool_hot_reload" in text
    assert "def to_optional_bool_hot_reload" not in text
    assert "runtime.config = replace(runtime.config, **values)" in text
    assert "reset_temporal=False" in text
def test_stereo_warmup_uses_runtime_frame_shape_and_dedup_key():
    root = Path(__file__).resolve().parents[1]
    main_source = root / "src" / "main.py"
    main_text = main_source.read_text(encoding="utf-8")

    callbacks_text = (root / "src" / "app_support" / "runtime_callbacks.py").read_text(encoding="utf-8")
    assert "def stereo_warmup_key(self, rgb_frame):" in callbacks_text
    assert "stereo_warmup_tracker.key_for_frame(rgb_frame)" in callbacks_text
    assert "warmup_stereo_once_for_frame=runtime_callbacks.warmup_stereo_once_for_frame" in main_text

    pipeline_source = root / "src" / "stereo_runtime" / "pipeline.py"
    pipeline_text = pipeline_source.read_text(encoding="utf-8")
    assert "ctx.warmup_stereo_once_for_frame(runtime_rgb)" in pipeline_text

    helper_source = root / "src" / "stereo_runtime" / "session_helpers.py"
    helper_text = helper_source.read_text(encoding="utf-8")
    assert "def key_for_frame(self, rgb_frame):" in helper_text
    assert "shape = tuple(getattr(rgb_frame, \"shape\", ()))" in helper_text
    assert "self.keys" in helper_text
    key_start = helper_text.index("def key_for_frame(self, rgb_frame):")
    key_end = helper_text.index("def warmup_once_for_frame", key_start)
    key_block = helper_text[key_start:key_end]
    assert "foreground_scale" not in key_block
    assert "depth_antialias_strength" not in key_block
    assert "warmup_stereo_kernels_for_frame(rgb_frame)" in helper_text

    runtime_source = root / "src" / "stereo_runtime" / "runtime.py"
    runtime_text = runtime_source.read_text(encoding="utf-8")
    assert "foreground_scales =" in runtime_text
    assert "antialias_values =" in runtime_text
    assert "def warmup_stereo_kernels_for_frame(self, rgb_frame: torch.Tensor)" in runtime_text
    assert "_, height, width = rgb_frame.shape" in runtime_text
    assert "3840" not in runtime_text[runtime_text.index("def warmup_stereo_kernels_for_frame"):runtime_text.index("def reset_stats")]
def test_stereo_scene_auto_switch_removed():
    source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = source.read_text(encoding="utf-8")

    assert "AutoModeRuntime" not in text
    assert "AutoModeSignals" not in text
    assert "_update_auto_stereo_mode(runtime_rgb)" not in text
    assert "_ensure_auto_signal_sampler()" not in text
    assert "auto={'on' if stereo_auto_enabled else 'off'}" not in text
    runtime_context_text = (Path(__file__).resolve().parents[1] / "src" / "app_support" / "runtime_context.py").read_text(encoding="utf-8")
    assert 'return False, "cinema" if preset == "auto" else preset' in runtime_context_text