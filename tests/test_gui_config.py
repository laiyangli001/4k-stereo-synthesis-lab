from pathlib import Path


GUI_PKG = Path(__file__).resolve().parents[1] / "src" / "gui"


def _gui_source():
    """Return combined source of all .py files in the gui package."""
    texts = {}
    for f in sorted(GUI_PKG.glob("*.py")):
        texts[f.name] = f.read_text(encoding="utf-8")
    return texts


def _all_text():
    return "\n".join(_gui_source().values())


def _file_text(name):
    return _gui_source().get(name, "")


def _controls_source():
    return GUI_PKG / "controls.py"


def _config_source():
    return GUI_PKG / "config.py"


def _localization_source():
    return GUI_PKG / "localization.py"


def test_gui_stereo_preset_uses_dropdown_value_for_load_and_save():
    text = _file_text("config_mgr.py")
    assert 'self.stereo_preset_dd.value = self._preset_to_display(' in text
    assert 'cfg.get("Stereo Preset", DEFAULTS["Stereo Preset"])' in text
    assert 'stereo_preset = self._display_to_preset(self.stereo_preset_dd.value)' in text
    assert '"Stereo Preset": stereo_preset,' in text
    assert '"Stereo Preset": DEFAULTS["Stereo Preset"],' not in text


def test_gui_hot_stereo_params_auto_save_on_select():
    all_text = _all_text()
    assert "def on_stereo_hot_param_change" in all_text
    assert "def _save_stereo_hot_params" in all_text
    assert 'on_select=self.on_stereo_hot_param_change' in _file_text("builders.py")
    assert '"IPD": self._display_ipd_mm_to_runtime_m(self.ipd_dd.value)' in _file_text("config_mgr.py")
    assert '"Stereo Scale": self._parse_float(self.stereo_scale_dd.value' in _file_text("config_mgr.py")
    assert '"Convergence": self._parse_float(self.convergence_dd.value' in _file_text("config_mgr.py")
    assert '"Depth Strength": self._parse_float(self.depth_strength_dd.value' in _file_text("config_mgr.py")
    assert '"Max Shift Ratio": self._parse_float(self.max_shift_dd.value' in _file_text("config_mgr.py")
    assert '"Temporal Strength": temporal_strength' in _file_text("config_mgr.py")
    assert '"Scene Reset Threshold": scene_reset_threshold' in _file_text("config_mgr.py")
    assert '"Reset Cooldown Frames": self._parse_int(self.reset_cooldown_dd.value' in _file_text("config_mgr.py")
    assert "def _clamp_foreground_scale" in all_text
    assert "fg_options = [f\"{i / 10:.1f}\" for i in range(-9, 0)]" in _file_text("builders.py")
    assert '"Foreground Scale": foreground_scale' in _file_text("config_mgr.py")
    assert '"Depth Antialias Strength": antialias_strength' in _file_text("config_mgr.py")
    assert '"Edge Dilation": self._parse_int(self.edge_dilation_dd.value' in _file_text("config_mgr.py")
    assert '"Mask Feather Radius": self._parse_int(self.mask_feather_dd.value' in _file_text("config_mgr.py")
    assert '"Hole Fill Mode": self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value)' in _file_text("config_mgr.py")
    assert '"Edge Threshold": self._parse_float(self.edge_threshold_dd.value' in _file_text("config_mgr.py")
    assert '"Anaglyph Method": self.anaglyph_dd.value' in _file_text("config_mgr.py")
    assert '"Cross Eyed": bool(self.cross_eyed_cb.value)' in _file_text("config_mgr.py")
    assert 'on_change=self.on_stereo_hot_param_change' in _file_text("builders.py")
    assert "self._schedule_stereo_hot_save()" in all_text



def test_compact_dropdown_click_updates_internal_value_before_callback():
    source = _controls_source()
    text = source.read_text(encoding="utf-8")
    assert "val = e.control.data\n            self._value = val\n            self._label.value = val" in text


def test_gui_status_translation_keys_are_safe_for_language_switch():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"stereo_parameters_saved": "Stereo parameters saved"' in localization_text
    assert '"stereo_parameters_saved": "立体参数已保存"' in localization_text
    assert 'key="stereo_parameters_saved"' in all_text
    assert 'UI_MESSAGES[self.locale].get(self._status_key, self.status_text.value)' in all_text
    assert 'key="Stereo parameters saved"' not in all_text


def test_gui_scene_preset_loads_complete_advanced_stereo_controls():
    text = _file_text("handlers.py")
    apply_start = text.index("def _apply_stereo_preset_values")
    block = text[apply_start:]
    assert "self.scene_reset_dd.value" in block
    assert "self.reset_cooldown_dd.value" in block
    assert '"scene_reset_threshold"' in _file_text("config_mgr.py")
    assert '"reset_cooldown_frames"' in _file_text("config_mgr.py")
    assert "self.stereo_scale_dd.value" in block
    assert "on_select=self.on_stereo_preset_change" in _file_text("builders.py")


def test_advanced_stereo_is_not_persisted_and_starts_collapsed():
    all_text = _all_text()
    cfg_text = _file_text("config_mgr.py")
    builders_text = _file_text("builders.py")
    assert '"Advanced Stereo": False' not in all_text
    assert 'label="Advanced Stereo", value=False, on_change=self.on_advanced_stereo_change' in builders_text
    assert 'self.advanced_stereo_cb.value = False' in cfg_text
    assert '"Advanced Stereo": self.advanced_stereo_cb.value' not in all_text
    assert 'cfg.get("Advanced Stereo"' not in all_text


def test_hole_fill_mode_is_visible_next_to_stereo_mode_without_advanced_stereo():
    builders_text = _file_text("builders.py")
    row_start = builders_text.index("stereo_row0 = ft.Row([")
    row_end = builders_text.index("], spacing=1)", row_start)
    stereo_row = builders_text[row_start:row_end]
    assert "self.stereo_preset_label" in stereo_row
    assert "self.stereo_preset_dd" in stereo_row
    assert "self.hole_fill_mode_label" in stereo_row
    assert "self.hole_fill_mode_dd" in stereo_row
    assert stereo_row.index("self.stereo_preset_dd") < stereo_row.index("self.hole_fill_mode_label")
    depth_start = builders_text.index("depth_group = ft.Container(")
    depth_end = builders_text.index("device_group = ft.Container(", depth_start)
    depth_block = builders_text[depth_start:depth_end]
    assert "stereo_row0" in depth_block
    assert "hole_fill_mode_row" not in builders_text
    advanced_start = builders_text.index("self._advanced_stereo_rows = [")
    advanced_end = builders_text.index("]", advanced_start)
    advanced_block = builders_text[advanced_start:advanced_end]
    assert "self.hole_fill_mode_label" not in advanced_block
    assert "self.hole_fill_mode_dd" not in advanced_block


def test_advanced_device_options_is_not_persisted_and_starts_collapsed():
    all_text = _all_text()
    cfg_text = _file_text("config_mgr.py")
    builders_text = _file_text("builders.py")
    assert '"Advanced Device Options": False' not in all_text
    assert 'label="Advanced Options", value=False, on_change=self.on_advanced_device_change' in builders_text
    assert 'self.advanced_device_cb.value = False' in cfg_text
    assert '"Advanced Device Options": self.advanced_device_cb.value' not in all_text
    assert 'cfg.get("Advanced Device Options"' not in all_text


def test_noisy_third_party_console_output_is_filtered():
    text = _file_text("process.py")
    assert "def _is_noisy_console_output" in text
    assert "[NativeUtil] sogou_native_util_pc loaded successfully" in text
    assert "[warmup] same version" in text
    filter_index = text.index("if _is_noisy_console_output(data):")
    console_write_index = text.index("self.original.write(data)", filter_index)
    log_write_index = text.index("with open(LOG_FILE", filter_index)
    assert "return len(data or \"\")" in text[filter_index:console_write_index]
    assert filter_index < console_write_index < log_write_index


def test_gui_uses_single_rolling_log_for_gui_and_child_output():
    paths_text = _file_text("paths.py")
    process_text = _file_text("process.py")
    assert 'LOG_FILE = os.path.join(LOG_DIR, "desktop2stereo.log")' in paths_text
    assert "DIAG_LOG = LOG_FILE" in paths_text
    assert 'stdout=asyncio.subprocess.PIPE' in process_text
    assert 'stderr=asyncio.subprocess.STDOUT' in process_text
    assert 'child_env["PYTHONIOENCODING"] = "utf-8"' in process_text
    assert "async def _pump_child_output" in process_text
    assert "asyncio.create_task(self._pump_child_output(self.process))" in process_text


def test_gui_stop_uses_stop_request_file_before_force_kill():
    paths_text = _file_text("paths.py")
    process_text = _file_text("process.py")
    assert 'STOP_REQUEST_FILE = os.path.join(LOG_DIR, "stop.request")' in paths_text
    assert "with open(STOP_REQUEST_FILE, \"w\", encoding=\"utf-8\")" in process_text
    assert "await asyncio.wait_for(proc.wait(), timeout=1)" in process_text
    assert "proc.kill()" in process_text
    graceful_index = process_text.index("with open(STOP_REQUEST_FILE")
    kill_index = process_text.index("proc.kill()", graceful_index)
    assert graceful_index < kill_index


def test_stereo_quality_is_hidden_and_derived_from_stereo_mode():
    all_text = _all_text()
    handlers_text = _file_text("handlers.py")
    builders_text = _file_text("builders.py")
    config_mgr_text = _file_text("config_mgr.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"fast": "Lowest"' in localization_text
    assert '"fast_plus": "Medium"' in localization_text
    assert '"quality_4k": "High"' in localization_text
    assert '"hq_4k": "Highest"' in localization_text
    assert "self.stereo_quality_dd.visible = False" in builders_text
    assert "self.stereo_quality_label.visible = False" in handlers_text
    assert "def _stereo_quality_for_preset" in config_mgr_text
    assert '"Stereo Quality": stereo_quality' in config_mgr_text
    assert '"Synthetic View": stereo_quality' in config_mgr_text
    assert '"Stereo Quality": self._display_to_stereo_quality(self.stereo_quality_dd.value)' not in config_mgr_text
    assert '"Synthetic View": self._display_to_stereo_quality(self.stereo_quality_dd.value)' not in config_mgr_text
    assert '(self.stereo_quality_dd, "tooltip_stereo_quality")' not in all_text


def test_depth_safety_gui_controls_removed():
    all_text = _all_text()
    assert "Depth Safety" not in all_text
    assert "depth_safety" not in all_text


def test_environment_dropdown_saves_canonical_key():
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    config_mgr_text = _file_text("config_mgr.py")

    assert "def environment_key_from_label" in config_text
    assert "def environment_display_label" in config_text
    assert "display_name" in config_text
    assert "on_select=self.on_env_change" in builders_text
    assert "def on_env_change" in handlers_text
    assert 'self._config["Environment Model"] = self.env_key' in handlers_text
    assert '"Environment Model": self.env_key' in config_mgr_text
    assert '"Environment Model": self.env_model_dd.value' not in config_mgr_text


def test_stereo_scale_control_is_next_to_ipd():
    builders_text = _file_text("builders.py")
    assert 'self.ipd_dd = CompactDropdown(options=[str(i) for i in range(50, 71)]' in builders_text
    assert 'self.stereo_scale_label = ft.Text("Stereo Scale:"' in builders_text
    assert 'self.stereo_scale_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 5)]' in builders_text
    assert 'value="0.4", width=S(130), on_select=self.on_stereo_hot_param_change)' in builders_text
    row_start = builders_text.index('row3 = ft.Row([')
    row_end = builders_text.index('# Row 5: Stereo runtime mode. Backend quality is derived from this preset.', row_start)
    row = builders_text[row_start:row_end]
    assert 'self.ipd_dd' in row
    assert 'self.stereo_scale_label' in row
    assert 'self.stereo_scale_dd' in row
    assert row.index('self.ipd_dd') < row.index('self.stereo_scale_label') < row.index('self.stereo_scale_dd')


def test_stereo_preset_exposes_four_mode_choices_without_auto_or_debug():
    all_text = _all_text()
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    assert '"Stereo Preset": "cinema"' in config_text
    assert 'options=["Traditional / Fastest", "Cinema", "Game / Low Latency", "Image  / High Quality"]' in builders_text
    assert '"traditional_fastest"' in _file_text("config_mgr.py")
    assert 'options=["Auto", "Cinema"' not in all_text
    assert '"Auto": "auto"' not in all_text
    assert '"自动": "auto"' not in all_text
    assert '"Debug / Export"]' not in builders_text


def test_ipd_display_maps_to_calibrated_runtime_value():
    import ast

    config_mgr_text = _file_text("config_mgr.py")
    tree = ast.parse(config_mgr_text)
    wanted = {
        "_IPD_RUNTIME_PER_DISPLAY_MM",
        "_runtime_ipd_to_display_mm",
        "_display_ipd_mm_to_runtime_m",
        "_parse_int",
    }
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GUIConfigMixin")
    selected = [
        node
        for node in class_node.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and ((isinstance(node, ast.Assign) and any(getattr(target, "id", None) in wanted for target in node.targets)) or getattr(node, "name", None) in wanted)
    ]
    module = ast.Module(body=[ast.ClassDef(name="GUIConfigMixin", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"DEFAULTS": {"IPD": 0.032}}
    exec(compile(module, "config_mgr.py", "exec"), namespace)
    mixin = namespace["GUIConfigMixin"]

    assert mixin._runtime_ipd_to_display_mm(0.032) == 64
    assert mixin._runtime_ipd_to_display_mm(0.064) == 64
    assert mixin._display_ipd_mm_to_runtime_m("60") == 0.030
    assert mixin._display_ipd_mm_to_runtime_m("64") == 0.032
    assert mixin._display_ipd_mm_to_runtime_m("70") == 0.035


def test_stereo_scale_has_tooltips():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"tooltip_ipd": "Binocular IPD, displayed in mm' in localization_text
    assert '"tooltip_ipd": "双眼瞳距，单位 mm' in localization_text
    assert '"tooltip_stereo_scale": "Stereo strength multiplier; lower values reduce parallax' in localization_text
    assert '"tooltip_stereo_scale": "立体强度倍率；数值越低视差越小' in localization_text
    assert '(self.stereo_scale_dd, "tooltip_stereo_scale")' in all_text


def test_convergence_and_foreground_scale_tooltips_explain_tuning():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"tooltip_convergence": "Zero-parallax screen plane. Raise it in 0.05 steps' in localization_text
    assert '"tooltip_convergence": "零视差屏幕平面。前景太突出或出现重影时，每次提高 0.05' in localization_text
    assert 'positive values strengthen near/far separation' in localization_text
    assert 'negative values compress depth toward the middle' in localization_text
    assert '正值增强近远层次' in localization_text
    assert '负值把深度压向中间' in localization_text
    assert '(self.convergence_dd, "tooltip_convergence")' in all_text
    assert '(self.foreground_scale_dd, "tooltip_foreground_scale")' in all_text


def test_depth_strength_and_antialiasing_tooltips_explain_tuning():
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert 'Use Standard / 2.5 as the baseline' in localization_text
    assert 'foreground objects show ghosts' in localization_text
    assert 'Depth-map smoothing level' in localization_text
    assert 'soften fine geometry' in localization_text
    assert '建议以标准档 2.5 为基准' in localization_text
    assert '前景重影、边缘撕裂或观看不舒服时下调' in localization_text
    assert '深度图平滑级别' in localization_text
    assert '游戏和实时观看保持较低' in localization_text


def test_convergence_dropdown_uses_five_percent_steps():
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    assert 'conv_options = [f"{i / 100:.2f}" for i in range(-50, 101, 5)]' in builders_text
    assert 'options=[v for v in conv_options], value="0.00"' in builders_text
    assert 'self.convergence_dd.value = f"{values[\'convergence\']:.2f}"' in handlers_text


def test_stereo_scale_label_is_localized():
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"Stereo Scale:": "Stereo Scale:"' in localization_text
    assert '"Stereo Scale:": "立体缩放:"' in localization_text
    assert '"Convergence:": "会聚位置:"' in localization_text
    assert '"Convergence:": "会聚点:"' not in localization_text
    assert 'self.stereo_scale_label.value = t["Stereo Scale:"]' in handlers_text


def test_cn_ipd_and_antialiasing_labels_are_user_facing():
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"IPD (m):": "IPD (mm):"' in localization_text
    assert '"IPD (m):": "双眼瞳距:"' in localization_text
    assert '"Anti-aliasing:": "Anti-aliasing:"' in localization_text
    assert '"Anti-aliasing:": "抗锯齿值:"' in localization_text
    assert '"Depth Resolution:": "深度细节:"' in localization_text
    assert '"tooltip_depth_res": "深度细节档位。建议使用最大 518' in localization_text
    assert '最好的立体细节' in localization_text
    assert '数值降低可减少推理耗时和显存占用' in localization_text
    assert '"瞳距 (mm):"' not in localization_text
    assert '"抗锯齿:"' not in localization_text
    assert '"深度分辨率:"' not in localization_text
    assert '"深度图分辨率"' not in localization_text


def test_stereo_quality_options_are_localized():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"quality_4k": "较高"' in localization_text
    assert '"hq_4k": "最高"' in localization_text
    assert 'STEREO_QUALITY_KEYS = ("fast", "fast_plus", "quality_4k", "hq_4k")' in localization_text
    assert 'return [messages[key] for key in STEREO_QUALITY_KEYS]' in localization_text
    assert 'return stereo_quality_options(self.locale)' in _file_text("config_mgr.py")


def test_shift_ratio_and_edge_threshold_options_are_dense():
    builders_text = _file_text("builders.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"Max Shift Ratio:": "Shift Ratio:"' in localization_text
    assert '"Max Shift Ratio:": "位移比例:"' in localization_text
    assert 'self.max_shift_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)]' in builders_text
    assert 'self.edge_threshold_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)]' in builders_text
    assert 'self.mask_feather_dd = CompactDropdown(options=["0", "1", "2", "3", "4", "5"]' in builders_text
    assert 'self.temporal_strength_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 11)]' in builders_text


def test_mask_feather_radius_gui_control_is_localized_and_hot_reloadable():
    builders_text = _file_text("builders.py")
    config_text = _config_source().read_text(encoding="utf-8")
    config_mgr_text = _file_text("config_mgr.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '"Mask Feather Radius": 3' in config_text
    assert 'self.mask_feather_label = ft.Text("Mask Feather:"' in builders_text
    assert 'self.mask_feather_dd = CompactDropdown(options=["0", "1", "2", "3", "4", "5"]' in builders_text
    assert 'self.mask_feather_dd.value = str(cfg.get("Mask Feather Radius", DEFAULTS["Mask Feather Radius"]))' in config_mgr_text
    assert '"Mask Feather Radius": self._parse_int(self.mask_feather_dd.value, DEFAULTS["Mask Feather Radius"])' in config_mgr_text
    assert 'self.mask_feather_label.value = t["Mask Feather:"]' in handlers_text
    assert '(self.mask_feather_dd, "tooltip_mask_feather")' in handlers_text
    assert '"Mask Feather:": "Mask Feather:"' in localization_text
    assert '"Mask Feather:": "遮罩羽化:"' in localization_text
    assert '"tooltip_mask_feather"' in localization_text


def test_reset_defaults_restore_cinema_stereo_defaults():
    config_text = _config_source().read_text(encoding="utf-8")
    process_text = _file_text("process.py")

    assert '"Stereo Preset": "cinema"' in config_text
    assert '"Depth Strength": 2.5' in config_text
    assert '"Depth Quick": "Standard"' in config_text
    assert '"IPD": 0.032' in config_text
    assert '"Foreground Scale": 0.0' in config_text
    assert '"Convergence": 0.0' in config_text
    assert '"Stereo Scale": 0.4' in config_text
    assert '"Anti-aliasing": 1' in config_text
    assert '"Depth Antialias Strength": 1.0' in config_text
    assert '"Mask Feather Radius": 3' in config_text
    assert '"Hole Fill Mode": "balanced"' in config_text
    assert "dynamic_defaults = DEFAULTS.copy()" in process_text
    assert "self.apply_config(dynamic_defaults, keep_optional=False)" in process_text


def test_hole_fill_mode_gui_control_is_localized_and_hot_reloadable():
    builders_text = _file_text("builders.py")
    config_text = _config_source().read_text(encoding="utf-8")
    config_mgr_text = _file_text("config_mgr.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '"Hole Fill Mode": "balanced"' in config_text
    assert 'self.hole_fill_mode_label = ft.Text("Hole Fill Mode:"' in builders_text
    assert 'options=self._hole_fill_mode_options()' in builders_text
    assert 'value=self._hole_fill_mode_to_display("balanced")' in builders_text
    assert 'self.hole_fill_mode_dd.value = self._hole_fill_mode_to_display(cfg.get("Hole Fill Mode", DEFAULTS["Hole Fill Mode"]))' in config_mgr_text
    assert '"Hole Fill Mode": self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value)' in config_mgr_text
    assert 'self.hole_fill_mode_label.value = t["Hole Fill Mode:"]' in handlers_text
    assert '(self.hole_fill_mode_dd, "tooltip_hole_fill_mode")' in handlers_text
    assert 'HOLE_FILL_MODE_KEYS = ("balanced", "soft_low_ghost", "sharp_test")' in localization_text
    assert '"Balanced / Standard": "均衡 / 标准"' in localization_text
    assert '"Sharp / High Detail": "锐利 / 高细节"' in localization_text
    assert '"Hole Fill Mode:": "补洞模式:"' in localization_text


def test_hole_fill_mode_labels_are_balanced_and_legacy_compatible():
    import ast

    source = _localization_source().read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "DEFAULT_LOCALE",
        "MESSAGE_CATALOGS",
        "LOCALE_ALIASES",
        "SUPPORTED_LOCALES",
        "HOLE_FILL_MODE_KEYS",
        "HOLE_FILL_MODE_LABELS",
        "HOLE_FILL_MODE_LEGACY_LABELS",
        "normalize_locale",
        "get_messages",
        "hole_fill_mode_options",
        "hole_fill_mode_to_display",
        "display_to_hole_fill_mode",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and ((isinstance(node, ast.Assign) and any(getattr(target, "id", None) in wanted for target in node.targets)) or getattr(node, "name", None) in wanted)
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    from types import MappingProxyType

    namespace = {"MappingProxyType": MappingProxyType}
    exec(compile(module, str(_localization_source()), "exec"), namespace)

    assert namespace["hole_fill_mode_options"]("EN") == ["Balanced / Standard", "Soft / Low Ghost", "Sharp / High Detail"]
    assert namespace["hole_fill_mode_options"]("CN") == ["均衡 / 标准", "柔和 / 低重影", "锐利 / 高细节"]
    assert namespace["display_to_hole_fill_mode"]("Balanced / Standard") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("均衡 / 标准") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("Balanced") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("锐利测试") == "sharp_test"
    assert namespace["display_to_hole_fill_mode"]("Sharp Test") == "sharp_test"


def test_model_backbone_size_dropdown_has_own_tooltip():
    config_text = _config_source().read_text(encoding="utf-8")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '_MODEL_SIZES = ["Small", "SmallPlus", "Base", "Large", "Giant"]' in config_text
    assert 'family_to_sizes[family].sort(key=lambda s: _SIZE_ORDER.get(s, 99))' in config_text
    assert '(self.model_size_dd, "tooltip_model_size")' in handlers_text
    assert '(self.model_size_dd, "tooltip_depth_model")' not in handlers_text
    assert '"tooltip_model_size": "Model backbone size"' in localization_text
    assert '"tooltip_model_size": "模型骨架大小"' in localization_text


def test_model_backbone_size_dropdown_orders_sizes_by_teammate_policy():
    import ast

    source = _config_source().read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_MODEL_SIZES", "_SIZE_ORDER", "parse_model_name", "build_family_size_map"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and ((isinstance(node, ast.Assign) and any(getattr(target, "id", None) in wanted for target in node.targets)) or getattr(node, "name", None) in wanted)
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(_config_source()), "exec"), namespace)

    assert namespace["parse_model_name"]("InfiniDepth-SmallPlus") == ("InfiniDepth", "SmallPlus")
    assert namespace["parse_model_name"]("DA3NESTED-GIANT-LARGE") == ("DA3NESTED", "Giant-Large")

    _, family_to_sizes = namespace["build_family_size_map"](
        [
            "Example-Large",
            "Example-SmallPlus",
            "Example-Giant",
            "Example-Base",
            "Example-Small",
        ]
    )

    assert family_to_sizes["Example"] == ["Small", "SmallPlus", "Base", "Large", "Giant"]


def test_vsync_uses_teammate_config_key_and_default():
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    config_mgr_text = _file_text("config_mgr.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '"VSync": False' in config_text
    assert '"Local VSync"' not in config_text
    assert 'label="VSync", value=DEFAULTS.get("VSync", False)' in builders_text
    assert 'cfg.get("VSync", DEFAULTS["VSync"])' in config_mgr_text
    assert '"VSync": self.local_vsync_cb.value' in config_mgr_text
    assert '"Local VSync": self.local_vsync_cb.value' not in config_mgr_text
    assert 'self.local_vsync_cb.label = t.get("VSync", "VSync")' in handlers_text
    assert '(self.local_vsync_cb, "tooltip_vsync")' in handlers_text
    assert '"VSync": "VSync"' in localization_text
    assert '"VSync": "垂直同步"' in localization_text
    assert '"tooltip_vsync"' in localization_text
    assert 'tooltip_local_vsync' not in localization_text


def test_gui_forces_fp16_off_for_mps_save():
    config_mgr_text = _file_text("config_mgr.py")

    assert 'fp16_value = False if "MPS" in (self.device_dd.value or "") else bool(self.fp16_cb.value)' in config_mgr_text
    assert '"FP16": fp16_value' in config_mgr_text
    assert '"FP16": self.fp16_cb.value' not in config_mgr_text


def test_accelerator_policy_matches_teammate_config_semantics():
    handlers_text = _file_text("handlers.py")
    config_mgr_text = _file_text("config_mgr.py")

    assert "def _platform_accelerator_values" in handlers_text
    assert "def _apply_platform_accelerator_policy" in handlers_text
    assert '"TensorRT": None' in handlers_text
    assert '"CoreML": None' in handlers_text
    assert '"OpenVINO": None' in handlers_text
    assert '"MIGraphX": None' in handlers_text
    assert "enabled = (saved_value is None) or bool(saved_value)" in handlers_text
    assert 'and self._config.get("torch.compile") is None' in handlers_text
    assert "self.auto_enable_optimizers_based_on_device()" in handlers_text
    assert "accelerator_values, recompile_values = self._platform_accelerator_values()" in config_mgr_text
    assert "**accelerator_values" in config_mgr_text
    assert "**recompile_values" in config_mgr_text
    assert '"TensorRT": self.tensorrt_cb.value' not in config_mgr_text
    assert '"MIGraphX": self.migraphx_cb.value' not in config_mgr_text
    assert '"CoreML": self.coreml_cb.value' not in config_mgr_text
    assert '"OpenVINO": self.openvino_cb.value' not in config_mgr_text
    assert 'trt_val = cfg.get("TensorRT")' in config_mgr_text
    assert 'if trt_val is not None:' in config_mgr_text
    assert 'mgx_val = cfg.get("MIGraphX")' in config_mgr_text
    assert 'cml_val = cfg.get("CoreML")' in config_mgr_text
    assert 'ov_val = cfg.get("OpenVINO")' in config_mgr_text


def test_default_depth_resolution_prefers_518_except_infinidepth():
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")

    assert '"Depth Resolution": 518' in config_text
    assert 'resolutions = ALL_MODELS.get(model_name, {}).get("resolutions", [DEFAULTS["Depth Resolution"]])' in builders_text
    assert 'preferred = 512 if "infinidepth" in str(model_name or "").lower() else DEFAULTS["Depth Resolution"]' in builders_text
    assert 'closest = min(resolutions, key=lambda x: abs(x - cur_num))' in builders_text
    assert '[322]' not in builders_text[builders_text.index("def update_depth_resolution_options"):]


def test_reset_defaults_uses_base_model_and_nvidia_acceleration_defaults():
    config_text = _config_source().read_text(encoding="utf-8")
    process_text = _file_text("process.py")

    assert "def default_base_depth_model" in config_text
    assert 'FAMILY_SIZE_TO_MODEL.get((default_family, "Base"))' in config_text
    assert 'if "Distill-Any-Depth-Base" in DEFAULT_MODEL_LIST:' in config_text
    assert 'dynamic_defaults["Depth Model"] = default_base_depth_model()' in process_text
    assert 'dynamic_defaults["XR Preview Window"] = False' in process_text
    assert 'is_nvidia_cuda = "CUDA" in (current_device_label or "") and not devices_module.IS_ROCM' in process_text
    assert 'dynamic_defaults["torch.compile"] = True' in process_text
    assert 'dynamic_defaults["TensorRT"] = True' in process_text


def test_stream_url_local_ip_detection_runs_async_after_gui_update():
    handlers_text = _file_text("handlers.py")
    gui_text = _file_text("gui.py")

    assert "import asyncio" in handlers_text
    assert "async def _refresh_local_ip_async" in handlers_text
    assert "await asyncio.to_thread(get_local_ip)" in handlers_text
    assert "def _schedule_local_ip_refresh" in handlers_text
    assert "asyncio.create_task(self._refresh_local_ip_async())" in handlers_text
    assert "def update_stream_url(self, e=None, resolve_ip=True):" in handlers_text
    update_block = handlers_text[handlers_text.index("def update_stream_url"):handlers_text.index("def _on_stream_protocol_change")]
    assert "get_local_ip()" not in update_block
    assert 'self._local_ip_cache = "127.0.0.1"' in gui_text
    assert "self._local_ip_task = None" in gui_text
