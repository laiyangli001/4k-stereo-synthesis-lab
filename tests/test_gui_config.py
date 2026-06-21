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
    assert '"Stereo Preset": self._display_to_preset(self.stereo_preset_dd.value),' in text
    assert '"Stereo Preset": DEFAULTS["Stereo Preset"],' not in text


def test_gui_hot_stereo_params_auto_save_on_select():
    all_text = _all_text()
    assert "def on_stereo_hot_param_change" in all_text
    assert "def _save_stereo_hot_params" in all_text
    assert 'on_select=self.on_stereo_hot_param_change' in _file_text("builders.py")
    assert '"IPD": self._parse_int(self.ipd_dd.value' in _file_text("config_mgr.py")
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


def test_gui_scene_preset_does_not_overwrite_reset_controls():
    text = _file_text("handlers.py")
    apply_start = text.index("def _apply_stereo_preset_values")
    block = text[apply_start:]
    assert "self.scene_reset_dd.value" not in block
    assert "self.reset_cooldown_dd.value" not in block
    assert '"scene_reset_threshold"' not in block
    assert '"reset_cooldown_frames"' not in block
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
    diag_write_index = text.index("with open(DIAG_LOG", filter_index)
    assert "return len(data or \"\")" in text[filter_index:console_write_index]
    assert filter_index < console_write_index < diag_write_index


def test_stereo_quality_dropdown_uses_localized_levels_but_saves_runtime_values():
    all_text = _all_text()
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"fast": "Lowest"' in localization_text
    assert '"fast_plus": "Medium"' in localization_text
    assert '"quality_4k": "High"' in localization_text
    assert '"hq_4k": "Highest"' in localization_text
    assert '"fast": "最低"' in localization_text
    assert '"fast_plus": "中等"' in localization_text
    assert '"quality_4k": "较高"' in localization_text
    assert '"hq_4k": "最高"' in localization_text
    assert "options=self._stereo_quality_options()" in all_text
    assert "stereo_quality_key = self._display_to_stereo_quality(self.stereo_quality_dd.value)" in handlers_text
    assert "self.stereo_quality_dd.options = self._stereo_quality_options()" in handlers_text
    assert '"Stereo Quality": self._display_to_stereo_quality(self.stereo_quality_dd.value)' in _file_text("config_mgr.py")
    assert '"Synthetic View": self._display_to_stereo_quality(self.stereo_quality_dd.value)' in _file_text("config_mgr.py")


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
    assert 'self.stereo_scale_label = ft.Text("Stereo Scale:"' in builders_text
    assert 'self.stereo_scale_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(1, 11)]' in builders_text
    row_start = builders_text.index('row3 = ft.Row([')
    row_end = builders_text.index('# Row 5: Stereo runtime mode and quality', row_start)
    row = builders_text[row_start:row_end]
    assert 'self.ipd_dd' in row
    assert 'self.stereo_scale_label' in row
    assert 'self.stereo_scale_dd' in row
    assert row.index('self.ipd_dd') < row.index('self.stereo_scale_label') < row.index('self.stereo_scale_dd')


def test_stereo_preset_auto_option_removed():
    all_text = _all_text()
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    assert '"Stereo Preset": "cinema"' in config_text
    assert 'options=["Cinema", "Game / Low Latency", "Still Image / HQ", "Debug / Export"]' in builders_text
    assert 'options=["Auto", "Cinema"' not in all_text
    assert '"Auto": "auto"' not in all_text
    assert '"自动": "auto"' not in all_text


def test_stereo_scale_has_tooltips():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"tooltip_stereo_scale": "Stereo strength multiplier applied to the physical IPD' in localization_text
    assert '"tooltip_stereo_scale": "作用在物理 IPD 上的立体强度倍率' in localization_text
    assert '(self.stereo_scale_dd, "tooltip_stereo_scale")' in all_text


def test_stereo_scale_label_is_localized():
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"Stereo Scale:": "Stereo Scale:"' in localization_text
    assert '"Stereo Scale:": "立体缩放:"' in localization_text
    assert 'self.stereo_scale_label.value = t["Stereo Scale:"]' in handlers_text


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
    assert 'self.temporal_strength_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 11)]' in builders_text
