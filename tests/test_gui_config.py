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



def test_gui_render_size_policy_is_fixed_to_scaled_for_load_and_save():
    config_text = _config_source().read_text(encoding="utf-8")
    config_mgr_text = _file_text("config_mgr.py")
    handlers_text = _file_text("handlers.py")
    builders_text = _file_text("builders.py")

    assert '"Render Size Policy": "scaled"' in config_text
    assert 'def _render_policy_options' in handlers_text
    assert 'def _render_policy_to_display' in handlers_text
    assert 'def _display_to_render_policy' in handlers_text
    assert 'return UI_MESSAGES[self.locale].get("Scaled", "Scaled")' in handlers_text
    assert 'return "scaled"' in handlers_text
    assert '"native": "native"' not in handlers_text
    assert '"fixed": "fixed"' not in handlers_text
    assert '"dynamic": "dynamic"' not in handlers_text
    assert 'self.render_policy_dd = CompactDropdown(' in builders_text
    assert 'options=["Scaled"]' in builders_text
    assert 'cfg.get("Render Size Policy", DEFAULTS["Render Size Policy"])' in config_mgr_text
    assert '"Render Size Policy": "scaled",' in config_mgr_text
    assert '"Render Size Policy": self._display_to_render_policy(self.render_policy_dd.value)' not in config_mgr_text


def test_gui_hot_stereo_params_auto_save_on_select():
    all_text = _all_text()
    config_mgr_text = _file_text("config_mgr.py")
    builders_text = _file_text("builders.py")
    assert "def on_stereo_hot_param_change" in all_text
    assert "def _save_stereo_hot_params" in all_text
    assert 'on_select=self.on_stereo_hot_param_change' in builders_text
    assert '"IPD":' not in config_mgr_text
    assert '"Stereo Scale":' not in config_mgr_text
    assert '"Max Shift Ratio":' not in config_mgr_text
    assert '"Parallax Budget Preset": parallax_budget' in config_mgr_text
    assert '"Convergence": self._parse_float(self.convergence_dd.value' in config_mgr_text
    assert '"Depth Strength": self._clamp_depth_strength(self.depth_strength_dd.value)' in config_mgr_text
    assert '"Temporal Strength": temporal_strength' in config_mgr_text
    assert '"Scene Reset Threshold": scene_reset_threshold' in config_mgr_text
    assert "self.temporal_strength_label, self.temporal_strength_dd,\n            ft.Container(width=S(40)), self.scene_reset_label, self.scene_reset_dd" in builders_text
    assert ("Reset " + "Cooldown " + "Frames") not in config_mgr_text
    assert "def _clamp_foreground_scale" in all_text
    assert "fg_options = [f\"{i / 10:.1f}\" for i in range(-9, 0)]" in builders_text
    assert '"Foreground Scale": foreground_scale' in config_mgr_text
    assert '"Depth Antialias Strength": antialias_strength' in config_mgr_text
    assert '"Edge Dilation": self._parse_int(self.edge_dilation_dd.value' in config_mgr_text
    assert '"Mask Feather Radius": self._parse_int(self.mask_feather_dd.value' in config_mgr_text
    assert '"Hole Fill Mode": self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value)' in config_mgr_text
    assert '"Hole Fill Radius": int(preset_values.get("hole_fill_radius", DEFAULTS["Hole Fill Radius"]))' in config_mgr_text
    assert '"Hole Fill Strength": float(preset_values.get("hole_fill_strength", DEFAULTS["Hole Fill Strength"]))' in config_mgr_text
    assert '"Edge Threshold": self._parse_float(self.edge_threshold_dd.value' in config_mgr_text
    assert '"Anaglyph Method": self.anaglyph_dd.value' in config_mgr_text
    assert '"Cross Eyed": bool(self.cross_eyed_cb.value)' in config_mgr_text
    assert 'on_change=self.on_stereo_hot_param_change' in builders_text
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
    assert '"scene_reset_threshold"' in _file_text("config_mgr.py")
    assert ("reset_" + "cooldown" + "_frames") not in _file_text("config_mgr.py")
    assert "self.parallax_budget_dd.value" in block
    assert "self.stereo_scale_dd.value" not in block
    assert "on_select=self.on_stereo_preset_change" in _file_text("builders.py")


def test_advanced_stereo_is_not_persisted_and_starts_collapsed():
    all_text = _all_text()
    cfg_text = _file_text("config_mgr.py")
    builders_text = _file_text("builders.py")
    assert '"Advanced Stereo": False' not in all_text
    assert "self.hole_fill_mode_label, self.hole_fill_mode_dd,\n            ft.Container(width=S(40)), self.advanced_stereo_cb" in builders_text
    assert 'label="Advanced Stereo", value=False, on_change=self.on_advanced_stereo_change' in builders_text
    assert "advanced_stereo_row =" not in builders_text
    assert 'self.advanced_stereo_cb.value = False' in cfg_text
    assert '"Advanced Stereo": self.advanced_stereo_cb.value' not in all_text
    assert 'cfg.get("Advanced Stereo"' not in all_text


def test_parallax_budget_is_visible_next_to_stereo_mode_without_advanced_stereo():
    builders_text = _file_text("builders.py")
    row_start = builders_text.index("stereo_row0 = ft.Row([")
    row_end = builders_text.index("], spacing=1)", row_start)
    stereo_row = builders_text[row_start:row_end]
    assert "self.stereo_preset_label" in stereo_row
    assert "self.stereo_preset_dd" in stereo_row
    assert "self.parallax_budget_label" in stereo_row
    assert "self.parallax_budget_dd" in stereo_row
    assert stereo_row.index("self.stereo_preset_dd") < stereo_row.index("self.parallax_budget_label")
    depth_start = builders_text.index("depth_group = ft.Container(")
    depth_end = builders_text.index("device_group = ft.Container(", depth_start)
    depth_block = builders_text[depth_start:depth_end]
    assert "stereo_row0" in depth_block
    assert "self.parallax_budget_dd" not in builders_text[builders_text.index("self._advanced_stereo_rows = ["):]


def test_advanced_device_options_is_not_persisted_and_starts_collapsed():
    all_text = _all_text()
    cfg_text = _file_text("config_mgr.py")
    builders_text = _file_text("builders.py")
    assert '"Advanced Device Options": False' not in all_text
    assert 'label="Advanced Options", value=False, on_change=self.on_advanced_device_change' in builders_text
    assert 'self.advanced_device_cb.value = False' in cfg_text
    assert '"Advanced Device Options": self.advanced_device_cb.value' not in all_text
    assert 'cfg.get("Advanced Device Options"' not in all_text


def test_xr_preview_window_is_advanced_next_to_capture_fps():
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert 'self.xr_preview_cb = ft.Checkbox(label="XR Preview Window"' in builders_text
    row_start = builders_text.index("self.row6b = ft.Row([")
    row_end = builders_text.index("], spacing=1)", row_start)
    row6b = builders_text[row_start:row_end]
    assert "self.target_fps_label" in row6b
    assert "self.target_fps_dd" in row6b
    assert "self.xr_preview_cb" in row6b
    assert row6b.index("self.target_fps_dd") < row6b.index("self.xr_preview_cb")

    row7_start = builders_text.index("self.row7a = ft.Row([")
    row7_end = builders_text.index("], spacing=1)", row7_start)
    assert "self.xr_preview_cb" not in builders_text[row7_start:row7_end]
    assert 'self.xr_preview_cb.visible = advanced and mode == "OpenXR Link"' in handlers_text
    assert "self.xr_preview_cb.visible = is_openxr" not in handlers_text
    assert '(self.xr_preview_cb, "tooltip_xr_preview")' in handlers_text
    assert '"tooltip_xr_preview"' in localization_text


def test_gui_requirements_do_not_install_terminal_styling_or_flet():
    repo_root = Path(__file__).resolve().parents[1]
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    risky_requirements = (repo_root / "requirements-risky.txt").read_text(encoding="utf-8")
    terminal_styling_package = "ri" + "ch"

    assert terminal_styling_package not in requirements.lower()
    assert terminal_styling_package not in risky_requirements.lower()
    assert "flet==" not in requirements
    assert "flet-desktop==" not in requirements
    assert "flet-cli==" not in requirements

def test_hidden_log_panel_does_not_contribute_to_window_width():
    builders_text = _file_text("builders.py")
    fit_start = builders_text.index("def _fit_window_to_content")
    fit_end = builders_text.index("def _on_page_resize", fit_start)
    fit_block = builders_text[fit_start:fit_end]

    assert "if log_panel.visible:" in fit_block
    assert "log_panel.expand = True" in fit_block
    assert "log_panel.width = self._estimate_log_panel_width(main_width)" in fit_block
    assert "log_panel.expand = False" in fit_block
    assert "log_panel.width = 0" in fit_block

    width_start = builders_text.index("def _estimate_window_width")
    width_end = builders_text.index("def _control_has_effective_content", width_start)
    width_block = builders_text[width_start:width_end]
    assert "self._estimate_log_panel_width" not in width_block
    assert "self.log_panel.width" not in width_block


def test_console_output_uses_standard_logging_with_filtered_stream_redirect():
    text = _file_text("process.py")
    assert "def _is_key_console_output" in text
    assert "[NativeUtil] sogou_native_util_pc loaded successfully" in text
    assert "[warmup] same version" in text
    assert "[INFO] [flet] Session was garbage collected:" in text
    assert "_DEBUG_CONSOLE_PREFIXES" in text
    assert "_console_logging_installed = False" in text
    assert "if _console_logging_installed:" in text
    assert "from " + "ri" + "ch" not in text
    assert "Ri" + "chHandler(" not in text
    assert "console=Console(file=console_stream)" not in text
    assert "console_stream = sys.__stderr__ or sys.stderr or open(os.devnull" in text
    assert "console_handler = logging.StreamHandler(console_stream)" in text
    assert "class _StreamToLogger" in text
    stream_index = text.index("class _StreamToLogger")
    filter_index = text.index("if line and _is_key_console_output(line):", stream_index)
    logger_index = text.index("self.stream_logger.log(self.level, line)", filter_index)
    assert stream_index < filter_index < logger_index


def test_console_filter_suppresses_flet_debug_patch_noise():
    logging_setup_text = (Path(__file__).resolve().parents[1] / "src" / "utils" / "logging_setup.py").read_text(encoding="utf-8")
    process_text = _file_text("process.py")

    assert 'record.name in ("flet", "flet_desktop", "flet_controls", "flet_transport")' in logging_setup_text
    assert "record.levelno <= logging.DEBUG" in logging_setup_text
    assert "class _FletInfoAsDebugFilter" in process_text
    assert 'record.name in _FLET_LOGGER_NAMES' in process_text
    assert 'record.getMessage().startswith(_FLET_MESSAGE_PREFIXES)' in process_text
    assert "record.levelno = logging.DEBUG" in process_text
    assert "console_handler.addFilter(_FletInfoAsDebugFilter())" in process_text
    assert "file_handler.addFilter(_FletInfoAsDebugFilter())" in process_text
    assert "gui_handler.addFilter(_FletInfoAsDebugFilter())" in process_text
    assert "console_handler.addFilter(_NoisyThirdPartyDebugFilter())" in process_text


def test_console_filter_suppresses_flet_startup_noise_lines():
    import ast

    source = _file_text("process.py")
    tree = ast.parse(source)
    noisy_prefixes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "_NOISY_CONSOLE_PREFIXES" for target in node.targets):
            noisy_prefixes = list(ast.literal_eval(node.value))
            break

    assert any("[NativeUtil] sogou_native_util_pc loaded successfully on %s win32".startswith(prefix) for prefix in noisy_prefixes)
    assert any("[warmup] same version (1.0.263), skip renderer warmup".startswith(prefix) for prefix in noisy_prefixes)
    assert not any("[Main] Runtime preparation: checking depth model lc700x/InfiniDepth-Large".startswith(prefix) for prefix in noisy_prefixes)


def test_gui_window_updates_after_layout_and_before_vendored_flet_preparation():
    text = _file_text("gui.py")
    main_block = text[text.index("def main():"):text.index("async def _async_main")]
    setup_block = text[text.index("async def setup(self):"):text.index("async def _prepare_startup_after_window_visible")]

    assert "ensure_vendored_flet_view()" not in main_block
    assert main_block.index("_setup_console_logging()") < main_block.index("ft.run(_async_main)")
    show_idx = setup_block.index("self.page.window.visible = True")
    update_idx = setup_block.index("self.page.update()", show_idx)
    ready_idx = setup_block.index("self._signal_gui_ready()", update_idx)
    task_idx = setup_block.index("asyncio.create_task(self._prepare_startup_after_window_visible())", ready_idx)
    populate_idx = setup_block.index("self.populate_monitors()")
    fit_idx = setup_block.index("self._fit_window_to_content(update=False)")
    log_visibility_idx = setup_block.index('self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]), update=False)')
    assert populate_idx < log_visibility_idx < fit_idx < show_idx < update_idx < ready_idx < task_idx
    assert "def _signal_gui_ready(self):" in text
    assert "await asyncio.to_thread(ensure_vendored_flet_view)" in text


def test_run_windows_waits_for_gui_ready_signal_before_closing_cmd():
    run_bat = Path(__file__).resolve().parents[1] / "run_windows.bat"
    data = run_bat.read_bytes()
    text = run_bat.read_text(encoding="utf-8")

    assert data[:3] != b"\xef\xbb\xbf"
    assert data.count(b"\n") == data.count(b"\r\n")
    assert r'set "PYTHON_EXE=%APP_DIR%\python3\python.exe"' in text
    assert r'set "LOG_DIR=%APP_DIR%\logs"' in text
    assert r'set "GUI_READY_FILE=%LOG_DIR%\gui_ready.flag"' in text
    assert r'set "LAUNCH_STDOUT=%LOG_DIR%\launcher_stdout.log"' in text
    assert r'set "LAUNCH_STDERR=%LOG_DIR%\launcher_stderr.log"' in text
    assert r'set "APP_LOG=%LOG_DIR%\desktop2stereo.log"' in text
    assert 'if exist "%GUI_READY_FILE%" del /f /q "%GUI_READY_FILE%"' in text
    assert 'if exist "%APP_LOG%" type nul > "%APP_LOG%"' in text
    assert "taskkill /f /t /im python.exe" in text
    assert "taskkill /f /t /im pythonw.exe" in text
    assert text.index("taskkill /f /t /im python.exe") < text.index("Start-Process -FilePath '%PYTHON_EXE%'")
    assert 'set "PYTHONPATH=%APP_DIR%"' in text
    assert "Start-Process -FilePath '%PYTHON_EXE%'" in text
    assert "-WindowStyle Hidden" in text
    assert "-RedirectStandardOutput '%LAUNCH_STDOUT%'" in text
    assert "-RedirectStandardError '%LAUNCH_STDERR%'" in text
    assert "Test-Path -LiteralPath '%GUI_READY_FILE%'" in text
    assert "if ($p.HasExited) { exit 1 }" in text
    assert "AddSeconds(60)" in text
    assert "exit 2" in text
    assert "if errorlevel 2 goto launch_timeout" in text
    assert "if errorlevel 1 goto launch_failed" in text
    assert "未在 60 秒内回传就绪标志" in text
    assert "在回传就绪标志前失败" in text
    show_logs = text[text.index(":show_logs"):]
    assert "pause" in show_logs
    assert "launcher_stderr.log" in show_logs
    assert "launcher_stdout.log" not in show_logs
    assert "desktop2stereo.log" not in show_logs
    assert "%PYTHON_EXE% -m gui" not in text


def test_gui_suppresses_known_asyncio_shutdown_unraisable_noise_only():
    text = _file_text("process.py")

    assert "def _install_asyncio_shutdown_noise_filter" in text
    assert "sys.unraisablehook" in text
    assert "asyncio.base_subprocess" in text
    assert "asyncio.proactor_events" in text
    assert "Event loop is closed" in text
    assert "I/O operation on closed pipe" in text
    assert "qualname.endswith(\".__del__\")" in text
    assert "previous_hook(unraisable)" in text
    setup_block = text.split("def _setup_console_logging():", 1)[1]
    setup_block = setup_block.split("class GUIProcessMixin:", 1)[0]
    assert "_install_asyncio_shutdown_noise_filter()" in setup_block


def test_debug_mode_gui_control_removed():
    all_text = _all_text()
    assert "debug_mode_cb" not in all_text
    assert 'label="Debug Mode"' not in all_text
    assert '"Debug Mode": False' not in all_text
    assert '"Debug Mode": self.' not in all_text
    assert 'cfg.get("Debug Mode"' not in all_text


def test_gui_uses_single_rolling_log_for_gui_and_child_output():
    paths_text = _file_text("paths.py")
    process_text = _file_text("process.py")
    gui_text = _file_text("gui.py")
    builders_text = _file_text("builders.py")
    config_text = _file_text("config.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _file_text("localization.py")
    assert 'LOG_FILE = os.path.join(LOG_DIR, "desktop2stereo.log")' in paths_text
    assert "DIAG_LOG = LOG_FILE" in paths_text
    assert "logging.FileHandler(LOG_FILE, mode=\"w\", encoding=\"utf-8\")" in process_text
    assert "GuiLogHandler(maxlen=2000)" in process_text
    assert "self.gui_log_handler = _setup_console_logging()" in gui_text
    assert "self._log_poll_task = asyncio.create_task(self._poll_log_queue())" in gui_text
    assert "self.log_panel = ft.Container" in builders_text
    assert "self.log_listview = ft.Column(scroll=ft.ScrollMode.AUTO" in builders_text
    assert "content=ft.Row([self.log_listview], scroll=ft.ScrollMode.AUTO" in builders_text
    assert "ft.ListView" not in builders_text
    assert "SelectionArea" not in builders_text
    assert "visible=False" in builders_text
    assert "page.add(ft.Row([self._main_panel, self.log_panel]" in builders_text
    assert "self.log_title = ft.Text" not in builders_text
    assert "self.log_clear_btn = ft.Button" not in builders_text
    assert "self.log_toggle_btn = ft.Button" not in builders_text
    assert "def _estimate_main_panel_width" in builders_text
    assert "def _estimate_log_panel_width" in builders_text
    assert "self._main_panel.width = main_width" in builders_text
    assert "log_panel.width = self._estimate_log_panel_width(main_width)" in builders_text
    assert "def _fit_window_to_content(self, update=True, resize_window=False)" in builders_text
    assert "if resize_window:" in builders_text
    assert "self.page.window.min_width = main_width" in builders_text
    assert "self.page.window.min_width = width" not in builders_text
    assert "min(width, S(520))" not in builders_text
    assert "self.page.window.width = width" in builders_text
    assert "self.page.window.update()" in builders_text
    assert "except RuntimeError:" in builders_text
    assert "self.page.window.max_width = width" in builders_text
    assert "self.page.on_resize = self._on_page_resize" in gui_text
    assert "max_total_width" not in builders_text
    assert "min_log_width = S(500)" in builders_text
    assert "available_width = window_width - main_width" in builders_text
    assert "controls[-200:]" in builders_text
    assert "self._fit_window_to_content(update=False)" in process_text
    assert "min_width=S(360)" not in builders_text
    assert "min_width=S(500)" not in builders_text
    assert "width=S(360)" not in builders_text
    assert "width=S(500)" in builders_text
    assert "expand=True" in builders_text
    assert 'UI_MESSAGES[self.locale].get("Report issue", "Report")' not in builders_text
    assert 'UI_MESSAGES[self.locale].get("Report issue", "Report bug")' in builders_text
    assert 'UI_MESSAGES[self.locale].get("Open log file", "Open log")' in builders_text
    assert "on_click=self.on_open_log_file" in builders_text
    assert builders_text.count("width=S(150)") >= 2
    assert "width=S(86)" not in builders_text
    assert 'self.report_issue_btn.content.value = t.get("Report issue", "Report bug")' in handlers_text
    assert 'self.open_log_file_btn.content.value = t.get("Open log file", "Open log")' in handlers_text
    assert '"Report issue": "反馈bug"' in localization_text
    assert '"Open log file": "查看log文件"' in localization_text
    assert "def on_open_log_file" in process_text
    assert '"-m", "gui.' + 'ri' + 'ch_log_viewer", LOG_FILE' not in process_text
    assert 'import webview' not in process_text
    assert "def _sync_" + "ri" + "ch_log_viewer" not in process_text
    assert "def _start_" + "ri" + "ch_log_viewer" not in process_text
    assert "def _stop_" + "ri" + "ch_log_viewer" not in process_text
    assert "self._sync_" + "ri" + "ch_log_viewer(panel.visible)" not in process_text
    assert "self._stop_" + "ri" + "ch_log_viewer()" not in process_text
    assert "os.startfile(LOG_FILE)" in process_text
    open_log_block = process_text.split("def on_open_log_file", 1)[1].split("    # ── reset ──", 1)[0]
    assert '"-m", "gui.' + 'ri' + 'ch_log_viewer", LOG_FILE' not in open_log_block
    assert "pywebview" not in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    assert 'self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]))' in process_text
    assert "self._fit_window_to_content(update=update, resize_window=True)" in process_text
    assert 'stdout=asyncio.subprocess.PIPE' in process_text
    assert 'stderr=asyncio.subprocess.STDOUT' in process_text
    assert 'child_env["PYTHONIOENCODING"] = "utf-8"' in process_text
    assert "async def _pump_child_output" in process_text
    assert "raw = await stream.read(4096)" in process_text
    assert "raw = await stream.readline()" not in process_text
    assert "child_logger.info(text)" in process_text
    assert "sys.stdout.write(text)" not in process_text
    assert "print(text)" not in process_text
    assert "asyncio.create_task(self._pump_child_output(self.process))" in process_text
    assert "from " + "ri" + "ch" not in process_text
    assert "from .controls import S" in process_text
    assert "def _log_text_width" in process_text
    assert "no_wrap=True" in process_text
    assert "max_lines=1" in process_text
    assert "overflow=ft.TextOverflow.VISIBLE" in process_text
    assert "width=self._log_text_width(line)" in process_text
    assert "def _progress_log_line" in process_text
    assert "def _progress_event" in process_text
    assert "def _update_download_progress" in process_text
    assert "if _PROGRESS_PREFIX not in text:" in process_text
    assert "text = text[text.index(_PROGRESS_PREFIX):]" in process_text
    assert "class _HideStructuredProgressFilter" in process_text
    assert process_text.count("addFilter(_HideStructuredProgressFilter())") == 2
    assert "self.download_progress_bar = ft.ProgressBar" in builders_text
    assert "self.download_progress_panel" in builders_text
    assert "_PROGRESS_PREFIX = \"[D2S_PROGRESS] \"" in process_text
    assert 'controls = getattr(self, "_progress_log_controls", {})' in process_text
    assert "existing.value = line" in process_text
    assert "self.log_listview.controls.append(text)" in process_text
    assert "self._progress_log_controls.clear()" in process_text
    assert "def _read_log_file_items" in process_text
    assert "items = _read_log_file_items()" in process_text
    assert "if not items and handler is not None:" in process_text
    assert 'if value == "STATUS":' in process_text
    assert 'getattr(handler, "status_cache", items)' in process_text
    assert 'handler.status_cache.clear()' in process_text
    assert '"Show Log Panel": True' in config_text
    assert "self.log_visibility_link" in builders_text
    assert "on_click=self.on_log_visibility_link" in builders_text
    assert "def on_log_visibility_link" in process_text
    assert "asyncio.create_task(self._resize_window_after_log_visibility_change())" in process_text
    assert "async def _resize_window_after_log_visibility_change" in process_text
    assert "await asyncio.sleep(0)" in process_text
    assert "self._fit_window_to_content(update=True, resize_window=True)" in process_text
    assert "await asyncio.sleep(0.5)" in process_text
    assert "self.page.window.max_width = None" in process_text
    assert 'cfg["Show Log Panel"] = panel.visible' in process_text
    assert "save_yaml(path, cfg)" in process_text
    assert "_log_panel_preference_seen" not in gui_text
    assert 'update=False' in gui_text
    assert "self._sync_log_visibility_link()" in handlers_text
    assert '"Hide log panel link": "隐藏log窗口->"' in localization_text
    assert '"Show log panel link": "显示log窗口->"' in localization_text


def test_gui_stop_uses_stop_request_file_before_force_kill():
    paths_text = _file_text("paths.py")
    process_text = _file_text("process.py")
    assert 'STOP_REQUEST_FILE = os.path.join(LOG_DIR, "stop.request")' in paths_text
    assert "with open(STOP_REQUEST_FILE, \"w\", encoding=\"utf-8\")" in process_text
    assert "await asyncio.wait_for(proc.wait(), timeout=1)" in process_text
    assert "async def _kill_process_tree" in process_text
    graceful_index = process_text.index("with open(STOP_REQUEST_FILE")
    kill_index = process_text.index("await self._kill_process_tree(proc, saved_pid)", graceful_index)
    assert graceful_index < kill_index


def test_gui_close_force_kills_child_process_tree_without_waiting_for_stop_file():
    process_text = _file_text("process.py")
    close_block = process_text.split("async def _async_stop(self):", 1)[1].split("status_logger.info", 1)[0]

    assert "force_kill = self._closed" in close_block
    assert "if force_kill:" in close_block
    assert "await self._kill_process_tree(proc, saved_pid)" in close_block
    assert "if self._closed and self.process and self.process.returncode is None:" in close_block
    assert "await self._kill_process_tree(proc, proc.pid)" in close_block


def test_safe_update_skips_missing_controls_without_warning_spam():
    handlers_text = _file_text("handlers.py")
    safe_update = handlers_text.split("def _safe_update(self, *controls):", 1)[1]
    safe_update = safe_update.split("    # ── stream URL ──", 1)[0]

    assert "if c is None:" in safe_update
    assert safe_update.index("if c is None:") < safe_update.index("c.update()")


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


def test_gui_environment_discovery_accepts_panorama_image_folders():
    config_text = _config_source().read_text(encoding="utf-8")

    assert "def _find_env_image_for_gui" in config_text
    assert '"background", "panorama", "equirectangular", "360", "sky", "skybox"' in config_text
    assert '".hdr"' in config_text
    assert "or _find_env_image_for_gui(room_dir)" in config_text


def test_parallax_budget_control_replaces_ipd_and_stereo_scale_controls():
    builders_text = _file_text("builders.py")
    assert "self.ipd_dd" not in builders_text
    assert "self.stereo_scale_label" not in builders_text
    assert "self.stereo_scale_dd" not in builders_text
    assert "self.max_shift_dd" not in builders_text
    assert 'self.parallax_budget_label = ft.Text("Parallax Budget:"' in builders_text
    assert 'self.parallax_budget_dd = CompactDropdown(options=self._parallax_budget_options()' in builders_text


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


def test_ipd_display_helpers_removed_from_gui_config():
    config_mgr_text = _file_text("config_mgr.py")
    assert "_IPD_RUNTIME_PER_DISPLAY_MM" not in config_mgr_text
    assert "_runtime_ipd_to_display_mm" not in config_mgr_text
    assert "_display_ipd_mm_to_runtime_m" not in config_mgr_text


def test_parallax_budget_standard_display_is_localized():
    from gui.localization import display_to_parallax_budget, parallax_budget_options, parallax_budget_to_display

    assert parallax_budget_to_display("standard", "EN") == "Standard"
    assert parallax_budget_to_display("standard", "CN") == "标准"
    assert parallax_budget_options("EN") == ["Comfort", "Standard", "Strong", "Extreme"]
    assert display_to_parallax_budget("Standard") == "standard"
    assert display_to_parallax_budget("标准") == "standard"


def test_parallax_budget_has_tooltips_without_legacy_ipd_scale_tooltips():
    all_text = _all_text()
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"tooltip_parallax_budget": "Maximum stereo parallax budget resolved from render size' in localization_text
    assert '"tooltip_parallax_budget": "根据渲染尺寸解析最大视差预算' in localization_text
    assert '"tooltip_ipd"' not in localization_text
    assert '"tooltip_stereo_scale"' not in localization_text
    assert '(self.parallax_budget_dd, "tooltip_parallax_budget")' in all_text


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
    builders_text = _file_text("builders.py")
    config_mgr_text = _file_text("config_mgr.py")
    assert 'ds_options = [f"{i / 100:.2f}" for i in range(0, 51, 5)]' in builders_text
    assert 'value="0.25"' in builders_text
    handlers_text = _file_text("handlers.py")
    assert 'def _clamp_depth_strength' in config_mgr_text
    assert 'self.depth_strength_dd.value = f"{values[\'depth_strength\']:.2f}"' in handlers_text
    assert 'self.depth_strength_dd.value = f"{strength:.2f}"' in handlers_text
    assert 'Use Standard / 0.25 as the baseline' in localization_text
    assert 'foreground objects show ghosts' in localization_text
    assert 'Depth-map smoothing level' in localization_text
    assert 'soften fine geometry' in localization_text
    assert '建议以标准档 0.25 为基准' in localization_text
    assert '前景重影、边缘撕裂或观看不舒服时下调' in localization_text
    assert '深度图平滑级别' in localization_text
    assert '游戏和实时观看保持较低' in localization_text


def test_convergence_dropdown_uses_five_percent_steps():
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    assert 'conv_options = [f"{i / 100:.2f}" for i in range(-50, 101, 5)]' in builders_text
    assert 'options=[v for v in conv_options], value="0.00"' in builders_text
    assert 'self.convergence_dd.value = f"{values[\'convergence\']:.2f}"' in handlers_text


def test_parallax_budget_label_is_localized():
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"Parallax Budget:": "Parallax Budget:"' in localization_text
    assert '"Parallax Budget:": "视差预算:"' in localization_text
    assert '"Stereo Scale:"' not in localization_text
    assert '"Convergence:": "会聚位置:"' in localization_text
    assert '"Convergence:": "会聚点:"' not in localization_text
    assert 'self.parallax_budget_label.value = t["Parallax Budget:"]' in handlers_text


def test_cn_antialiasing_labels_are_user_facing_without_legacy_ipd():
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"IPD (m):"' not in localization_text
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


def test_edge_threshold_options_are_dense_without_legacy_shift_ratio():
    builders_text = _file_text("builders.py")
    localization_text = _localization_source().read_text(encoding="utf-8")
    assert '"Max Shift Ratio:"' not in localization_text
    assert "self.max_shift_dd" not in builders_text
    assert 'self.edge_threshold_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)]' in builders_text
    assert 'self.mask_feather_dd = CompactDropdown(options=["0", "1", "2", "3", "4", "5"]' in builders_text
    assert 'self.temporal_strength_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 11)]' in builders_text


def test_mask_feather_radius_gui_control_is_localized_and_hot_reloadable():
    builders_text = _file_text("builders.py")
    config_text = _config_source().read_text(encoding="utf-8")
    config_mgr_text = _file_text("config_mgr.py")
    handlers_text = _file_text("handlers.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '"Mask Feather Radius": 1' in config_text
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
    assert '"Depth Strength": 0.25' in config_text
    assert '"Depth Quick": "Standard"' in config_text
    assert '"IPD":' not in config_text
    assert '"Foreground Scale": 0.0' in config_text
    assert '"Convergence": 0.0' in config_text
    assert '"Parallax Budget Preset": "standard"' in config_text
    assert '"Stereo Scale":' not in config_text
    assert '"Anti-aliasing": 1' in config_text
    assert '"Depth Antialias Strength": 1.0' in config_text
    assert '"Temporal Strength": 0.25' in config_text
    assert '"Edge Dilation": 1' in config_text
    assert '"Mask Feather Radius": 1' in config_text
    assert '"Hole Fill Mode": "balanced"' in config_text
    assert '"Hole Fill Radius": 1' in config_text
    assert '"Hole Fill Strength": 0.60' in config_text
    assert "dynamic_defaults = DEFAULTS.copy()" in process_text
    assert "self.apply_config(dynamic_defaults, keep_optional=False)" in process_text


def test_stereo_mode_presets_keep_expected_parallax_budget_combinations():
    config_mgr_text = _file_text("config_mgr.py")

    assert '"traditional_fastest": {\n                "quality": "fast", "parallax_budget": "standard"' in config_mgr_text
    assert '"cinema": {\n                "quality": "quality_4k", "parallax_budget": "standard"' in config_mgr_text
    assert '"game_low_latency": {\n                "quality": "fast_plus", "parallax_budget": "comfort"' in config_mgr_text
    assert '"still_image_hq": {\n                "quality": "hq_4k", "parallax_budget": "strong"' in config_mgr_text
    assert '"cinema": {\n                "quality": "quality_4k", "parallax_budget": "standard", "depth_strength": 0.25' in config_mgr_text
    assert '"edge_dilation": 1, "mask_feather_radius": 1, "hole_fill_mode": "balanced",' in config_mgr_text
    assert '"hole_fill_radius": 1, "hole_fill_strength": 0.60' in config_mgr_text
    assert '"game_low_latency": {\n                "quality": "fast_plus", "parallax_budget": "comfort", "depth_strength": 0.20' in config_mgr_text
    assert '"temporal_strength": 0.0, "scene_reset_threshold": 0.18' in config_mgr_text
    assert '"still_image_hq": {\n                "quality": "hq_4k", "parallax_budget": "strong", "depth_strength": 0.30' in config_mgr_text
    assert '"edge_dilation": 3, "mask_feather_radius": 3, "hole_fill_mode": "quality",' in config_mgr_text


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
    assert 'HOLE_FILL_MODE_KEYS = ("balanced", "soft_low_ghost", "sharp_test", "quality")' in localization_text
    assert '"Balanced / Standard": "均衡 / 标准"' in localization_text
    assert '"Sharp / High Detail": "锐利 / 高细节"' in localization_text
    assert '"Content Aware / Highest Quality": "内容感知 / 最高质量"' in localization_text
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

    assert namespace["hole_fill_mode_options"]("EN") == [
        "Balanced / Standard",
        "Soft / Low Ghost",
        "Sharp / High Detail",
        "Content Aware / Highest Quality",
    ]
    assert namespace["hole_fill_mode_options"]("CN") == ["均衡 / 标准", "柔和 / 低重影", "锐利 / 高细节", "内容感知 / 最高质量"]
    assert namespace["display_to_hole_fill_mode"]("Balanced / Standard") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("均衡 / 标准") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("Balanced") == "balanced"
    assert namespace["display_to_hole_fill_mode"]("锐利测试") == "sharp_test"
    assert namespace["display_to_hole_fill_mode"]("Sharp Test") == "sharp_test"
    assert namespace["display_to_hole_fill_mode"]("内容感知 / 最高质量") == "quality"
    assert namespace["display_to_hole_fill_mode"]("Content Aware / Highest Quality") == "quality"


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


def test_gui_render_size_controls_expose_only_fixed_4k_scale_tiers():
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    config_mgr_text = _file_text("config_mgr.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    for key, value in {
        "Render Size Policy": "scaled",
        "Render Scale": "4K / 100%",
        "Render Fixed Width": 1920,
        "Render Fixed Height": 1080,
        "Render Min Dimension": 480,
        "Render Align": 8,
    }.items():
        expected = f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {value}'
        assert expected in config_text
    assert '"Render Max Pixels": 3840 * 2160' in config_text

    assert 'self.render_policy_dd = CompactDropdown(' in builders_text
    assert 'options=["Scaled"]' in builders_text
    assert 'self.render_policy_dd.visible = False' in builders_text
    assert 'self.render_scale_dd = CompactDropdown(options=self._render_scale_options()' in builders_text
    assert '"4K / 100%"' in handlers_text
    assert '"3K / 85%"' in handlers_text
    assert '"2K / 75%"' in handlers_text
    assert '"1K / 50%"' in handlers_text
    assert 'def _display_to_render_scale' in handlers_text
    assert 'def _render_scale_to_display' in handlers_text
    assert 'return float(match.group(0))' not in handlers_text
    assert 'compact.startswith(tier_name)' not in handlers_text
    assert 'resolution.upper() in compact' not in handlers_text
    assert '"1K / 50%": 0.5' not in handlers_text
    assert 'self.render_fixed_dd.visible = False' in builders_text
    assert 'self.row6d = ft.Row([self.render_scale_label, self.render_scale_dd' in builders_text
    row6d_start = builders_text.index('self.row6d = ft.Row([')
    row6d_end = builders_text.index('], spacing=1)', row6d_start)
    row6d = builders_text[row6d_start:row6d_end]
    assert row6d.index('self.render_scale_dd') < row6d.index('self.render_align_label')
    assert 'self.row6e.visible = False' in handlers_text
    assert 'self.row6f.visible = False' in handlers_text
    assert 'def on_render_policy_change' in handlers_text
    assert 'def _update_render_size_control_visibility' in handlers_text
    assert 'self._update_render_size_control_visibility(show_render_size)' in handlers_text

    assert 'cfg.get("Render Size Policy", DEFAULTS["Render Size Policy"])' in config_mgr_text
    assert 'cfg.get("Render Scale", DEFAULTS["Render Scale"])' in config_mgr_text
    assert '"Render Size Policy": "scaled",' in config_mgr_text
    assert '"Render Size Policy": self._display_to_render_policy(self.render_policy_dd.value)' not in config_mgr_text
    assert '"Render Scale": self._display_to_render_scale(self.render_scale_dd.value)' in config_mgr_text
    assert 'self.render_scale_dd.value = self._render_scale_to_display(' in config_mgr_text
    assert '"Render Align": self._parse_int(self.render_align_dd.value, DEFAULTS["Render Align"])' in config_mgr_text

    assert '"Render Scale:": "4K Render Scale:"' in localization_text
    assert '"Render Scale:": "4K缩放档位:"' in localization_text
    assert '"tooltip_render_scale"' in localization_text
    assert '(self.render_scale_dd, "tooltip_render_scale")' in handlers_text
    assert '(self.render_align_dd, "tooltip_render_align")' in handlers_text


def test_gui_fp16_defaults_on_loads_config_and_still_forces_off_for_mps_save():
    config_text = _config_source().read_text(encoding="utf-8")
    config_mgr_text = _file_text("config_mgr.py")

    assert '"FP16": True' in config_text
    assert 'self.fp16_cb.value = bool(cfg.get("FP16", DEFAULTS["FP16"]))' in config_mgr_text
    assert 'self.fp16_cb.value = DEFAULTS["FP16"]' not in config_mgr_text
    process_text = _file_text("process.py")

    assert 'fp16_value = False if "MPS" in (self.device_dd.value or "") else bool(self.fp16_cb.value)' in config_mgr_text
    assert '"FP16": fp16_value' in config_mgr_text
    assert '"FP16": self.fp16_cb.value' not in config_mgr_text
    assert 'self._config["FP16"] = DEFAULTS["FP16"]' not in process_text


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


def test_openxr_headset_preset_dropdown_is_visible_only_for_openxr_and_saved():
    config_text = _config_source().read_text(encoding="utf-8")
    builders_text = _file_text("builders.py")
    handlers_text = _file_text("handlers.py")
    config_mgr_text = _file_text("config_mgr.py")
    localization_text = _localization_source().read_text(encoding="utf-8")

    assert '"XR Headset Model": DEFAULT_XR_HEADSET_MODEL' in config_text
    assert 'from utils.xr_headset_presets import DEFAULT_XR_HEADSET_MODEL' in config_text
    assert 'from utils.xr_headset_presets import xr_headset_options, xr_headset_to_display' in builders_text
    assert 'self.xr_headset_label = ft.Text("Headset Model:"' in builders_text
    assert 'self.xr_headset_dd = CompactDropdown(' in builders_text
    assert 'options=xr_headset_options(self.locale)' in builders_text
    assert 'value=xr_headset_to_display(DEFAULTS.get("XR Headset Model"), self.locale)' in builders_text
    assert 'width=S(130))' in builders_text[builders_text.index('self.xr_headset_dd = CompactDropdown('):builders_text.index('self.display_mode_label = ft.Text', builders_text.index('self.xr_headset_dd = CompactDropdown('))]
    row_start = builders_text.index('self.row7a = ft.Row([')
    row_end = builders_text.index('], spacing=1)', row_start)
    row7a = builders_text[row_start:row_end]
    assert row7a.index('self.run_mode_dd') < row7a.index('self.xr_headset_label') < row7a.index('self.display_mode_label')

    assert 'def on_xr_headset_change' in handlers_text
    assert 'self._config["XR Headset Model"] = display_to_xr_headset(value)' in handlers_text
    assert 'self.xr_headset_label.visible = is_openxr' in handlers_text
    assert 'self.xr_headset_dd.visible = is_openxr' in handlers_text
    assert 'self.xr_headset_dd.options = xr_headset_options(self.locale)' in handlers_text
    assert '(self.xr_headset_dd, "tooltip_xr_headset")' in handlers_text

    assert 'cfg.get("XR Headset Model", DEFAULTS["XR Headset Model"])' in config_mgr_text
    assert '"XR Headset Model": display_to_xr_headset(self.xr_headset_dd.value)' in config_mgr_text
    assert '"Headset Model:": "Headset Model:"' in localization_text
    assert '"Headset Model:": "头显型号:"' in localization_text
    assert '"tooltip_xr_headset"' in localization_text
