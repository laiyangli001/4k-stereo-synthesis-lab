"""
Desktop2Stereo Flet GUI
Flet desktop app, migrated from tkinter gui.py.
Feature-equivalent; all interaction logic matches the original.

Custom control overview:

CompactTextField (L535)
  Compact text input, 32px height. Displays as bordered text label by default;
  switches to TextField on click for editing.
  Constructor params:
    value      — initial value
    width      — control width
    read_only  — read-only flag
    on_change  — value change callback (param: e.control.value)
    tooltip    — tooltip text
    filter     — regex to restrict allowed characters (e.g. r"[0-9]" for digits only)
    max_length — max input length
  On submit (Enter/Blur), non-matching characters are stripped.
  Properties: .value (r/w), .set_tooltip(text)

CompactDropdown (L615)
  Custom dropdown based on PopupMenuButton, 32px height, visually consistent with TextField.
  Constructor params:
    options    — list of options
    value      — currently selected value
    on_select  — selection callback (param: e.control.value)
    expand     — expand_loose flag
    dyna_width — dynamic width mode (uses LABEL_ALIGN_WIDTH)
    width      — fixed width
    min_width  — auto-width lower bound
    max_width  — auto-width upper bound
    tooltip    — tooltip text
  Width priority: fixed > dynamic mode > auto-calculated (bounded by min/max).
  Properties: .value (r/w), .options (r/w, triggers menu rebuild), .set_tooltip(text)
"""
import os
import sys
import subprocess
import time
import asyncio
import ctypes
import flet as ft
from utils import (
    VERSION, OS_NAME, ALL_MODELS, DEFAULT_PORT, STEREO_MIX_NAMES,
    DISABLE_TRT_KEYWORDS, DISABLE_COREML_KEYWORDS, DISABLE_OPENVINO_KEYWORDS,
    get_local_ip, shutdown_event, read_yaml
)
from . import devices as devices_module
from .capture_sources import (
    PRIMARY_MONITOR_SUFFIX,
    get_capture_tool_options,
    get_default_windows_capture_tool,
    get_monitor_index_for_point,
    get_primary_monitor_index,
    list_windows,
)
from .config import (
    DEFAULTS,
    DEFAULT_FAMILIES,
    DEFAULT_MODEL_LIST,
    FAMILY_SIZE_TO_MODEL,
    FAMILY_TO_SIZES,
    get_environment_model_options,
    parse_model_name,
    save_yaml,
)
from .controls import (
    FONT_SIZE,
    SCALE,
    CompactDropdown,
    CompactTextField,
    S,
    set_label_align_width,
)
from .paths import BASE_DIR, DIAG_LOG
from .localization import (
    UI_MESSAGES,
    display_to_stereo_quality,
    is_supported_locale,
    stereo_quality_options,
    stereo_quality_to_display,
)
_NOISY_CONSOLE_PREFIXES = (
    "[NativeUtil] sogou_native_util_pc loaded successfully",
    "[warmup] same version",
)


def _is_noisy_console_output(data):
    text = str(data or "").strip()
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in _NOISY_CONSOLE_PREFIXES)


def _setup_console_logging():
    """Redirect stdout/stderr to also write to the diag log file."""
    import datetime
    os.makedirs(os.path.dirname(DIAG_LOG), exist_ok=True)

    class _TeeStream:
        def __init__(self, original, label):
            self.original = original
            self.label = label
            self._buffer = ""

        def write(self, data):
            if _is_noisy_console_output(data):
                return len(data or "")
            self.original.write(data)
            if data and data.strip():
                try:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    with open(DIAG_LOG, "a", encoding="utf-8") as f:
                        for line in data.splitlines():
                            stripped = line.strip()
                            if stripped:
                                f.write(f"[{ts}] [{self.label}] {stripped}\n")
                except Exception:
                    pass

        def flush(self):
            self.original.flush()

        def isatty(self):
            return self.original.isatty()

    sys.stdout = _TeeStream(sys.stdout, "out")
    sys.stderr = _TeeStream(sys.stderr, "err")

def _set_console_quick_edit(enabled: bool):
    """Toggle Windows console Quick Edit mode when a real console is attached."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            mode.value |= ENABLE_EXTENDED_FLAGS
            if enabled:
                mode.value |= ENABLE_QUICK_EDIT_MODE
            else:
                mode.value &= ~ENABLE_QUICK_EDIT_MODE
            kernel32.SetConsoleMode(h_stdin, mode)
    except Exception:
        pass  # Not a console or Windows — safe to ignore


# Disable Quick Edit while the worker is running to avoid accidental console freeze.
_set_console_quick_edit(False)

# ─────────────────────────────────────────────
# UI Text Dictionary (EN / CN)
# ─────────────────────────────────────────────
DEVICES = devices_module.DEVICES

def _get_capture_tool_options(device_label):
    return get_capture_tool_options(device_label)

# Main GUI Class
class Desktop2StereoGUI:
    """Flet GUI for Desktop2Stereo — full equivalent of tk ConfigGUI."""
    def __init__(self, page: ft.Page):
        self.page = page
        self._loop = None  # Assigned in setup()
        # ── state ──
        self.locale = "EN"
        self._config = {}
        self.run_mode_key = DEFAULTS.get("Run Mode", "Local Viewer")
        self.capture_mode_key = DEFAULTS.get("Capture Mode", "Monitor")
        self.stream_protocol_key = DEFAULTS.get("Stream Protocol", "RTMP")
        self.selected_window_name = ""
        self.selected_window_handle = None
        self.selected_window_rect = None
        self._window_objects = []
        self.process = None
        self._starting = False
        self._proc_lock = None  # Assigned in setup()
        self.monitor_label_to_index = {}
        self.device_label_to_index = {}
        self._esc_down = None        # GetAsyncKeyState: ESC press timestamp
        self._esc_stopped = False    # Prevent duplicate stop triggers
        self._closed = False         # Window closed flag
        self._cancel_starting = False  # Cancel pending start
        self._stopping = False       # Prevent _async_stop reentry
        self._labels_aligned = False  # One-shot label alignment guard
        self._status_key = ""          # Track status for locale change re-translation

    # ── setup ──
    async def setup(self):
        _setup_console_logging()
        # Get the correct event loop in async context
        self._loop = asyncio.get_running_loop()
        self._proc_lock = asyncio.Lock()
        self._hot_save_task = None

        self.page.title = f"Desktop2Stereo v{VERSION}"
        self.page.window.icon = os.path.join(BASE_DIR, "icon.ico")
        self.page.padding = S(24)
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.window.min_width = S(520)
        self.page.window.min_height = S(300)

        # Build UI
        self.build_ui()
        self._auto_align_labels()
        self.page.on_close = self._on_page_close

        # Populate monitors & devices
        self.monitor_label_to_index = self.populate_monitors()
        self.device_label_to_index = self.populate_devices()

        # Load config
        self._config = DEFAULTS.copy()
        if os.path.exists(os.path.join(BASE_DIR, "settings.yaml")):
            try:
                cfg = read_yaml(os.path.join(BASE_DIR, "settings.yaml"))
                if cfg:
                    self._config.update(cfg)
                    self._yaml_loaded = True
                    self.locale = self._config.get("Language", "EN")
                    self.apply_config(self._config)
                    self.set_status(UI_MESSAGES[self.locale]["Loaded settings.yaml at startup"], key="Loaded settings.yaml at startup")
            except Exception as e:
                self.apply_config(self._config)
                self.set_status(f"{UI_MESSAGES[self.locale]['Failed to load settings.yaml:']} {e}")
        else:
            self.apply_config(self._config)

        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.on_keyboard_event = self._on_key
        self._esc_task = asyncio.ensure_future(self._esc_poll_task())
        # Calculate window size without triggering update
        self._fit_window_to_content(update=False)

        # Show window once
        self.page.window.visible = True
        self.page.update()
        await asyncio.sleep(0)

    def _ctrl_width(self, ctrl):
        """Get actual width of a control, accounting for CompactDropdown min/max constraints."""
        # CompactDropdown: estimate from current text, not stale width property
        if hasattr(ctrl, '_calc_auto_width'):
            auto = ctrl._calc_auto_width()
            fixed = getattr(ctrl, '_fixed', None)
            mn = getattr(ctrl, '_min', 0) or 0
            mx = getattr(ctrl, '_max', 0) or 0
            if fixed is not None:
                return fixed
            if mn and auto < mn:
                return mn
            if mx and auto > mx:
                return mx
            return auto
        # Standard control width attribute
        w = getattr(ctrl, "width", None) or 0
        if w:
            return w
        if hasattr(ctrl, '_fixed') and ctrl._fixed:
            return ctrl._fixed
        if hasattr(ctrl, '_label'):
            txt = ctrl._label.value or ""
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        # CompactTextField
        if hasattr(ctrl, '_value'):
            txt = str(ctrl._value or "")
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        # ft.Button / ft.FilledButton: extract from content.controls[0] or content.value
        content = getattr(ctrl, "content", None)
        if content is not None:
            if hasattr(content, "value") and content.value:
                txt = content.value
            elif hasattr(content, "controls"):
                txt = "".join(c.value for c in content.controls if hasattr(c, "value") and c.value)
            else:
                txt = ""
            if txt:
                return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 40
        # ft.Checkbox / generic: use label or value
        txt = getattr(ctrl, "label", None) or getattr(ctrl, "value", None) or ""
        return sum(13 if ord(ch) > 127 else 7 for ch in str(txt)) + 28

    def _fit_window_to_content(self, update=True):
        """Auto-adjust the window around the visible content."""
        width = self._estimate_window_width()
        self.page.window.min_width = min(width, S(520))
        self.page.window.width = width
        self.page.window.max_width = width
        self.page.window.height = self._estimate_window_height()
        if update:
            self.page.update()

    def _spacing_width(self, controls, spacing):
        visible_count = sum(1 for ctrl in controls if self._control_has_effective_content(ctrl))
        return max(0, visible_count - 1) * (spacing or 0)

    def _estimate_control_width(self, ctrl):
        if ctrl is None or getattr(ctrl, "visible", True) is False:
            return 0
        if getattr(ctrl, "expand", None) and getattr(ctrl, "width", None) is None:
            content = getattr(ctrl, "content", None)
            if content is None:
                return 0
        if isinstance(ctrl, ft.Container):
            content = getattr(ctrl, "content", None)
            if content is not None:
                child_width = self._estimate_control_width(content)
                explicit = getattr(ctrl, "width", None) or 0
                padding = getattr(ctrl, "padding", None)
                pad_x = 0
                if padding is not None:
                    pad_x = (getattr(padding, "left", 0) or 0) + (getattr(padding, "right", 0) or 0)
                return max(explicit, child_width + pad_x)
        if isinstance(ctrl, ft.Row):
            controls = getattr(ctrl, "controls", []) or []
            return sum(self._estimate_control_width(c) for c in controls) + self._spacing_width(controls, getattr(ctrl, "spacing", 0))
        if isinstance(ctrl, ft.Column):
            controls = getattr(ctrl, "controls", []) or []
            return max((self._estimate_control_width(c) for c in controls), default=0)
        return self._ctrl_width(ctrl)

    def _estimate_group_width(self, container):
        if container is None or getattr(container, "visible", True) is False:
            return 0
        content = getattr(container, "content", None)
        content_width = self._estimate_control_width(content)
        padding = getattr(container, "padding", None)
        pad_x = 0
        if padding is not None:
            pad_x = (getattr(padding, "left", 0) or 0) + (getattr(padding, "right", 0) or 0)
        border_x = 2
        return content_width + pad_x + border_x

    def _estimate_window_width(self):
        if not getattr(self, "depth_group", None):
            return S(696)
        sections = [self.lang_group, self.depth_group, self.device_group]
        widths = [self._estimate_group_width(section) for section in sections]
        if getattr(self, "stream_container", None) and self.stream_container.visible:
            widths.append(self._estimate_group_width(self.stream_container))
        content_width = max(widths + [0])
        page_padding = (getattr(self.page, "padding", 0) or 0) * 2
        window_chrome = S(0)
        safety_margin = S(12)
        min_width = S(520)
        max_width = S(1100)
        return max(min_width, min(max_width, content_width + page_padding + window_chrome + safety_margin))

    def _control_has_effective_content(self, ctrl):
        if ctrl is None:
            return False
        if getattr(ctrl, "visible", True) is False:
            return False
        content = getattr(ctrl, "content", None)
        if content is not None:
            return self._control_has_effective_content(content)
        controls = getattr(ctrl, "controls", None)
        if controls is not None:
            return any(self._control_has_effective_content(child) for child in controls)
        return True

    def _estimate_group_height(self, container, include_margin=True):
        if container is None or getattr(container, "visible", True) is False:
            return 0
        content = getattr(container, "content", None)
        controls = getattr(content, "controls", None)
        if not controls:
            return 0
        visible_rows = sum(1 for ctrl in controls if self._control_has_effective_content(ctrl))
        if visible_rows <= 0:
            return 0
        row_height = S(34)
        row_spacing = getattr(content, "spacing", S(8)) or 0
        padding_v = S(24)
        border_v = 2
        margin_v = S(8) if include_margin else 0
        return padding_v + visible_rows * row_height + max(0, visible_rows - 1) * row_spacing + border_v + margin_v

    def _estimate_window_height(self):
        if not getattr(self, "depth_group", None):
            return S(768)
        scroll_spacing = getattr(getattr(self, "_scroll_area", None), "spacing", S(8)) or 0
        visible_sections = []
        for section in [self.lang_group, self.depth_group, self.device_group]:
            if section is not None and getattr(section, "visible", True):
                visible_sections.append(section)
        scroll_height = sum(self._estimate_group_height(section) for section in visible_sections)
        if getattr(self, "stream_container", None) and self.stream_container.visible:
            scroll_height += scroll_spacing
            scroll_height += self._estimate_group_height(self.stream_container, include_margin=False)
        page_padding = (getattr(self.page, "padding", 0) or 0) * 2
        footer_height = S(58)
        window_chrome = S(42)
        safety_margin = S(0)
        min_height = S(560)
        max_height = S(1040)
        estimated = scroll_height + footer_height + page_padding + window_chrome + safety_margin
        return max(min_height, min(max_height, estimated))

    # Build UI — grid-aligned, matching original GUI

    def _auto_align_labels(self, force=False):
        """Auto-detect longest label text and unify label widths for grid alignment."""
        if self._labels_aligned and not force:
            return
        left_labels = [
            self.depth_model_label, self.depth_resolution_label, self.depth_quick_label, self.convergence_label, self.depth_strength_label, self.foreground_scale_label, self.antialiasing_label,
            self.stereo_preset_label, self.max_shift_label, self.scene_reset_label,
            self.edge_dilation_label, self.stereo_scale_label,
            self.acceleration_label, self.computing_device_label, self.capture_tool_label, self.target_fps_label,
            self.upscaler_label, self.run_mode_label, self.stereo_output_label, self.controller_label,
            self.lang_label,
            self.stream_url_label, self.stream_port_label,
            self.stream_proto_label, self.audio_label, self.crf_label,
        ]
        right_labels = [
            self.ipd_label,
            self.stereo_quality_label, self.temporal_strength_label, self.reset_cooldown_label,
            self.edge_threshold_label, self.anaglyph_label,
            self.upscaler_sharpness_label, self.display_mode_label, self.environment_label, self.theme_label,
            self.stream_quality_label, self.stream_key_label, self.audio_delay_label,
        ]

        def _est(t):
            return sum(S(12) if ord(c) > 127 else S(7) for c in t)

        all_labels = left_labels + right_labels
        max_w = max(_est(lbl.value) for lbl in all_labels)
        final_w = int(max_w * 1.15) + S(10)

        for lbl in all_labels:
            lbl.width = final_w

        self._label_max_width = final_w
        set_label_align_width(final_w)
        for inst in getattr(self, '_dropdowns', []):
            inst.reapply_width()
        if hasattr(self, '_row8_spacer'):
            capture_mode_w = self._ctrl_width(self.capture_mode_dd)
            self._row8_spacer.width = max(0, final_w - capture_mode_w - 1)
            try:
                self._row8_spacer.update()
            except RuntimeError:
                pass
        if hasattr(self, '_accel_spacer'):
            self._accel_spacer.width = final_w
            try:
                self._accel_spacer.update()
            except RuntimeError:
                pass
        self._labels_aligned = True

    def build_ui(self):
        page = self.page
        page.controls.clear()
        self._dropdowns = []
        CompactDropdown._instances = self._dropdowns

        # Row 1: Depth model
        self.depth_model_label = ft.Text("Depth Model:", size=FONT_SIZE, width=S(130))
        default_family, default_size = parse_model_name(DEFAULT_MODEL_LIST[0]) if DEFAULT_MODEL_LIST else ("", "")
        self.depth_model_dd = CompactDropdown(
            options=[f for f in DEFAULT_FAMILIES],
            value=default_family,
            on_select=self.on_model_family_change,
            min_width=S(200), max_width=S(300))
        self.model_size_dd = CompactDropdown(
            options=FAMILY_TO_SIZES.get(default_family, []),
            value=default_size,
            on_select=self.on_model_size_change,
            width=S(110))
        self.fp16_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="FP16")
        row0 = ft.Row([
            self.depth_model_label,
            self.depth_model_dd,
            ft.Container(width=S(8)),
            self.model_size_dd,
        ], spacing=1)

        # Row 2: Depth resolution + Convergence
        self.depth_resolution_label = ft.Text("Depth Resolution:", size=FONT_SIZE, width=S(130))
        self.depth_res_dd = CompactDropdown(options=[], width=S(130))
        self.convergence_label = ft.Text("Convergence:", size=FONT_SIZE, width=S(130))
        conv_options = [str(i / 4) for i in range(-2, 5)]
        self.convergence_dd = CompactDropdown(width=S(130),
            options=[v for v in conv_options],
            value="0.0",
            on_select=self.on_stereo_hot_param_change)
        self.depth_quick_label = ft.Text("Depth Quick:", size=FONT_SIZE, width=S(130))
        self.depth_quick_dd = CompactDropdown(
            options=["Soft", "Standard", "Enhanced"],
            value="Standard",
            on_select=self.on_depth_quick_change,
            width=S(130))
        row1 = ft.Row([
            self.depth_resolution_label,
            self.depth_res_dd,
            ft.Container(width=S(40)),
            self.depth_quick_label,
            self.depth_quick_dd,
        ], spacing=1)

        # Row 3: Quick depth preset + advanced depth strength

        self.depth_strength_label = ft.Text("Depth Strength:", size=FONT_SIZE, width=S(130))
        ds_options = [f"{i / 2:.1f}" for i in range(21)]
        self.depth_strength_dd = CompactDropdown(width=S(130),
            options=[v for v in ds_options],
            value="2.0",
            on_select=self.on_stereo_hot_param_change)
        convergence_depth_row = ft.Row([
            self.convergence_label,
            self.convergence_dd,
            ft.Container(width=S(40)),
            self.depth_strength_label,
            self.depth_strength_dd,
        ], spacing=1)

        # Row 3b: Foreground scale + anti-aliasing
        self.foreground_scale_label = ft.Text("Foreground Scale:", size=FONT_SIZE, width=S(130))
        fg_options = [f"{i / 10:.1f}" for i in range(-9, 0)] + [f"{i / 2:.1f}" for i in range(0, 11)]
        self.foreground_scale_dd = CompactDropdown(width=S(130),
            options=[v for v in fg_options],
            value="0.5",
            on_select=self.on_stereo_hot_param_change)
        self.antialiasing_label = ft.Text("Anti-aliasing:", size=FONT_SIZE, width=S(130))
        aa_options = [str(i) for i in range(11)]
        self.antialiasing_dd = CompactDropdown(width=S(130), 
            options=[v for v in aa_options],
            value="2",
            on_select=self.on_stereo_hot_param_change)
        row2b = ft.Row([
            self.foreground_scale_label,
            self.foreground_scale_dd,
            ft.Container(width=S(40)),
            self.antialiasing_label,
            self.antialiasing_dd,
        ], spacing=1)

        # Row 4: IPD + stereo scale
        self.ipd_label = ft.Text("IPD (mm):", size=FONT_SIZE, width=S(130))
        self.ipd_dd = CompactDropdown(options=[str(i) for i in range(58, 71)], value="64", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.stereo_scale_label = ft.Text("Stereo Scale:", size=FONT_SIZE, width=S(130))
        self.stereo_scale_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(1, 11)], value="0.5", width=S(130), on_select=self.on_stereo_hot_param_change)
        row3 = ft.Row([
            self.ipd_label,
            self.ipd_dd,
            ft.Container(width=S(40)),
            self.stereo_scale_label,
            self.stereo_scale_dd,
        ], spacing=1)

        # Row 5: Stereo runtime mode and quality
        self.stereo_preset_label = ft.Text("Stereo Mode:", size=FONT_SIZE, width=S(130))
        self.stereo_preset_dd = CompactDropdown(
            options=["Cinema", "Game / Low Latency", "Still Image / HQ", "Debug / Export"],
            value="Cinema",
            width=S(130),
            on_select=self.on_stereo_preset_change)
        self.stereo_quality_label = ft.Text("Synthetic View:", size=FONT_SIZE, width=S(130))
        self.stereo_quality_dd = CompactDropdown(options=self._stereo_quality_options(), value=self._stereo_quality_to_display("quality_4k"), width=S(130))
        stereo_row0 = ft.Row([self.stereo_preset_label, self.stereo_preset_dd, ft.Container(width=S(40)), self.stereo_quality_label, self.stereo_quality_dd], spacing=1)

        self.max_shift_label = ft.Text("Max Shift Ratio:", size=FONT_SIZE, width=S(130))
        self.max_shift_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)], value="0.05", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.temporal_strength_label = ft.Text("Temporal Strength:", size=FONT_SIZE, width=S(130))
        self.temporal_strength_dd = CompactDropdown(options=[f"{i / 10:.1f}" for i in range(0, 11)], value="0.7", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row1 = ft.Row([self.max_shift_label, self.max_shift_dd, ft.Container(width=S(40)), self.temporal_strength_label, self.temporal_strength_dd], spacing=1)

        self.scene_reset_label = ft.Text("Scene Threshold:", size=FONT_SIZE, width=S(130))
        self.scene_reset_dd = CompactDropdown(options=["0.00", "0.12", "0.18", "0.22", "0.28", "0.35"], value="0.22", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.reset_cooldown_label = ft.Text("Reset Cooldown:", size=FONT_SIZE, width=S(130))
        self.reset_cooldown_dd = CompactDropdown(options=["1", "2", "3", "4", "6"], value="3", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row2 = ft.Row([self.scene_reset_label, self.scene_reset_dd, ft.Container(width=S(40)), self.reset_cooldown_label, self.reset_cooldown_dd], spacing=1)

        self.edge_dilation_label = ft.Text("Edge Dilation:", size=FONT_SIZE, width=S(130))
        self.edge_dilation_dd = CompactDropdown(options=["0", "1", "2", "3", "4"], value="2", width=S(130), on_select=self.on_stereo_hot_param_change)
        self.edge_threshold_label = ft.Text("Edge Threshold:", size=FONT_SIZE, width=S(130))
        self.edge_threshold_dd = CompactDropdown(options=[f"{i / 100:.2f}" for i in range(0, 11)], value="0.04", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row3 = ft.Row([self.edge_dilation_label, self.edge_dilation_dd, ft.Container(width=S(40)), self.edge_threshold_label, self.edge_threshold_dd], spacing=1)

        self.cross_eyed_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Cross Eyed", value=False, on_change=self.on_stereo_hot_param_change)
        self.anaglyph_label = ft.Text("Anaglyph:", size=FONT_SIZE, width=S(130))
        self.anaglyph_dd = CompactDropdown(options=["red_cyan", "green_magenta", "amber_blue"], value="red_cyan", width=S(130), on_select=self.on_stereo_hot_param_change)
        stereo_row4 = ft.Row([self.anaglyph_label, self.anaglyph_dd, ft.Container(width=S(40)), self.cross_eyed_cb, ft.Container(width=S(20)), self.fp16_cb], spacing=1)
        self.advanced_stereo_cb = ft.Checkbox(
            scale=SCALE,
            visual_density=ft.VisualDensity.COMPACT,
            label="Advanced Stereo",
            value=False,
            on_change=self.on_advanced_stereo_change,
        )
        advanced_stereo_row = ft.Row([self.advanced_stereo_cb], spacing=1)
        self._advanced_stereo_rows = [convergence_depth_row, row2b, row3, stereo_row1, stereo_row2, stereo_row3, stereo_row4]
        # Row 10: Acceleration group (two rows, 4 columns each)
        self.acceleration_label = ft.Text("Acceleration:", size=FONT_SIZE, width=S(130))
        self.torch_compile_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="torch.compile")
        self.tensorrt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="TensorRT", on_change=self._on_trt_toggle)
        self.coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="CoreML", on_change=self._on_coreml_toggle)
        self.openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="OpenVINO", on_change=self._on_openvino_toggle)
        self.recompile_trt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile TensorRT")
        self.recompile_coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile CoreML")
        self.recompile_openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile OpenVINO")
        accel_row1 = ft.Row([self.torch_compile_cb, self.tensorrt_cb, self.recompile_trt_cb], spacing=S(20))
        accel_row2 = ft.Row([self.coreml_cb, self.recompile_coreml_cb, self.openvino_cb, self.recompile_openvino_cb], spacing=S(20))
        self._accel_spacer = ft.Container(width=0)
        self.row4a = ft.Row([
            self.acceleration_label,
            accel_row1,
        ], spacing=1)
        self.row4b = ft.Row([
            self._accel_spacer,
            accel_row2,
        ], spacing=1)
        self._advanced_stereo_rows.extend([self.row4a, self.row4b])
        for row in self._advanced_stereo_rows:
            row.visible = self.advanced_stereo_cb.value

        # Row 6: Computing device
        self.computing_device_label = ft.Text("Computing Device:", size=FONT_SIZE, width=S(130))
        device_names = [v["name"] for v in DEVICES.values()]
        self.device_dd = CompactDropdown(
            options=[n for n in device_names],
            on_select=self.on_device_change,
            min_width=S(180))
        self.showfps_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Show FPS")
        self.local_vsync_cb = ft.Checkbox(
            scale=SCALE,
            visual_density=ft.VisualDensity.COMPACT,
            label="Local VSync",
            value=DEFAULTS.get("Local VSync", True),
        )
        self.target_fps_label = ft.Text("Capture FPS:", size=FONT_SIZE, width=S(130))
        self.target_fps_dd = CompactDropdown(
            options=["Auto", "60", "72", "80", "90", "120"],
            value="Auto",
            width=S(74),
        )
        self.advanced_device_cb = ft.Checkbox(
            scale=SCALE,
            visual_density=ft.VisualDensity.COMPACT,
            label="Advanced Options",
            value=False,
            on_change=self.on_advanced_device_change,
        )
        row5 = ft.Row([
            self.computing_device_label,
            self.device_dd,
            ft.Container(width=S(15)),
            self.advanced_device_cb,
        ], spacing=1)

        # Row 7: Capture tool
        self.capture_tool_label = ft.Text("Capture Tool:", size=FONT_SIZE, width=S(130))
        ct_options = _get_capture_tool_options(DEVICES.get(0, {}).get("name", ""))
        self.capture_tool_dd = CompactDropdown(
            options=[o for o in ct_options],
            on_select=self.on_capture_tool_change,
            min_width=S(160))
        row6 = ft.Row([
            self.capture_tool_label,
            self.capture_tool_dd,
            ft.Container(width=S(15)),
            self.showfps_cb,
        ], spacing=1)
        self.row6b = ft.Row([
            self.target_fps_label,
            self.target_fps_dd,
            ft.Container(width=S(20)),
            self.local_vsync_cb,
        ], spacing=1)
        self.upscaler_label = ft.Text("Upscaler:", size=FONT_SIZE, width=S(130))
        self.upscaler_dd = CompactDropdown(
            options=["Off", "FSR1"],
            value="Off",
            width=S(90),
        )
        self.upscaler_sharpness_label = ft.Text("Sharpness:", size=FONT_SIZE, width=S(130))
        self.upscaler_sharpness_dd = CompactDropdown(
            options=["0.00", "0.25", "0.35", "0.50", "0.75", "1.00"],
            value="0.35",
            width=S(74),
        )
        self.row6c = ft.Row([
            self.upscaler_label,
            self.upscaler_dd,
            ft.Container(width=S(20)),
            self.upscaler_sharpness_label,
            self.upscaler_sharpness_dd,
        ], spacing=1)
        if OS_NAME == "Linux":
            self.capture_tool_label.visible = False
            self.capture_tool_dd.visible = False

        # Row 8: Run mode + Display mode / Controller
        self.run_mode_label = ft.Text("Run Mode:", size=FONT_SIZE, width=S(130))
        self.run_mode_dd = CompactDropdown(on_select=self.on_run_mode_change, width=S(130))
        self.display_mode_label = ft.Text("Display Mode:", size=FONT_SIZE, width=S(130))
        self.display_mode_dd = CompactDropdown(
            options=[m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB", "Depth Map", "Anaglyph", "Interleaved", "Mono", "Leia"]],
            value="Half-SBS", width=S(130))
        self.xr_preview_cb = ft.Checkbox(
            label="XR Preview Window",
            value=DEFAULTS.get("XR Preview Window", True),
        )
        self.controller_label = ft.Text("Controller:", size=FONT_SIZE, width=S(130))
        try:
            ctrl_base = os.path.join(BASE_DIR, "controllers")
            ctrl_dirs = [d for d in os.listdir(ctrl_base) if os.path.isdir(os.path.join(ctrl_base, d))]
        except (FileNotFoundError, OSError):
            ctrl_dirs = []
        if not ctrl_dirs:
            ctrl_dirs = ["PICO"]
        self.ctrl_model_dd = CompactDropdown(
            options=[c for c in ctrl_dirs],
            value="PICO", width=S(130))
        self.environment_label = ft.Text("Environment:", size=FONT_SIZE, width=S(130))
        env_dirs = get_environment_model_options()
        self.env_model_dd = CompactDropdown(
            options=[e for e in env_dirs],
            value="None" if "None" in env_dirs else env_dirs[0],
            width=S(130))
        self.row7a = ft.Row([
            self.run_mode_label,
            self.run_mode_dd,
            ft.Container(width=S(40)),
            self.display_mode_label,
            self.display_mode_dd,
            self.xr_preview_cb
        ], spacing=1)
        self.row7b = ft.Row([
            self.controller_label,
            self.ctrl_model_dd,
            ft.Container(width=S(40)),
            self.environment_label,
            self.env_model_dd
        ], spacing=1)

        # Row 9: Input monitor/window + Refresh
        self.capture_mode_dd = CompactDropdown(
            options=["Monitor",
                     "Window"],
            value="Monitor", on_select=self.on_capture_mode_change,
            dyna_width=True)
        self.monitor_dd = CompactDropdown(on_select=self._on_monitor_change, max_width=S(300))
        self.window_dd = CompactDropdown(on_select=self.on_window_selected, max_width=S(300))
        self.refresh_btn = ft.Button(content=ft.Text("Refresh", size=FONT_SIZE), width=S(130), on_click=self.refresh_monitor_and_window)
        self._row8_spacer = ft.Container(width=S(60))
        row8 = ft.Row([
            self.capture_mode_dd,
            self._row8_spacer,
            self.monitor_dd,
            self.window_dd,
            ft.Container(width=S(8)),
            ft.Container(expand=True),
            self.refresh_btn
        ], spacing=1)

        # Row 10: Stereo output + checkboxes
        self.stereo_output_label = ft.Text("Stereo Output:", size=FONT_SIZE, width=S(130))
        self.stereo_monitor_dd = CompactDropdown(options=[], on_select=lambda e: self._fit_window_to_content())
        self.fill_16_9_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fill 16:9")
        self.fix_aspect_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fix Viewer Aspect")
        self.lossless_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="LSFG")
        self._stereo_spacer = ft.Container(width=S(10))
        self.row9 = ft.Row([
            self.stereo_output_label,
            self.stereo_monitor_dd,
            self._stereo_spacer,
            ft.Row([self.fill_16_9_cb, self.fix_aspect_cb, self.lossless_cb], spacing=S(20)),
        ], spacing=1)

        # Bottom: Language + Theme + Buttons
        self.lang_label = ft.Text("Set Language:", size=FONT_SIZE, width=S(130))
        self.lang_dd = CompactDropdown(
            options=["English", "简体中文"],
            value="English", on_select=self.on_language_change,
            width=S(130))
        self.theme_label = ft.Text("Theme:", size=FONT_SIZE, width=S(130))
        self.theme_dd = CompactDropdown(
            options=["system", "blue", "green", "red", "purple", "orange", "teal", "pink", "grey"],
            value="system", on_select=self.on_theme_change,
            width=S(130))
        self.reset_btn = ft.Button(content=ft.Text("Reset", size=FONT_SIZE), width=S(130), on_click=self.reset_defaults)
        self.stop_btn = ft.Button(content=ft.Text("Stop", size=FONT_SIZE), width=S(130), on_click=self.stop_process)
        self.run_btn = ft.Button(content=ft.Text("Run", size=FONT_SIZE), width=S(150), on_click=self.save_and_run)
        lang_row = ft.Row([
            self.lang_label,
            self.lang_dd,
            ft.Container(width=S(40)),
            self.theme_label,
            self.theme_dd,
        ], spacing=1)

        self.status_text = ft.Text("", italic=True, size=FONT_SIZE)

        # ========== Assembly ==========
        depth_group = ft.Container(
            ft.Column([row0, row1, stereo_row0, convergence_depth_row, row2b, row3, stereo_row1, stereo_row2, stereo_row3, stereo_row4, self.row4a, self.row4b, advanced_stereo_row], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        device_group = ft.Container(
            ft.Column([row5, row6, self.row6b, self.row7a, self.row7b, row8, self.row6c, self.row9], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        lang_group = ft.Container(
            ft.Column([lang_row], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        self.lang_group = lang_group
        self.depth_group = depth_group
        self.device_group = device_group
        self._build_streamer_rows()

        scroll_area = ft.Column([
            self.lang_group,
            self.depth_group,
            self.device_group,
            self.stream_container,
        ], scroll=ft.ScrollMode.AUTO, expand=False, tight=True, spacing=S(8))

        btn_row = ft.Row([
            self.reset_btn,
            ft.Container(expand=True),
            ft.Container(content=ft.Row([self.stop_btn, self.run_btn], spacing=S(20)), padding=ft.Padding(0, 0, S(10), 0)),
        ])

        self._btn_bar = ft.Container(content=btn_row)
        self._status_bar = ft.Row([
            ft.Container(
                content=self.status_text,
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border_radius=0, padding=ft.Padding(S(8), S(4), S(8), S(4)),
                expand=True,
            )
        ])

        footer = ft.Container(
            ft.Column([self._btn_bar, self._status_bar], spacing=S(6)),
            padding=ft.Padding(0, S(6), 0, 0),
        )

        self._scroll_area = scroll_area
        self._footer = footer

        root = ft.Column([scroll_area, footer], expand=False, tight=True, spacing=0)
        page.add(root)

    # All event/logic methods below preserve original behavior
    def _build_streamer_rows(self):
        self.stream_url_label = ft.Text("Stream URL:", size=FONT_SIZE, width=S(150))
        self.stream_url_tf = ft.Container(
            content=ft.Row([ft.Text("", size=FONT_SIZE)], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0), expand=True,
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4, on_click=self.copy_url_to_clipboard,
        )
        self.preview_btn = ft.Button(content=ft.Text("Preview", size=FONT_SIZE), width=S(130), on_click=self.preview_in_browser)
        self.stream_url_row = ft.Row(
            [self.stream_url_label, self.stream_url_tf, ft.Container(width=S(10)), self.preview_btn],
            spacing=2,
        )
        self.stream_port_label = ft.Text("Streamer Port:", size=FONT_SIZE, width=S(150))
        self.stream_port_tf = CompactTextField(value=str(DEFAULT_PORT), width=S(130), on_change=self.update_stream_url, filter=r"[0-9]", max_length=5)
        self.stream_quality_label = ft.Text("Stream Quality:", size=FONT_SIZE)
        qual_vals = [str(i) for i in range(100, 49, -5)]
        self.stream_quality_dd = CompactDropdown(width=S(130),
            options=[q for q in qual_vals], value="100")
        self.stream_port_quality_row = ft.Row(
            [self.stream_port_label, self.stream_port_tf, ft.Container(width=S(40)), self.stream_quality_label, self.stream_quality_dd],
            spacing=1,
        )
        self.stream_proto_label = ft.Text("Stream Protocol:", size=FONT_SIZE, width=S(150))
        self.stream_proto_dd = CompactDropdown(width=S(130),
            options=[p for p in ["RTMP", "RTSP", "HLS", "HLS M3U8", "WebRTC"]],
            value="HLS", on_select=self._on_stream_protocol_change)
        self.stream_key_label = ft.Text("Stream Key:", size=FONT_SIZE, width=S(130))
        self.stream_key_tf = CompactTextField(value="live", width=S(130), on_change=self._on_stream_key_change)
        self.stream_proto_row = ft.Row([self.stream_proto_label, self.stream_proto_dd, ft.Container(width=S(40)), self.stream_key_label, self.stream_key_tf], spacing=1)
        self.audio_label = ft.Text("Stereo Mix:", size=FONT_SIZE, width=S(150))
        self.audio_dd = CompactDropdown(options=[], min_width=S(130))
        self.audio_row = ft.Row([self.audio_label, self.audio_dd], spacing=1)
        self.crf_label = ft.Text("CRF:", size=FONT_SIZE, width=S(150))
        self.crf_tf = CompactTextField(value="20", width=S(130), filter=r"[0-9]", max_length=2)
        self.audio_delay_label = ft.Text("Audio Delay (s):", size=FONT_SIZE, width=S(130))
        self.audio_delay_tf = CompactTextField(value="-0.15", width=S(130), filter=r"[0-9\-\.]", max_length=6)
        self.crf_row = ft.Row([self.crf_label, self.crf_tf, ft.Container(width=S(40)), self.audio_delay_label, self.audio_delay_tf], spacing=1)
        self._streamer_rows = [
            self.stream_url_row,
            self.stream_port_quality_row,
            self.stream_proto_row,
            self.crf_row,
            self.audio_row,
        ]
        self.stream_container = ft.Container(
            ft.Column([], spacing=S(8)),
            visible=False, padding=ft.Padding(S(16), S(10), S(16), S(10)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6,
        )

    def _show_streamer_rows(self, *row_indices):
        col = self.stream_container.content.controls
        col.clear()
        for i in row_indices:
            if 0 <= i < len(self._streamer_rows):
                col.append(self._streamer_rows[i])
        self.stream_container.visible = bool(row_indices)
        self.stream_container.update()
        self._fit_window_to_content()

    def _get_streamer_row_map(self):
        return {
            "Local Viewer": [],
            "3D Monitor": [],
            "OpenXR Link": [],
            "MJPEG Streamer": [0, 1],
            "Legacy Streamer": [0, 1],
            "RTMP Streamer": [0, 1, 2, 3, 4],
        }

    def populate_monitors(self):
        self.monitor_label_to_index = {}
        monitors = []
        try:
            import mss
            with mss.mss() as sct:
                monitors = sct.monitors[1:]
        except Exception:
            monitors = []
        if not monitors:
            self.monitor_dd.options = []
            self.monitor_dd.update()
            return {}
        primary_index = get_primary_monitor_index()
        if primary_index < 1 or primary_index > len(monitors):
            primary_index = 1
        current_val = self.monitor_dd.value if hasattr(self, 'monitor_dd') else ""
        found = False
        opts = []
        for idx, mon in enumerate(monitors, start=1):
            is_primary = idx == primary_index
            suffix = PRIMARY_MONITOR_SUFFIX if is_primary else ""
            label = f"{idx}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']}){suffix}"
            self.monitor_label_to_index[label] = idx
            opts.append(label)
            if label == current_val:
                found = True
        self.monitor_dd.options = opts
        if found:
            self.monitor_dd.value = current_val
        else:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == primary_index), None)
            self.monitor_dd.value = primary_label or (list(self.monitor_label_to_index.keys())[0] if self.monitor_label_to_index else "")
        self.monitor_dd.update()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        return self.monitor_label_to_index

    def populate_devices(self):
        self.device_label_to_index = {}
        device_dict = DEVICES
        opts = []
        for idx, dev_info in device_dict.items():
            label = dev_info["name"]
            self.device_label_to_index[label] = idx
            opts.append(label)
        self.device_dd.options = opts
        default_idx = DEFAULTS.get("Computing Device", 0)
        default_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == default_idx), None)
        self.device_dd.value = default_label or (opts[0] if opts else "")
        self.device_dd.update()
        return self.device_label_to_index

    def _upscaler_display_options(self):
        t = UI_MESSAGES[self.locale]
        return [t.get("Auto", "Auto"), t.get("Off", "Off"), "FSR1"]

    def _upscaler_to_display(self, value):
        value_l = str(value or "").strip().lower()
        if value_l == "fsr1":
            return "FSR1"
        if value_l in ("auto", "自动"):
            return UI_MESSAGES[self.locale].get("Auto", "Auto")
        return UI_MESSAGES[self.locale].get("Off", "Off")

    def _upscaler_from_display(self, value):
        value_l = str(value or "").strip().lower()
        if value_l == "fsr1":
            return "FSR1"
        if value_l in ("auto", "自动"):
            return "Auto"
        return "Off"

    def _target_fps_to_display(self, value):
        fps = self._parse_int(value, DEFAULTS["Target FPS"])
        if fps <= 0:
            return UI_MESSAGES[self.locale].get("Auto", "Auto")
        return str(fps)

    @staticmethod
    def _target_fps_from_display(value):
        value_l = str(value or "").strip().lower()
        if value_l in ("auto", "自动"):
            return 0
        try:
            return int(value_l)
        except (TypeError, ValueError):
            return 0
    def apply_config(self, cfg, keep_optional=True):
        self._config = cfg.copy()
        current_primary = get_primary_monitor_index()
        mon_idx = cfg.get("Monitor Index", DEFAULTS["Monitor Index"])
        label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == mon_idx), None)
        if label:
            self.monitor_dd.value = label
        elif self.monitor_label_to_index:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == current_primary), None)
            self.monitor_dd.value = primary_label or next(iter(self.monitor_label_to_index))
        self.selected_window_name = cfg.get("Window Title", "")
        self.selected_window_handle = None
        self.selected_window_rect = None
        if keep_optional and self.capture_mode_key == "Window":
            self.refresh_window_list()
            dev_idx = cfg.get("Computing Device", DEFAULTS["Computing Device"])
            dev_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == dev_idx), None)
            if dev_label:
                self.device_dd.value = dev_label
        model_list = DEFAULT_MODEL_LIST
        selected_model = cfg.get("Depth Model", DEFAULTS["Depth Model"])
        if selected_model not in model_list:
            selected_model = model_list[0] if model_list else DEFAULTS["Depth Model"]
        family, size = parse_model_name(selected_model)
        if family not in DEFAULT_FAMILIES:
            family = DEFAULT_FAMILIES[0] if DEFAULT_FAMILIES else ""
        self.depth_model_dd.options = [f for f in DEFAULT_FAMILIES]
        self.depth_model_dd.value = family
        avail_sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in avail_sizes]
        if size in avail_sizes:
            self.model_size_dd.value = size
        elif avail_sizes:
            self.model_size_dd.value = avail_sizes[0]
        else:
            self.model_size_dd.value = ""
        self.depth_res_dd.value = str(cfg.get("Depth Resolution", DEFAULTS["Depth Resolution"]))
        self.update_depth_resolution_options(self.current_model_name)
        self.depth_strength_dd.value = str(cfg.get("Depth Strength", DEFAULTS["Depth Strength"]))
        self.depth_quick_dd.value = self._depth_quick_to_display(cfg.get("Depth Quick", self._depth_quick_from_strength(cfg.get("Depth Strength", DEFAULTS["Depth Strength"]))))
        self.display_mode_dd.value = cfg.get("Display Mode", DEFAULTS["Display Mode"])
        self.xr_preview_cb.value = cfg.get("XR Preview Window", DEFAULTS["XR Preview Window"])
        self.local_vsync_cb.value = cfg.get("Local VSync", DEFAULTS["Local VSync"])
        self.advanced_device_cb.value = False
        self.upscaler_dd.options = self._upscaler_display_options()
        self.upscaler_dd.value = self._upscaler_to_display(cfg.get("Upscaler", DEFAULTS["Upscaler"]))
        self.upscaler_sharpness_dd.value = f'{self._parse_float(cfg.get("Upscaler Sharpness", DEFAULTS["Upscaler Sharpness"]), DEFAULTS["Upscaler Sharpness"]):.2f}'
        target_fps = self._parse_int(cfg.get("Target FPS", DEFAULTS["Target FPS"]), DEFAULTS["Target FPS"])
        self.target_fps_dd.value = self._target_fps_to_display(target_fps)
        self.antialiasing_dd.value = str(cfg.get("Anti-aliasing", DEFAULTS["Anti-aliasing"]))
        self.foreground_scale_dd.value = str(self._clamp_foreground_scale(cfg.get("Foreground Scale", DEFAULTS["Foreground Scale"])))
        self.convergence_dd.value = str(cfg.get("Convergence", DEFAULTS["Convergence"]))
        self.stereo_preset_dd.value = self._preset_to_display(cfg.get("Stereo Preset", DEFAULTS["Stereo Preset"]))
        self.stereo_quality_dd.value = self._stereo_quality_to_display(cfg.get("Stereo Quality", cfg.get("Synthetic View", DEFAULTS["Stereo Quality"])))
        self.max_shift_dd.value = f'{self._parse_float(cfg.get("Max Shift Ratio", DEFAULTS["Max Shift Ratio"]), DEFAULTS["Max Shift Ratio"]):.2f}'
        self.temporal_strength_dd.value = f'{self._parse_float(cfg.get("Temporal Strength", DEFAULTS["Temporal Strength"]), DEFAULTS["Temporal Strength"]):.2f}'
        self.edge_dilation_dd.value = str(cfg.get("Edge Dilation", DEFAULTS["Edge Dilation"]))
        self.edge_threshold_dd.value = f'{self._parse_float(cfg.get("Edge Threshold", DEFAULTS["Edge Threshold"]), DEFAULTS["Edge Threshold"]):.2f}'
        self.cross_eyed_cb.value = cfg.get("Cross Eyed", DEFAULTS["Cross Eyed"])
        self.anaglyph_dd.value = cfg.get("Anaglyph Method", DEFAULTS["Anaglyph Method"])
        self.advanced_stereo_cb.value = False
        self._sync_advanced_stereo_visibility()
        self._sync_device_advanced_visibility(cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Local Viewer")))
        ipd_m = cfg.get("IPD", DEFAULTS["IPD"])
        self.ipd_dd.value = str(int(ipd_m * 1000))
        self.stereo_scale_dd.value = f'{self._parse_float(cfg.get("Stereo Scale", cfg.get("Stereo Strength Scale", DEFAULTS["Stereo Scale"])), DEFAULTS["Stereo Scale"]):.1f}'
        self.fp16_cb.value = DEFAULTS["FP16"]
        self.showfps_cb.value = cfg.get("Show FPS", DEFAULTS["Show FPS"])
        self.fill_16_9_cb.value = cfg.get("Fill 16:9", DEFAULTS["Fill 16:9"])
        self.fix_aspect_cb.value = cfg.get("Fix Viewer Aspect", DEFAULTS["Fix Viewer Aspect"])
        self.lossless_cb.value = cfg.get("Lossless Scaling Support", DEFAULTS["Lossless Scaling Support"])
        if keep_optional:
            self.locale = cfg.get("Language", DEFAULTS["Language"])
            self.lang_dd.value = "English" if self.locale == "EN" else "简体中文"
        
        saved_ctrl = cfg.get("Controller Model", DEFAULTS.get("Controller Model", "PICO"))
        self.ctrl_model_dd.value = saved_ctrl if saved_ctrl in self.ctrl_model_dd.options else "PICO"
        saved_env = cfg.get("Environment Model", DEFAULTS.get("Environment Model", "Default"))
        self.env_model_dd.value = saved_env if saved_env in self.env_model_dd.options else self.env_model_dd.options[0]
        self.torch_compile_cb.value = cfg.get("torch.compile", False)
        self.tensorrt_cb.value = cfg.get("TensorRT", False)
        self.recompile_trt_cb.value = cfg.get("Recompile TensorRT", DEFAULTS["Recompile TensorRT"])
        self.coreml_cb.value = cfg.get("CoreML", False)
        self.recompile_coreml_cb.value = cfg.get("Recompile CoreML", DEFAULTS["Recompile CoreML"])
        self.openvino_cb.value = cfg.get("OpenVINO", False)
        self.recompile_openvino_cb.value = cfg.get("Recompile OpenVINO", DEFAULTS["Recompile OpenVINO"])
        self.recompile_trt_cb.visible = self.tensorrt_cb.value and self.tensorrt_cb.visible
        self.recompile_coreml_cb.visible = self.coreml_cb.value and self.coreml_cb.visible
        self.recompile_openvino_cb.visible = self.openvino_cb.value and self.openvino_cb.visible
                # Capture tool
        ct = cfg.get("Capture Tool", DEFAULTS["Capture Tool"])
        self.capture_tool_dd.value = ct if ct in self.capture_tool_dd.options else (self.capture_tool_dd.options[0] if self.capture_tool_dd.options else '')
        # Run mode
        if keep_optional:
            run_mode = cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Local Viewer"))
            if run_mode == "3D Monitor" and OS_NAME != "Windows":
                run_mode = "Local Viewer"
            if run_mode == "OpenXR Link" and OS_NAME == "Darwin":
                run_mode = "Local Viewer"
            self.run_mode_key = run_mode
        self.stream_protocol_key = cfg.get("Stream Protocol", DEFAULTS.get("Stream Protocol", "RTMP"))
        self.stream_proto_dd.value = self.stream_protocol_key
        self.stream_port_tf.value = str(cfg.get("Streamer Port", DEFAULTS.get("Streamer Port", DEFAULT_PORT)))
        self.stream_quality_dd.value = str(cfg.get("Stream Quality", DEFAULTS["Stream Quality"]))
        self.stream_key_tf.value = cfg.get("Stream Key", DEFAULTS["Stream Key"])
        self.audio_dd.value = cfg.get("Stereo Mix", "")
        self.crf_tf.value = str(cfg.get("CRF", DEFAULTS["CRF"]))
        self.audio_delay_tf.value = str(cfg.get("Audio Delay", DEFAULTS["Audio Delay"]))
        self.capture_mode_key = cfg.get("Capture Mode", DEFAULTS["Capture Mode"])
        cm_t = UI_MESSAGES[self.locale]
        self.capture_mode_dd.value = cm_t["Monitor"] if self.capture_mode_key == "Monitor" else cm_t["Window"]
        self._sync_capture_mode_visibility()
        self._apply_stereo_output(cfg)
        self.update_tensorrt_visibility_based_on_model(selected_model)
        self.update_coreml_visibility_based_on_model(selected_model)
        self.update_openvino_visibility_based_on_model(selected_model)
        self.update_ui_texts()
        self._sync_visibility()
        self.update_stream_url()
        self.on_device_change(None)
        self.on_capture_tool_change(None)

    def _apply_stereo_output(self, cfg):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            self.stereo_monitor_dd.value = "Viewer Window"
            return
        saved = cfg.get("Stereo Output")
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        if saved is not None:
            label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == saved), None)
            if label and label != input_label:
                self.stereo_monitor_dd.value = label
                return
        fallback = None
        for lbl in self.monitor_label_to_index:
            if lbl != input_label:
                fallback = lbl
                break
        self.stereo_monitor_dd.value = fallback if fallback else "Viewer Window"

    def _get_monitor_count(self):
        try:
            import mss
            with mss.mss() as sct:
                return len(sct.monitors) - 1
        except Exception:
            return 0

    def update_stereo_monitor_menu(self):
        if not hasattr(self, 'stereo_monitor_dd'):
            return
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        opts = ["Viewer Window"]
        for label in self.monitor_label_to_index:
            if label != input_label:
                opts.append(label)
        current = self.stereo_monitor_dd.value
        valid = current in opts
        self.stereo_monitor_dd.options = opts
        if not valid:
            self.stereo_monitor_dd.value = opts[0] if opts else "Viewer Window"
        self.stereo_monitor_dd.update()

    def update_depth_resolution_options(self, model_name):
        resolutions = ALL_MODELS.get(model_name, {}).get("resolutions", [322])
        self.depth_res_dd.options = [str(r) for r in resolutions]
        cur = self.depth_res_dd.value
        if cur and cur in [str(r) for r in resolutions]:
            return
        try:
            cur_num = int(cur) if cur else 0
            closest = min(resolutions, key=lambda x: abs(x - cur_num))
            self.depth_res_dd.value = str(closest)
        except (ValueError, TypeError):
            self.depth_res_dd.value = str(resolutions[0])
        self.depth_res_dd.update()

    @property
    def current_model_name(self):
        family = self.depth_model_dd.value
        size = self.model_size_dd.value
        return FAMILY_SIZE_TO_MODEL.get((family, size), family if not size else f"{family}-{size}")

    def on_model_family_change(self, e):
        family = e.control.value
        sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in sizes]
        if sizes:
            self.model_size_dd.value = sizes[0]
        else:
            self.model_size_dd.value = ""
        self.model_size_dd.update()
        self._on_model_changed()

    def on_model_size_change(self, e):
        self._on_model_changed()

    def _on_model_changed(self):
        model = self.current_model_name
        self._config["Depth Model"] = model
        self.update_depth_resolution_options(model)
        self.auto_enable_optimizers_based_on_device()
        if not devices_module.IS_ROCM and "CUDA" in self.device_dd.value:
            self.update_tensorrt_visibility_based_on_model(model)
        elif "MPS" in self.device_dd.value:
            self.update_coreml_visibility_based_on_model(model)
        elif "XPU" in self.device_dd.value:
            self.update_openvino_visibility_based_on_model(model)
        self._fit_window_to_content()

    def on_device_change(self, e):
        device_label = e.control.value if e else self.device_dd.value
        self._config["Computing Device"] = self.device_label_to_index.get(device_label, 0)
        if OS_NAME in ("Windows", "Darwin"):
            new_opts = _get_capture_tool_options(device_label)
            self.capture_tool_dd.options = [o for o in new_opts]
            if self.capture_tool_dd.value not in new_opts:
                self.capture_tool_dd.value = new_opts[0]
            self.capture_tool_dd.update()
        self._update_accelerator_visibility(device_label)
        self._fit_window_to_content()

    def _update_accelerator_visibility(self, device_label):
        cuda = "CUDA" in device_label
        dml = "DirectML" in device_label
        mps = "MPS" in device_label
        xpu = "XPU" in device_label
        other = not (cuda or dml or mps or xpu)
        self.torch_compile_cb.visible = cuda
        self.tensorrt_cb.visible = cuda and not devices_module.IS_ROCM
        self.tensorrt_cb.disabled = False
        self.recompile_trt_cb.visible = self.tensorrt_cb.value if self.tensorrt_cb.visible else False
        self.coreml_cb.visible = mps
        self.coreml_cb.disabled = False
        self.recompile_coreml_cb.visible = self.coreml_cb.value if self.coreml_cb.visible else False
        self.openvino_cb.visible = xpu
        self.openvino_cb.disabled = False
        self.recompile_openvino_cb.visible = self.openvino_cb.value if self.openvino_cb.visible else False
        self.fp16_cb.visible = not (dml or mps)
        self.acceleration_label.visible = not (dml or other)
        is_windows_or_mac = OS_NAME in ("Windows", "Darwin")
        self.capture_tool_label.visible = is_windows_or_mac
        self.capture_tool_dd.visible = is_windows_or_mac
        if dml or other:
            self.torch_compile_cb.visible = False
            self.tensorrt_cb.visible = False
            self.recompile_trt_cb.visible = False
            self.coreml_cb.visible = False
            self.recompile_coreml_cb.visible = False
            self.openvino_cb.visible = False
            self.recompile_openvino_cb.visible = False
        current_model = self.current_model_name
        if cuda and not devices_module.IS_ROCM:
            self.update_tensorrt_visibility_based_on_model(current_model)
        if mps:
            self.update_coreml_visibility_based_on_model(current_model)
        if xpu:
            self.update_openvino_visibility_based_on_model(current_model)

    def _on_trt_toggle(self, e):
        self.recompile_trt_cb.visible = self.tensorrt_cb.value
        self._fit_window_to_content()

    def _on_coreml_toggle(self, e):
        self.recompile_coreml_cb.visible = self.coreml_cb.value
        self._fit_window_to_content()

    def _on_openvino_toggle(self, e):
        self.recompile_openvino_cb.visible = self.openvino_cb.value
        self._fit_window_to_content()

    def auto_enable_optimizers_based_on_device(self):
        device_label = self.device_dd.value
        model_lower = (self.current_model_name or "").lower()
        self.tensorrt_cb.value = False
        self.coreml_cb.value = False
        self.openvino_cb.value = False
        if "CUDA" in device_label and not devices_module.IS_ROCM:
            should = not any(kw in model_lower for kw in DISABLE_TRT_KEYWORDS)
            self.tensorrt_cb.value = should
            self.torch_compile_cb.value = True
        elif "MPS" in device_label:
            should = not any(kw in model_lower for kw in DISABLE_COREML_KEYWORDS)
            self.coreml_cb.value = should
        elif "XPU" in device_label:
            should = not any(kw in model_lower for kw in DISABLE_OPENVINO_KEYWORDS)
            self.openvino_cb.value = should
        self._on_trt_toggle(None)
        self._on_coreml_toggle(None)
        self._on_openvino_toggle(None)

    def update_tensorrt_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if not devices_module.IS_ROCM and "CUDA" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_TRT_KEYWORDS)
            if should_disable:
                self.tensorrt_cb.value = False
                self.tensorrt_cb.disabled = True
                self.recompile_trt_cb.visible = False
            else:
                self.tensorrt_cb.disabled = False

    def update_coreml_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "MPS" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_COREML_KEYWORDS)
            if should_disable:
                self.coreml_cb.value = False
                self.coreml_cb.disabled = True
                self.recompile_coreml_cb.visible = False
            else:
                self.coreml_cb.disabled = False

    def update_openvino_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "XPU" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_OPENVINO_KEYWORDS)
            if should_disable:
                self.openvino_cb.value = False
                self.openvino_cb.disabled = True
                self.recompile_openvino_cb.visible = False
            else:
                self.openvino_cb.disabled = False

    def on_capture_tool_change(self, e):
        tool = e.control.value if e else self.capture_tool_dd.value
        if tool == "DesktopDuplication":
            self.capture_mode_key = "Monitor"
            self.capture_mode_dd.value = UI_MESSAGES[self.locale]["Monitor"]
            self.capture_mode_dd.disabled = True
            self.set_status(UI_MESSAGES[self.locale]["DesktopDuplication selected: Window capture mode disabled."])
        else:
            self.capture_mode_dd.disabled = False
            self.set_status("", key="")
        self.capture_mode_dd.update()
        self._sync_capture_mode_visibility()

    def _sync_capture_mode_visibility(self):
        """Show monitor_dd or window_dd based on capture_mode_key."""
        if self.capture_mode_key == "Monitor":
            self.monitor_dd.visible = True
            self.window_dd.visible = False
        else:
            self.monitor_dd.visible = False
            self.window_dd.visible = True
        self._safe_update(self.monitor_dd, self.window_dd)
        self._fit_window_to_content()

    def on_capture_mode_change(self, e):
        mode = e.control.value
        texts = UI_MESSAGES[self.locale]
        reverse_map = {texts["Monitor"]: "Monitor", texts["Window"]: "Window"}
        self.capture_mode_key = reverse_map.get(mode, "Monitor")
        self._sync_capture_mode_visibility()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()

    def on_window_selected(self, e):
        label = e.control.value if e else self.window_dd.value
        # Extract handle from label: "Title [h:123456]"
        m = re.search(r'\[h:(\d+)\]$', label)
        target_handle = int(m.group(1)) if m else None
        # Extract clean title (strip suffix)
        display_title = re.sub(r'\s*\[h:\d+\]$', '', label).strip()
        self.selected_window_name = display_title
        for win in self._window_objects:
            wh = win.get("handle") or 0
            if target_handle is not None and wh == target_handle:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
            elif target_handle is None and win["title"] == display_title:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
        self._fit_window_to_content()

    def _on_monitor_change(self, e):
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        self.set_status(f"{UI_MESSAGES[self.locale]['Selected input monitor:']} {e.control.value}")

    def on_run_mode_change(self, e):
        label = e.control.value
        texts = UI_MESSAGES[self.locale]
        mode_map = {
            texts.get("Local Viewer", "Local Viewer"): "Local Viewer",
            texts.get("OpenXR Link", "OpenXR Link"): "OpenXR Link",
            texts.get("RTMP Streamer", "RTMP Streamer"): "RTMP Streamer",
            texts.get("MJPEG Streamer", "MJPEG Streamer"): "MJPEG Streamer",
            texts.get("Legacy Streamer", "Legacy Streamer"): "Legacy Streamer",
            texts.get("3D Monitor", "3D Monitor"): "3D Monitor",
        }
        self.run_mode_key = mode_map.get(label, "Local Viewer")
        self._config["Run Mode"] = self.run_mode_key
        self._sync_visibility()

    def on_advanced_device_change(self, e):
        self._sync_visibility()
        self._fit_window_to_content()
        try:
            self.page.update()
        except Exception:
            pass

    def _sync_device_advanced_visibility(self, mode):
        advanced = bool(getattr(self, "advanced_device_cb", None) and self.advanced_device_cb.value)
        show_timing = advanced and mode in ["Local Viewer", "3D Monitor", "OpenXR Link"]
        show_enhance = advanced and mode in ["Local Viewer", "3D Monitor"]
        self.row6b.visible = show_timing
        self.row6c.visible = show_enhance
        self.target_fps_label.visible = show_timing
        self.target_fps_dd.visible = show_timing
        self.local_vsync_cb.visible = advanced and mode in ["Local Viewer", "3D Monitor"]
        self.upscaler_label.visible = show_enhance
        self.upscaler_dd.visible = show_enhance
        self.upscaler_sharpness_label.visible = show_enhance
        self.upscaler_sharpness_dd.visible = show_enhance

    def _sync_visibility(self):
        mode = self.run_mode_key
        texts = UI_MESSAGES[self.locale]
        mode_reverse = {
            "Local Viewer": texts["Local Viewer"],
            "OpenXR Link": texts["OpenXR Link"],
            "RTMP Streamer": texts["RTMP Streamer"],
            "MJPEG Streamer": texts["MJPEG Streamer"],
            "Legacy Streamer": texts["Legacy Streamer"],
            "3D Monitor": texts["3D Monitor"],
        }
        self.run_mode_dd.value = mode_reverse.get(mode, texts["Local Viewer"])
        is_openxr = mode == "OpenXR Link"
        self.display_mode_label.visible = not is_openxr
        self.display_mode_dd.visible = not is_openxr
        self.xr_preview_cb.visible = is_openxr
        self._sync_device_advanced_visibility(mode)
        self.row7b.visible = is_openxr
        if is_openxr:
            self.showfps_cb.visible = False
            self.fill_16_9_cb.visible = False
            self.fix_aspect_cb.visible = False
        else:
            self.showfps_cb.visible = True
            self.fill_16_9_cb.visible = True
            self.fix_aspect_cb.visible = mode in ["Local Viewer", "3D Monitor"]
        if mode in ["Legacy Streamer", "3D Monitor"]:
            self.display_mode_dd.options = [m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"]]
        else:
            self.display_mode_dd.options = [m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB", "Depth Map", "Anaglyph", "Interleaved", "Mono", "Leia"]]
        mon_count = self._get_monitor_count()
        stereo_full = mode in ["Local Viewer", "3D Monitor", "RTMP Streamer"] and mon_count > 1
        self.stereo_output_label.visible = not is_openxr
        self.stereo_monitor_dd.visible = stereo_full
        if hasattr(self, '_stereo_spacer'):
            self._stereo_spacer.visible = stereo_full
        self.row9.visible = (not is_openxr) and (stereo_full or self.fill_16_9_cb.visible or self.fix_aspect_cb.visible or self.lossless_cb.visible)
        if mode == "3D Monitor":
            self.stereo_monitor_dd.value = self.monitor_dd.value
        row_map = self._get_streamer_row_map()
        indices = row_map.get(mode, [])
        self._show_streamer_rows(*indices)
        self.lossless_cb.visible = (OS_NAME == "Windows" and mode == "RTMP Streamer")
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        if mode in ["Local Viewer", "RTMP Streamer"]:
            self._auto_select_stereo_monitor()
        if mode == "RTMP Streamer":
            self.populate_audio_devices()
            self.auto_select_stereo_mix()
            saved_mix = self._config.get("Stereo Mix", "")
            if saved_mix and saved_mix in (self.audio_dd.options or []):
                self.audio_dd.value = saved_mix
        self.update_stream_url()
        self._fit_window_to_content()
        self.page.update()

    def _auto_select_stereo_monitor(self):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            return
        cur = self.stereo_monitor_dd.value
        valid = cur and cur in self.stereo_monitor_dd.options and cur != "Viewer Window"
        if not valid:
            input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
            for lbl in self.monitor_label_to_index:
                if lbl != input_label:
                    self.stereo_monitor_dd.value = lbl
                    break

    def on_theme_change(self, e):
        color = e.control.value
        cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
        color = cn_map.get(color, color.lower())
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        if color == "system":
            self.page.theme_mode = ft.ThemeMode.SYSTEM
            self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        else:
            self.page.theme = ft.Theme(color_scheme_seed=color, font_family=font)
        self.page.update()

    def on_language_change(self, e):
        lang_display = e.control.value
        _LANG_MAP = {"English": "EN", "简体中文": "CN"}
        lang = _LANG_MAP.get(lang_display, "EN")
        if is_supported_locale(lang):
            self.locale = lang
            self._config["Language"] = lang
            self.update_ui_texts()
            self._sync_visibility()
            if self._status_key:
                self.set_status(UI_MESSAGES[self.locale].get(self._status_key, self.status_text.value), key=self._status_key)
            # Theme: update_ui_texts already set options, here we only set the value
            t = UI_MESSAGES[self.locale]
            cur = self.theme_dd.value
            cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
            key = cn_map.get(cur, cur.lower() if cur else "system")
            self.theme_dd.value = t.get(key, key) if self.locale == "CN" else key.capitalize()
            self._fit_window_to_content()
            self.page.update()

    def update_ui_texts(self):
        t = UI_MESSAGES[self.locale]
        self.depth_model_label.value = t["Depth Model:"]
        self.fp16_cb.label = t["FP16"]
        self.depth_resolution_label.value = t["Depth Resolution:"]
        self.convergence_label.value = t["Convergence:"]
        self.depth_strength_label.value = t["Depth Strength:"]
        self.depth_quick_label.value = t["Depth Quick:"]
        depth_quick_key = self._display_to_depth_quick(self.depth_quick_dd.value)
        self.depth_quick_dd.options = [t["Soft"], t["Standard"], t["Enhanced"]]
        self.depth_quick_dd.value = self._depth_quick_to_display(depth_quick_key)
        self.foreground_scale_label.value = t["Foreground Scale:"]
        self.antialiasing_label.value = t["Anti-aliasing:"]
        self.ipd_label.value = t["IPD (m):"]
        self.stereo_scale_label.value = t["Stereo Scale:"]
        self.stereo_preset_label.value = t["Stereo Mode:"]
        preset_key = self._display_to_preset(self.stereo_preset_dd.value)
        self.stereo_preset_dd.options = [t["Cinema"], t["Game / Low Latency"], t["Still Image / HQ"], t["Debug / Export"]]
        self.stereo_preset_dd.value = self._preset_to_display(preset_key)
        self.stereo_quality_label.value = t["Synthetic View:"]
        stereo_quality_key = self._display_to_stereo_quality(self.stereo_quality_dd.value)
        self.stereo_quality_dd.options = self._stereo_quality_options()
        self.stereo_quality_dd.value = self._stereo_quality_to_display(stereo_quality_key)
        self.max_shift_label.value = t["Max Shift Ratio:"]
        self.temporal_strength_label.value = t["Temporal Strength:"]
        self.scene_reset_label.value = t["Scene Threshold:"]
        self.reset_cooldown_label.value = t["Reset Cooldown:"]
        self.edge_dilation_label.value = t["Edge Dilation:"]
        self.edge_threshold_label.value = t["Edge Threshold:"]
        self.anaglyph_label.value = t["Anaglyph:"]
        self.cross_eyed_cb.label = t["Cross Eyed"]
        self.advanced_stereo_cb.label = t["Advanced Stereo"]
        self.acceleration_label.value = t["Inference Acceleration:"]
        self.torch_compile_cb.label = t["torch.compile"]
        self.tensorrt_cb.label = t["TensorRT"]
        self.coreml_cb.label = t["CoreML"]
        self.openvino_cb.label = t["OpenVINO"]
        self.recompile_trt_cb.label = t["Recompile TensorRT"]
        self.recompile_coreml_cb.label = t["Recompile CoreML"]
        self.recompile_openvino_cb.label = t["Recompile OpenVINO"]
        self.computing_device_label.value = t["Computing Device:"]
        self.advanced_device_cb.label = t["Advanced Device Options"]
        self.showfps_cb.label = t["Show FPS"]
        self.capture_tool_label.value = t["Capture Tool:"]
        self.run_mode_label.value = t["Run Mode:"]
        self.display_mode_label.value = t["Display Mode:"]
        self.xr_preview_cb.label = t.get("XR Preview Window", "XR画面预览窗口" if self.locale == "CN" else "XR Preview Window")
        self.local_vsync_cb.label = t.get("Local VSync", "Local VSync")
        self.target_fps_label.value = t.get("Capture FPS:", "Capture FPS:")
        target_fps_value = self._target_fps_from_display(self.target_fps_dd.value)
        self.target_fps_dd.options = [t["Auto"], "60", "72", "80", "90", "120"]
        self.target_fps_dd.value = self._target_fps_to_display(target_fps_value)
        self.upscaler_label.value = t.get("Upscaler:", "Upscaler:")
        self.upscaler_sharpness_label.value = t.get("Upscaler Sharpness:", "Sharpness:")
        upscaler_value = self._upscaler_from_display(self.upscaler_dd.value)
        self.upscaler_dd.options = self._upscaler_display_options()
        self.upscaler_dd.value = self._upscaler_to_display(upscaler_value)
        self.stereo_output_label.value = t["Stereo Output:"]
        self.theme_label.value = t["Theme:"]
        theme_display = ["System","Blue","Green","Red","Purple","Orange","Teal","Pink","Grey"]
        self.theme_dd.options = [t.get(k.lower(), k) for k in theme_display]
        cur = self.theme_dd.value
        cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
        key = cn_map.get(cur, cur.lower() if cur else "system")
        self.theme_dd.value = t.get(key, key) if self.locale == "CN" else key.capitalize()
        self.fill_16_9_cb.label = t["Fill 16:9"]
        self.fix_aspect_cb.label = t["Fix Viewer Aspect"]
        self.lossless_cb.label = t["Lossless Scaling Support"]
        self.controller_label.value = t["Controller:"]
        self.environment_label.value = t["Environment:"]
        self.lang_label.value = t["Set Language:"]
        run_mode_texts = {}
        if OS_NAME == "Darwin":
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"],
                "RTMP Streamer": t["RTMP Streamer"],
                "MJPEG Streamer": t["MJPEG Streamer"],
                "Legacy Streamer": t["Legacy Streamer"],
            }
        else:
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"],
                "OpenXR Link": t["OpenXR Link"],
                "RTMP Streamer": t["RTMP Streamer"],
                "MJPEG Streamer": t["MJPEG Streamer"],
                "Legacy Streamer": t["Legacy Streamer"],
            }
            if OS_NAME == "Windows":
                run_mode_texts["3D Monitor"] = t["3D Monitor"]
        self.run_mode_dd.options = [v for v in run_mode_texts.values()]
        self.capture_mode_dd.options = [
            t["Monitor"],
            t["Window"],
        ]
        self.capture_mode_dd.value = t["Monitor"] if self.capture_mode_key == "Monitor" else t["Window"]
        self.capture_mode_dd.reapply_width()
        self.stream_url_label.value = t["Streamer URL"]
        self.stream_port_label.value = t["Streamer Port:"]
        self.stream_quality_label.value = t["Stream Quality:"]
        self.stream_proto_label.value = t["Stream Protocol:"]
        self.stream_key_label.value = t["Stream Key"]
        self.audio_label.value = t["Stereo Mix"]
        self.crf_label.value = t["CRF"]
        self.audio_delay_label.value = t["Audio Delay"]
        self.preview_btn.content.value = t["Preview"]
        self.refresh_btn.content.value = t["Refresh"]
        self.reset_btn.content.value = t["Reset"]
        self.stop_btn.content.value = t["Stop"]
        self.run_btn.content.value = t["Run"]
        # Update tooltips
        def _set_tooltip(ctrl, text):
            if hasattr(ctrl, "set_tooltip"):
                ctrl.set_tooltip(text)
            else:
                try:
                    ctrl.tooltip = text
                except Exception:
                    pass

        _set_tooltip(self.window_dd, t["tooltip_window"])
        for ctrl, key in [
            (self.depth_model_dd, "tooltip_depth_model"),
            (self.model_size_dd, "tooltip_depth_model"),
            (self.depth_res_dd, "tooltip_depth_res"),
            (self.convergence_dd, "tooltip_convergence"),
            (self.depth_strength_dd, "tooltip_depth_strength"),
            (self.depth_quick_dd, "tooltip_depth_quick"),
            (self.stereo_preset_dd, "tooltip_stereo_preset"),
            (self.stereo_quality_dd, "tooltip_stereo_quality"),
            (self.max_shift_dd, "tooltip_max_shift"),
            (self.temporal_strength_dd, "tooltip_temporal_strength"),
            (self.scene_reset_dd, "tooltip_scene_reset"),
            (self.reset_cooldown_dd, "tooltip_reset_cooldown"),
            (self.edge_dilation_dd, "tooltip_edge_dilation"),
            (self.edge_threshold_dd, "tooltip_edge_threshold"),
            (self.anaglyph_dd, "tooltip_anaglyph"),
            (self.cross_eyed_cb, "tooltip_cross_eyed"),
            (self.advanced_stereo_cb, "tooltip_advanced_stereo"),
            (self.foreground_scale_dd, "tooltip_foreground_scale"),
            (self.antialiasing_dd, "tooltip_antialiasing"),
            (self.ipd_dd, "tooltip_ipd"),
            (self.stereo_scale_dd, "tooltip_stereo_scale"),
            (self.device_dd, "tooltip_device"),
            (self.advanced_device_cb, "tooltip_advanced_device_options"),
            (self.capture_tool_dd, "tooltip_capture_tool"),
            (self.run_mode_dd, "tooltip_run_mode"),
            (self.display_mode_dd, "tooltip_display_mode"),
            (self.local_vsync_cb, "tooltip_local_vsync"),
            (self.target_fps_dd, "tooltip_target_fps"),
            (self.ctrl_model_dd, "tooltip_ctrl_model"),
            (self.env_model_dd, "tooltip_env_model"),
            (self.capture_mode_dd, "tooltip_capture_mode"),
            (self.monitor_dd, "tooltip_monitor"),
            (self.stereo_monitor_dd, "tooltip_stereo_monitor"),
            (self.lang_dd, "tooltip_lang"),
            (self.theme_dd, "tooltip_theme"),
            (self.stream_quality_dd, "tooltip_stream_quality"),
            (self.stream_proto_dd, "tooltip_stream_proto"),
            (self.audio_dd, "tooltip_audio"),
            (self.stream_port_tf, "tooltip_stream_port"),
            (self.stream_key_tf, "tooltip_stream_key"),
            (self.crf_tf, "tooltip_crf"),
            (self.audio_delay_tf, "tooltip_audio_delay"),
        ]:
            _set_tooltip(ctrl, t.get(key, UI_MESSAGES["EN"].get(key, key)))
        self._auto_align_labels(force=True)

    def _safe_update(self, *controls):
        for c in controls:
            try:
                c.update()
            except RuntimeError:
                pass
            except Exception as e:
                print(f"[Warning] _safe_update failed: {e}")

    def update_stream_url(self, e=None):
        if not self.stream_container.visible:
            return
        protocol = self.stream_proto_dd.value
        port = self.stream_port_tf.value or str(DEFAULT_PORT)
        stream_key = self.stream_key_tf.value or "live"
        local_ip = get_local_ip()
        if self.run_mode_key in ["MJPEG Streamer", "Legacy Streamer"]:
            self.stream_url_tf.content.controls[0].value = f"http://{local_ip}:{port}/"
        else:
            templates = {
                "RTMP": f"rtmp://{local_ip}:{port}/{stream_key}",
                "RTSP": f"rtsp://{local_ip}:{port}/{stream_key}",
                "HLS": f"http://{local_ip}:{port}/{stream_key}/",
                "HLS M3U8": f"http://{local_ip}:{port}/{stream_key}/index.m3u8",
                "WebRTC": f"http://{local_ip}:{port}/{stream_key}/",
            }
            self.stream_url_tf.content.controls[0].value = templates.get(protocol, f"http://{local_ip}:{port}/{stream_key}/")
        self._safe_update(self.stream_url_tf)
        self.preview_btn.visible = protocol not in ["RTMP", "RTSP"]
        self._safe_update(self.preview_btn)

    def _on_stream_protocol_change(self, e):
        self.stream_protocol_key = e.control.value
        self._config["Stream Protocol"] = self.stream_protocol_key
        self.update_stream_url()
        self._fit_window_to_content()

    def _on_stream_key_change(self, e):
        val = e.control.value or ""
        if not re.match(r'^[A-Za-z0-9_-]*$', val) or len(val) > 64:
            self.set_status(UI_MESSAGES[self.locale]["err_stream_key"])
        self._config["Stream Key"] = val
        self.update_stream_url()

    def populate_audio_devices(self):
        if OS_NAME == "Linux":
            self._populate_audio_linux()
        else:
            self._populate_audio_generic()
        if self.audio_devices:
            self.audio_dd.options = [d for d in self.audio_devices]
            self.audio_dd.value = self.audio_devices[0]
            self.audio_dd.update()
            if self.audio_devices[0] in ["No Stereo Mix device found", "sounddevice not available"]:
                self.set_status(self.audio_devices[0])

    def _populate_audio_generic(self):
        """Use sounddevice to find Stereo Mix / loopback devices."""
        self.audio_devices = []
        try:
            import sounddevice as sd
            all_devices = sd.query_devices()
            found = set()
            for dev in all_devices:
                name = (dev.get("name", "") or "").lower()
                in_ch = dev.get("max_input_channels", 0)
                out_ch = dev.get("max_output_channels", 0)
                if in_ch > 0 or out_ch > 0:
                    for mix in STEREO_MIX_NAMES:
                        if mix in name and "audio stereo input" not in name:
                            found.add(dev.get("name"))
                            break
                    if "virtual-audio-capturer" in name:
                        found.add(dev.get("name"))
            if not found and OS_NAME == "Darwin":
                print("[Info] No audio capture devices found on MacOS.\nRecommended tools:\n- BlackHole: https://existential.audio/blackhole/\n- Virtual Desktop Streamer: https://www.vrdesktop.net/\n- Loopback: https://rogueamoeba.com/loopback/")
                self.audio_devices = ["No audio capture devices found"]
            elif not found and OS_NAME == "Windows":
                print("[Warning] No Stereo Mix devices found, please enable it in audio settings.\nIf no Stereo Mix, install 'Screen Capture Recorder':\nhttps://github.com/rdp/screen-capture-recorder-to-video-windows-free/releases")
                self.audio_devices = ["virtual-audio-capturer"]
            else:
                self.audio_devices = list(found) or ["No Stereo Mix device found"]
        except ImportError:
            self.audio_devices = ["sounddevice not available"]
        except Exception as e:
            self.audio_devices = [f"Error: {e}"]

    def _populate_audio_linux(self):
        self.audio_devices = []
        try:
            result = subprocess.run(["pacmd", "list-sources"],
                                    capture_output=True, text=True, check=True)
            sources = []
            for block in result.stdout.split("index:")[1:]:
                m = re.search(r"name:\s*<(.+?)>", block)
                if m:
                    sources.append(m.group(1))
            self.audio_devices = sources or ["No audio sources found"]
        except Exception:
            self.audio_devices = ["pacmd not available"]

    def auto_select_stereo_mix(self):
        """Auto-select the first stereo-mix-like device."""
        if not self.audio_devices:
            return
        for dev in self.audio_devices:
            dl = dev.lower()
            for mix in STEREO_MIX_NAMES:
                if mix in dl and "audio stereo input" not in dl:
                    self.audio_dd.value = dev
                    self.audio_dd.update()
                    return
        # Fallback: virtual-audio-capturer
        for dev in self.audio_devices:
            if "virtual-audio-capturer" in dev.lower():
                self.audio_dd.value = dev
                self.audio_dd.update()
                return
        self.audio_dd.value = "No Stereo Mix device found"
        self.audio_dd.update()

    def refresh_monitor_and_window(self, e=None):
        self.populate_monitors()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        if self.run_mode_key == "RTMP Streamer":
            self.populate_audio_devices()
            self.auto_select_stereo_mix()
        self.update_stereo_monitor_menu()
        self._sync_visibility()
        self._fit_window_to_content()
        if self.capture_mode_key == "Monitor" and self.monitor_dd.value:
            self.set_status(f"{UI_MESSAGES[self.locale]['Selected input monitor:']} {self.monitor_dd.value}")
        elif self.capture_mode_key == "Window" and self.selected_window_name:
            self.set_status(f"{UI_MESSAGES[self.locale]['Selected input window:']} {self.selected_window_name}")

    def refresh_window_list(self):
        try:
            windows = list_windows()
            if not windows:
                self.window_dd.options = []
                self.window_dd.update()
                return
            self._window_objects = windows
            # Include handle in label to avoid duplicate title ambiguity
            win_labels = [f"{w['title']} [h:{w['handle'] or 0}]" for w in windows]
            self.window_dd.options = win_labels
            if self.selected_window_name:
                labels_by_title = [lbl for lbl in win_labels if lbl.startswith(self.selected_window_name + " [")]
                selected_handle_str = f"[h:{self.selected_window_handle or 0}]"
                match = next((lbl for lbl in labels_by_title if selected_handle_str in lbl), None)
                if match:
                    self.window_dd.value = match
                elif labels_by_title:
                    self.window_dd.value = labels_by_title[0]
                    self.on_window_selected(None)
                else:
                    self.window_dd.value = win_labels[0] if windows else ""
                    self.on_window_selected(None)
            elif windows:
                self.window_dd.value = win_labels[0]
                self.on_window_selected(None)
            self.window_dd.update()
        except Exception as e:
            self.set_status(UI_MESSAGES[self.locale]["err_refresh_window"].format(e))


    # URL actions

    def preview_in_browser(self, e):
        try:
            import webbrowser
            url = self.stream_url_tf.content.controls[0].value
            if not url.startswith(("http://", "https://")):
                self.set_status(UI_MESSAGES[self.locale]["invalid_url_scheme"].format(url))
                return
            webbrowser.open(url)
            self.set_status(f"{UI_MESSAGES[self.locale]['Opening URL in browser']}: {url}")
        except Exception as ex:
            self.set_status(UI_MESSAGES[self.locale]["error_preview"].format(ex))

    def copy_url_to_clipboard(self, e):
        url = self.stream_url_tf.content.controls[0].value
        if url:
            try:
                import pyperclip
                pyperclip.copy(url)
            except ImportError:
                import subprocess
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=url, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run("pbcopy", input=url, text=True)
            self.set_status(UI_MESSAGES[self.locale]["url_copied"], key="url_copied")
            asyncio.create_task(self._fade_status(2.0))

    async def _fade_status(self, delay):
        await asyncio.sleep(delay)
        self.set_status("", key="")


    def on_stereo_preset_change(self, e=None):
        preset = self._display_to_preset(self.stereo_preset_dd.value)
        if self._apply_stereo_preset_values(preset):
            self._schedule_stereo_hot_save()
        else:
            self.on_stereo_hot_param_change(e)

    def _apply_stereo_preset_values(self, preset):
        values = self._stereo_preset_gui_values(preset)
        if not values:
            return False
        self.stereo_quality_dd.value = self._stereo_quality_to_display(values["quality"])
        self.depth_strength_dd.value = f"{values['depth_strength']:.1f}"
        self.depth_quick_dd.value = self._depth_quick_to_display(values["depth_quick"])
        self.convergence_dd.value = f"{values['convergence']:.2f}".rstrip("0").rstrip(".")
        self.max_shift_dd.value = f"{values['max_shift_ratio']:.2f}"
        self.stereo_scale_dd.value = f"{values['stereo_scale']:.1f}"
        self.temporal_strength_dd.value = f"{values['temporal_strength']:.2f}"
        self.foreground_scale_dd.value = f"{values['foreground_scale']:.1f}"
        self.antialiasing_dd.value = str(values["antialiasing"])
        self.edge_dilation_dd.value = str(values["edge_dilation"])
        self.edge_threshold_dd.value = f"{values['edge_threshold']:.2f}"
        self.cross_eyed_cb.value = False
        for ctrl in (
            self.stereo_quality_dd,
            self.depth_strength_dd,
            self.depth_quick_dd,
            self.convergence_dd,
            self.max_shift_dd,
            self.temporal_strength_dd,
            self.foreground_scale_dd,
            self.antialiasing_dd,
            self.edge_dilation_dd,
            self.edge_threshold_dd,
            self.cross_eyed_cb,
        ):
            try:
                ctrl.update()
            except Exception:
                pass
        return True

    @staticmethod
    def _stereo_preset_gui_values(preset):
        return {
            "cinema": {
                "quality": "quality_4k",
                "depth_strength": 2.4,
                "depth_quick": "Enhanced",
                "convergence": 0.25,
                "max_shift_ratio": 0.05,
                "stereo_scale": 0.5,
                "temporal_strength": 0.75,
                "foreground_scale": 0.5,
                "antialiasing": 1,
                "edge_dilation": 2,
                "edge_threshold": 0.04,
            },
            "game_low_latency": {
                "quality": "fast_plus",
                "depth_strength": 1.6,
                "depth_quick": "Soft",
                "convergence": 0.25,
                "max_shift_ratio": 0.04,
                "stereo_scale": 0.5,
                "temporal_strength": 0.25,
                "foreground_scale": 0.0,
                "antialiasing": 0,
                "edge_dilation": 1,
                "edge_threshold": 0.04,
            },
            "still_image_hq": {
                "quality": "hq_4k",
                "depth_strength": 3.0,
                "depth_quick": "Enhanced",
                "convergence": 0.25,
                "max_shift_ratio": 0.06,
                "stereo_scale": 0.5,
                "temporal_strength": 0.00,
                "foreground_scale": 0.5,
                "antialiasing": 2,
                "edge_dilation": 3,
                "edge_threshold": 0.04,
            },
            "debug_export": {
                "quality": "quality_4k",
                "depth_strength": 2.4,
                "depth_quick": "Enhanced",
                "convergence": 0.25,
                "max_shift_ratio": 0.05,
                "stereo_scale": 0.5,
                "temporal_strength": 0.75,
                "foreground_scale": 0.0,
                "antialiasing": 0,
                "edge_dilation": 2,
                "edge_threshold": 0.04,
            },
        }.get(preset)

    def on_stereo_hot_param_change(self, e=None):
        self._schedule_stereo_hot_save()

    def _schedule_stereo_hot_save(self, delay=0.15):
        task = getattr(self, "_hot_save_task", None)
        if task and not task.done():
            task.cancel()
        try:
            self._hot_save_task = asyncio.create_task(self._save_stereo_hot_params_after_delay(delay))
        except RuntimeError:
            self._save_stereo_hot_params()

    async def _save_stereo_hot_params_after_delay(self, delay):
        try:
            await asyncio.sleep(delay)
            self._save_stereo_hot_params()
        except asyncio.CancelledError:
            return

    def _save_stereo_hot_params(self):
        path = os.path.join(BASE_DIR, "settings.yaml")
        cfg = self._config.copy()
        if os.path.exists(path):
            try:
                loaded = read_yaml(path)
                if loaded:
                    cfg.update(loaded)
            except Exception:
                pass
        temporal_strength = self._parse_float(self.temporal_strength_dd.value, DEFAULTS["Temporal Strength"])
        scene_reset_threshold = self._parse_float(self.scene_reset_dd.value, DEFAULTS["Scene Reset Threshold"])
        antialias_strength = self._parse_float(self.antialiasing_dd.value, DEFAULTS["Depth Antialias Strength"])
        foreground_scale = self._clamp_foreground_scale(self._parse_float(self.foreground_scale_dd.value, DEFAULTS["Foreground Scale"]))
        self.foreground_scale_dd.value = f"{foreground_scale:.1f}"
        cfg.update({
            "Stereo Preset": self._display_to_preset(self.stereo_preset_dd.value),
            "Stereo Quality": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "Synthetic View": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "IPD": self._parse_int(self.ipd_dd.value, int(DEFAULTS["IPD"] * 1000)) / 1000.0,
            "Stereo Scale": self._parse_float(self.stereo_scale_dd.value, DEFAULTS["Stereo Scale"]),
            "Convergence": self._parse_float(self.convergence_dd.value, DEFAULTS["Convergence"]),
            "Depth Strength": self._parse_float(self.depth_strength_dd.value, DEFAULTS["Depth Strength"]),
            "Depth Quick": self._display_to_depth_quick(self.depth_quick_dd.value),
            "Max Shift Ratio": self._parse_float(self.max_shift_dd.value, DEFAULTS["Max Shift Ratio"]),
            "Temporal": temporal_strength > 0.0,
            "Temporal Strength": temporal_strength,
            "Auto Scene Reset": scene_reset_threshold > 0.0,
            "Scene Reset Threshold": scene_reset_threshold,
            "Reset Cooldown Frames": self._parse_int(self.reset_cooldown_dd.value, DEFAULTS["Reset Cooldown Frames"]),
            "Foreground Scale": foreground_scale,
            "Anti-aliasing": self._parse_int(self.antialiasing_dd.value, DEFAULTS["Anti-aliasing"]),
            "Depth Antialias Strength": antialias_strength,
            "Edge Dilation": self._parse_int(self.edge_dilation_dd.value, DEFAULTS["Edge Dilation"]),
            "Edge Threshold": self._parse_float(self.edge_threshold_dd.value, DEFAULTS["Edge Threshold"]),
            "Anaglyph Method": self.anaglyph_dd.value,
            "Cross Eyed": bool(self.cross_eyed_cb.value),
        })
        ok, err = save_yaml(path, cfg)
        if ok:
            self._config.update(cfg)
            self.set_status(UI_MESSAGES[self.locale]["stereo_parameters_saved"], key="stereo_parameters_saved")
        else:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))

    def set_status(self, msg, key=None):
        self.status_text.value = msg
        if key is not None:
            self._status_key = key
        self._safe_update(self.status_text)

    def _set_running_ui(self, running: bool):
        """Toggle Run/Stop button enabled state."""
        self.run_btn.disabled = running
        self.stop_btn.disabled = not running
        self._safe_update(self.run_btn, self.stop_btn)


    # Save, Run, Stop, Reset

    def _validate_config_before_run(self):
        """Validate config before run. Returns (ok: bool, error_msg: str)."""
        try:
            port_val = int(self.stream_port_tf.value) if self.stream_port_tf.value else DEFAULT_PORT
            if not (1 <= port_val <= 65535):
                return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        try:
            crf_val = int(self.crf_tf.value) if self.crf_tf.value else DEFAULTS["CRF"]
            if not (0 <= crf_val <= 51):
                return False, UI_MESSAGES[self.locale]["err_crf"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_crf"]
        try:
            delay_val = float(self.audio_delay_tf.value) if self.audio_delay_tf.value else DEFAULTS["Audio Delay"]
            if not (-10 <= delay_val <= 10):
                return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        sk = self.stream_key_tf.value or "live"
        if not re.match(r'^[A-Za-z0-9_-]+$', sk) or len(sk) > 64:
            return False, UI_MESSAGES[self.locale]["err_stream_key"]
        if self.capture_mode_key == "Window":
            if not self.selected_window_name:
                return False, UI_MESSAGES[self.locale]["Please select a window before running in Window capture mode"]
            windows = list_windows()
            exists = any(
                (w.get("handle") is not None and w["handle"] == self.selected_window_handle)
                or (w.get("handle") is None and w["title"] == self.selected_window_name)
                for w in windows
            )
            if not exists:
                return False, UI_MESSAGES[self.locale]["The selected window no longer exists. Please refresh and select a valid window."]
        return True, ""

    def save_and_run(self, e):
        """Collect config, save YAML, start subprocess."""
        # Re-entry guard: prevent multiple clicks from launching duplicate processes
        if self._starting or (self.process and self.process.returncode is None):
            self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
            self.page.update()
            return
        ok, err = self._validate_config_before_run()
        if not ok:
            self.set_status(err)
            return
        self._starting = True
        self._cancel_starting = False
        self._esc_stopped = False
        self._stopping = False
        _set_console_quick_edit(False)
        self._set_running_ui(True)

        # Build config
        self._collect_config()

        # Save YAML
        ok, err = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))
            self._starting = False
            self._set_running_ui(False)
            return

        self.set_status(UI_MESSAGES[self.locale]["Countdown"], key="Countdown")
        self.page.update()

        # Start process with countdown
        asyncio.create_task(self._countdown_and_run(0.5))

    def on_advanced_stereo_change(self, e):
        self._sync_advanced_stereo_visibility()
        self._fit_window_to_content()
        try:
            self.page.update()
        except Exception:
            pass

    def _sync_advanced_stereo_visibility(self):
        visible = bool(getattr(self, "advanced_stereo_cb", None) and self.advanced_stereo_cb.value)
        for row in getattr(self, "_advanced_stereo_rows", []):
            row.visible = visible

    def on_depth_quick_change(self, value):
        quick = value if isinstance(value, str) else getattr(getattr(value, "control", None), "value", self.depth_quick_dd.value)
        strength = self._depth_strength_for_quick(self._display_to_depth_quick(quick))
        self.depth_strength_dd.value = f"{strength:.1f}"
        try:
            self.depth_strength_dd.update()
        except Exception:
            pass
        self._schedule_stereo_hot_save()

    @staticmethod
    def _depth_strength_for_quick(value):
        mapping = {
            "Soft": 1.4,
            "Standard": 2.0,
            "Enhanced": 2.6,
        }
        return mapping.get(value, 2.0)

    def _stereo_quality_options(self):
        return stereo_quality_options(self.locale)

    def _stereo_quality_to_display(self, value):
        return stereo_quality_to_display(value, self.locale)

    def _display_to_stereo_quality(self, value):
        return display_to_stereo_quality(value)
    def _depth_quick_to_display(self, value):
        return UI_MESSAGES[self.locale].get(value, value)

    @staticmethod
    def _display_to_depth_quick(value):
        mapping = {
            "Soft": "Soft",
            "Standard": "Standard",
            "Enhanced": "Enhanced",
            "柔和": "Soft",
            "标准": "Standard",
            "增强": "Enhanced",
        }
        return mapping.get(value, str(value or "Standard"))

    @staticmethod
    def _depth_quick_from_strength(value):
        try:
            strength = float(value)
        except (TypeError, ValueError):
            return "Standard"
        if strength < 1.7:
            return "Soft"
        if strength > 2.3:
            return "Enhanced"
        return "Standard"

    @staticmethod
    def _display_to_preset(value):
        mapping = {
            "Cinema": "cinema",
            "影院": "cinema",
            "电影 / 偏均衡": "cinema",
            "Game / Low Latency": "game_low_latency",
            "游戏 / 低延迟": "game_low_latency",
            "Still Image / HQ": "still_image_hq",
            "图片 / 高质量": "still_image_hq",
            "Debug / Export": "debug_export",
            "调试 / 导出": "debug_export",
        }
        return mapping.get(value, str(value or "cinema").strip().lower())

    def _preset_to_display(self, value):
        mapping = {
            "auto": "Cinema",
            "cinema": "Cinema",
            "game_low_latency": "Game / Low Latency",
            "still_image_hq": "Still Image / HQ",
            "debug_export": "Debug / Export",
        }
        key = mapping.get(str(value or "cinema").strip().lower(), "Cinema")
        return UI_MESSAGES[self.locale].get(key, key) if hasattr(self, "locale") else key

    @staticmethod
    def _clamp_foreground_scale(value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            value = float(DEFAULTS["Foreground Scale"])
        return max(-0.9, min(5.0, value))
    @staticmethod
    def _parse_int(val, default):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_float(val, default):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _collect_config(self):
        """Read all UI control values into self._config (single source of truth)."""
        if self.capture_mode_key == "Window":
            window_rect = getattr(self, 'selected_window_rect', None)
            if window_rect:
                center_x = window_rect[0] + window_rect[2] // 2
                center_y = window_rect[1] + window_rect[3] // 2
                monitor_idx = get_monitor_index_for_point(center_x, center_y)
            else:
                monitor_idx = get_primary_monitor_index()
        else:
            monitor_idx = self.monitor_label_to_index.get(self.monitor_dd.value, DEFAULTS["Monitor Index"])
        stereo_val = self.stereo_monitor_dd.value
        if stereo_val == "Viewer Window" or not stereo_val:
            stereo_idx = None
        else:
            stereo_idx = self.monitor_label_to_index.get(stereo_val, None)

        temporal_strength = self._parse_float(self.temporal_strength_dd.value, DEFAULTS["Temporal Strength"])
        scene_reset_threshold = self._parse_float(self.scene_reset_dd.value, DEFAULTS["Scene Reset Threshold"])
        foreground_scale = self._clamp_foreground_scale(self._parse_float(self.foreground_scale_dd.value, DEFAULTS["Foreground Scale"]))

        self._config.update({
            "Capture Mode": self.capture_mode_key,
            "Monitor Index": monitor_idx,
            "Window Title": self.selected_window_name if self.capture_mode_key == "Window" else "",
            "Show FPS": self.showfps_cb.value,
            "Stereo Preset": self._display_to_preset(self.stereo_preset_dd.value),
            "Stereo Quality": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "Synthetic View": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "IPD": self._parse_int(self.ipd_dd.value, int(DEFAULTS["IPD"] * 1000)) / 1000.0,
            "Stereo Scale": self._parse_float(self.stereo_scale_dd.value, DEFAULTS["Stereo Scale"]),
            "Convergence": self._parse_float(self.convergence_dd.value, DEFAULTS["Convergence"]),
            "Display Mode": self.display_mode_dd.value,
            "Model List": ALL_MODELS,
            "Depth Model": self.current_model_name,
            "Depth Strength": self._parse_float(self.depth_strength_dd.value, DEFAULTS["Depth Strength"]),
            "Depth Quick": self._display_to_depth_quick(self.depth_quick_dd.value),
            "Anti-aliasing": self._parse_int(self.antialiasing_dd.value, DEFAULTS["Anti-aliasing"]),
            "Depth Antialias Strength": self._parse_float(self.antialiasing_dd.value, DEFAULTS["Depth Antialias Strength"]),
            "Stereo Preset": self._display_to_preset(self.stereo_preset_dd.value),
            "Stereo Quality": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "Synthetic View": self._display_to_stereo_quality(self.stereo_quality_dd.value),
            "Max Shift Ratio": self._parse_float(self.max_shift_dd.value, DEFAULTS["Max Shift Ratio"]),
            "Temporal": temporal_strength > 0.0,
            "Temporal Strength": temporal_strength,
            "Auto Scene Reset": scene_reset_threshold > 0.0,
            "Scene Reset Threshold": scene_reset_threshold,
            "Reset Cooldown Frames": self._parse_int(self.reset_cooldown_dd.value, DEFAULTS["Reset Cooldown Frames"]),
            "Edge Dilation": self._parse_int(self.edge_dilation_dd.value, DEFAULTS["Edge Dilation"]),
            "Edge Threshold": self._parse_float(self.edge_threshold_dd.value, DEFAULTS["Edge Threshold"]),
            "Cross Eyed": self.cross_eyed_cb.value,
            "Anaglyph Method": self.anaglyph_dd.value,
            "Foreground Scale": foreground_scale,
            "Depth Resolution": self._parse_int(self.depth_res_dd.value, DEFAULTS["Depth Resolution"]),
            "FP16": self.fp16_cb.value,
            "Computing Device": self.device_label_to_index.get(self.device_dd.value, DEFAULTS["Computing Device"]),
            "Language": self.locale,
            "Run Mode": self.run_mode_key,
            "XR Preview Window": self.xr_preview_cb.value,
            "Local VSync": self.local_vsync_cb.value,
            "Target FPS": self._target_fps_from_display(self.target_fps_dd.value),
            "Processing Resolution": self._config.get("Processing Resolution", DEFAULTS["Processing Resolution"]),
            "Upscaler": self._upscaler_from_display(self.upscaler_dd.value),
            "Upscaler Sharpness": self._parse_float(self.upscaler_sharpness_dd.value, DEFAULTS["Upscaler Sharpness"]),
            "Stream Protocol": self.stream_proto_dd.value,
            "Streamer Port": self._parse_int(self.stream_port_tf.value, DEFAULTS["Streamer Port"]),
            "Stream Quality": self._parse_int(self.stream_quality_dd.value, DEFAULTS["Stream Quality"]),
            "torch.compile": self.torch_compile_cb.value,
            "TensorRT": self.tensorrt_cb.value,
            "Recompile TensorRT": self.recompile_trt_cb.value,
            "CoreML": self.coreml_cb.value,
            "Recompile CoreML": self.recompile_coreml_cb.value,
            "OpenVINO": self.openvino_cb.value,
            "Recompile OpenVINO": self.recompile_openvino_cb.value,
            "Capture Tool": self.capture_tool_dd.value,
            "Fill 16:9": self.fill_16_9_cb.value,
            "Fix Viewer Aspect": self.fix_aspect_cb.value,
            "Lossless Scaling Support": self.lossless_cb.value,
            "Stream Key": self.stream_key_tf.value,
            "Stereo Mix": self.audio_dd.value,
            "CRF": self._parse_int(self.crf_tf.value, DEFAULTS["CRF"]),
            "Audio Delay": self._parse_float(self.audio_delay_tf.value, DEFAULTS["Audio Delay"]),
            "Stereo Output": stereo_idx,
            "Controller Model": self.ctrl_model_dd.value,
            "Environment Model": self.env_model_dd.value,
        })
        self.recompile_trt_cb.value = False
        self.recompile_coreml_cb.value = False
        self.recompile_openvino_cb.value = False

    async def _countdown_and_run(self, seconds):
        self._diag("_countdown_and_run scheduled")
        try:
            if self.process and self.process.returncode is None:
                self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
                self._diag("already running, return")
                return
            if seconds > 0:
                await asyncio.sleep(seconds)
            if self._cancel_starting:
                self._cancel_starting = False
                self._diag("cancelled, return")
                return
            print(f"[Main] Initializing Desktop2Stereo {self.run_mode_key}…")
            shutdown_event.clear()
            if OS_NAME == "Windows":
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, os.path.join(BASE_DIR, "main.py"),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, os.path.join(BASE_DIR, "main.py"),
                    start_new_session=True,
                )
            self._diag(f"process started, pid={self.process.pid}")
            self.set_status(UI_MESSAGES[self.locale]["Running"], key="Running")
            self.page.update()
            asyncio.create_task(self._monitor_process_task())
            self._diag("monitor_task created")
            # Reset recompile flags after 8s
            for _ in range(8):
                await asyncio.sleep(1)
                if self.process and self.process.returncode is not None:
                    self._diag(f"process exited during wait, code={self.process.returncode}")
                    break
            self._config["Recompile TensorRT"] = False
            self._config["Recompile CoreML"] = False
            self._config["Recompile OpenVINO"] = False
            self._config["FP16"] = DEFAULTS["FP16"]
            self._config["Stereo Preset"] = "cinema"
            save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        except Exception as e:
            self.set_status(UI_MESSAGES[self.locale]["err_start_failed"].format(e))
            self.page.update()
        finally:
            self._starting = False

    def _diag(self, msg):
        """Write diagnostic log to file."""
        import datetime
        os.makedirs(os.path.dirname(DIAG_LOG), exist_ok=True)
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [diag] {msg}"
        with open(DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def _monitor_process_task(self):
        """Wait for process to exit, then update status."""
        proc = self.process
        if not proc:
            self._diag("monitor_task: proc is None, return")
            return
        self._diag(f"monitor_task started, pid={proc.pid}")
        try:
            await proc.wait()
            self._diag(f"proc.wait returned, rc={proc.returncode}")
        except Exception as e:
            self._diag(f"proc.wait() exception: {e}")
        finally:
            self._diag(f"finally: process is proc={self.process is proc}, returncode={proc.returncode}")
            if self.process is proc:
                self.process = None
            self._starting = False
            code = proc.returncode if proc else None
            if code and code != 0:
                self.set_status(UI_MESSAGES[self.locale]["exited_with_code"].format(code))
            else:
                self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
            _set_console_quick_edit(True)
            self._set_running_ui(False)
            self._diag("monitor_task done, status updated")

    def stop_process(self, e=None):
        """Stop subprocess (called from UI button)."""
        future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    async def _on_page_close(self, e=None):
        """Stop subprocess when user closes the window. Settings are only saved on Run."""
        self._closed = True
        if hasattr(self, '_esc_task') and self._esc_task and not self._esc_task.done():
            self._esc_task.cancel()
        await self._async_stop()

    async def _async_stop(self):
        """Actual stop logic — runs in async context."""
        # Prevent reentry: skip if already stopping
        if self._stopping:
            return
        self._stopping = True

        # Suppress ESC repeat triggers + cancel pending start
        self._esc_stopped = True
        self._esc_down = None
        self._cancel_starting = True

        if self._proc_lock is not None:
            shutdown_event.set()
            saved_pid = None
            proc = None
            async with self._proc_lock:
                proc = self.process
                if proc and proc.returncode is None:
                    saved_pid = proc.pid
                    proc.terminate()
                self.process = None

            if saved_pid and proc:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass
                except Exception:
                    pass
                # Cleanup RTMP process tree
                if self.run_mode_key == "RTMP Streamer":
                    try:
                        if OS_NAME == "Windows":
                            p = await asyncio.create_subprocess_exec(
                                'taskkill', '/f', '/t', '/pid', str(saved_pid),
                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                            await p.wait()
                        else:
                            import signal
                            try:
                                os.killpg(os.getpgid(saved_pid), signal.SIGTERM)
                            except Exception:
                                pass
                    except Exception:
                        pass
        print("[Main] Stopped")
        self._starting = False
        self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
        _set_console_quick_edit(True)
        if not self._closed:
            self._set_running_ui(False)

    def reset_defaults(self, e):
        """Reset to defaults, preserve locale & device."""
        current_locale = self.locale
        current_device_label = self.device_dd.value
        current_device_idx = self.device_label_to_index.get(current_device_label, DEFAULTS["Computing Device"])
        current_primary = get_primary_monitor_index()
        dynamic_defaults = DEFAULTS.copy()
        dynamic_defaults["Monitor Index"] = current_primary
        self.apply_config(dynamic_defaults, keep_optional=False)

        # Restore locale & device
        self.locale = current_locale
        self.lang_dd.value = "English" if current_locale == "EN" else "简体中文"
        self.device_dd.value = current_device_label
        self._config["Language"] = current_locale
        self._config["Computing Device"] = current_device_idx

        self.update_ui_texts()
        self._sync_visibility()
        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.update()

	
    # ESC long-press monitoring

    VK_ESC = 0x1B

    async def _esc_poll_task(self):
        """Background polling for ESC long-press (3 seconds).

        Uses Windows GetAsyncKeyState to query key state:
        - No global hook, won't trigger antivirus
        - No external dependencies (ctypes only, Python built-in)
        - Single asyncio task, no extra threads
        - Can still detect when unfocused, but no hook registration
        - High-frequency polling only while process is running; low-frequency when idle
        """
        if OS_NAME != "Windows":
            return  # Non-Windows: rely on Flet on_keyboard_event only

        user32 = ctypes.windll.user32
        try:
            while not self._closed:
                await asyncio.sleep(0.2)  # 200ms polling for faster ESC response
                if self._closed:
                    break
                # High bit (0x8000) = key is currently pressed
                if user32.GetAsyncKeyState(self.VK_ESC) & 0x8000:
                    if self._esc_down is None:
                        self._esc_down = time.time()
                    elif not self._esc_stopped and (time.time() - self._esc_down >= 3.0):
                        self._esc_stopped = True
                        self._esc_down = None
                        self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                        asyncio.ensure_future(self._async_stop())
                else:
                    # Reset on release so it can re-trigger next time
                    if self._esc_down is not None:
                        self._esc_down = None
                        self._esc_stopped = False
        except asyncio.CancelledError:
            pass  # Normal cancellation on window close

    def _on_key(self, e: ft.KeyboardEvent):
        """Flet keyboard event (auxiliary, only used on macOS/Linux when window is focused)."""
        if e.key != "Esc" or self._esc_stopped or OS_NAME == "Windows":
            return
        # macOS/Linux fallback: start a short-lived monitor task on first press
        now = time.time()
        if self._esc_down is None:
            self._esc_down = now
            # Create a short-lived monitor task, not dependent on key repeat events
            asyncio.create_task(self._esc_watch_task())
        elif now - self._esc_down >= 3.0:
            self._esc_stopped = True
            self._esc_down = None
            self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
            asyncio.ensure_future(self._async_stop())

    async def _esc_watch_task(self):
        """Non-Windows platforms: monitor 3-second timeout after first ESC press."""
        try:
            for _ in range(60):  # 60 × 0.05 = 3s
                await asyncio.sleep(0.05)
                if self._esc_down is None or self._esc_stopped or self._closed:
                    return
                if time.time() - self._esc_down >= 3.0:
                    self._esc_stopped = True
                    self._esc_down = None
                    self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                    asyncio.ensure_future(self._async_stop())
                    return
        except asyncio.CancelledError:
            pass


# Entry point
def main():
    ft.run(_async_main)


async def _async_main(page: ft.Page):
    app = Desktop2StereoGUI(page)
    await app.setup()


if __name__ == "__main__":
    main()

