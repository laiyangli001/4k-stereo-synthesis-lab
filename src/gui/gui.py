"""Desktop2Stereo Flet GUI — main application class combining all mixins.

Mixins:
  GUIBuilderMixin  — UI construction, layout sizing, data population
  GUIHandlerMixin  — event handlers, visibility sync, i18n, audio, refresh
  GUIConfigMixin   — config read/write, stereo preset data, hot-param save
  GUIProcessMixin  — subprocess lifecycle, ESC monitoring, URL actions
"""
import os
import asyncio
import flet as ft
from utils import VERSION, OS_NAME, read_yaml
from .builders import GUIBuilderMixin
from .handlers import GUIHandlerMixin
from .config_mgr import GUIConfigMixin
from .process import GUIProcessMixin, _setup_console_logging
from .config import DEFAULTS
from .controls import S
from .paths import BASE_DIR
from .localization import UI_MESSAGES
from .flet_runtime import ensure_vendored_flet_view


class Desktop2StereoGUI(
    GUIBuilderMixin,
    GUIHandlerMixin,
    GUIConfigMixin,
    GUIProcessMixin,
):
    """Flet GUI for Desktop2Stereo — full equivalent of tk ConfigGUI."""
    def __init__(self, page: ft.Page):
        self.page = page
        self._loop = None
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
        self._proc_lock = None
        self.monitor_label_to_index = {}
        self.device_label_to_index = {}
        self._esc_down = None
        self._esc_stopped = False
        self._closed = False
        self._cancel_starting = False
        self._stopping = False
        self._labels_aligned = False
        self._status_key = ""
        self._local_ip_cache = "127.0.0.1"
        self._local_ip_task = None

    async def setup(self):
        _setup_console_logging()
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
                    self.set_status(UI_MESSAGES[self.locale]["Loaded settings.yaml at startup"],
                                    key="Loaded settings.yaml at startup")
            except Exception as e:
                self.apply_config(self._config)
                self.set_status(
                    f"{UI_MESSAGES[self.locale]['Failed to load settings.yaml:']} {e}")
        else:
            self.apply_config(self._config)

        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.on_keyboard_event = self._on_key
        self._esc_task = asyncio.ensure_future(self._esc_poll_task())
        self._fit_window_to_content(update=False)

        # Show window
        self.page.window.visible = True
        self.page.update()
        await asyncio.sleep(0)


def main():
    """Entry point for the GUI application."""
    _setup_console_logging()
    ensure_vendored_flet_view()
    ft.run(_async_main)


async def _async_main(page: ft.Page):
    app = Desktop2StereoGUI(page)
    await app.setup()


if __name__ == "__main__":
    main()
