"""GUI Process Mixin — subprocess lifecycle, ESC monitoring, URL actions."""
import os
import re
import sys
import time
import asyncio
import ctypes
import subprocess
import traceback
import flet as ft
from utils import OS_NAME, DEFAULT_PORT, shutdown_event, read_yaml
from . import devices as devices_module
from .config import DEFAULTS, default_base_depth_model, save_yaml
from .paths import BASE_DIR, DIAG_LOG, LOG_DIR, LOG_FILE, STOP_REQUEST_FILE
from .capture_sources import get_primary_monitor_index, list_windows
from .localization import UI_MESSAGES

# ── module-level console helpers ──

_NOISY_CONSOLE_PREFIXES = (
    "[NativeUtil] sogou_native_util_pc loaded successfully",
    "[warmup] same version",
    "[INFO] [flet] Session was garbage collected:",
)
_DEBUG_CONSOLE_PREFIXES = (
    "[debug]",
    "debug:",
)
_ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES = (
    "asyncio.base_subprocess",
    "asyncio.proactor_events",
)
_ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES = (
    "Event loop is closed",
    "I/O operation on closed pipe",
)
_asyncio_shutdown_noise_filter_installed = False
_console_logging_installed = False


def _is_asyncio_shutdown_unraisable(unraisable):
    exc = getattr(unraisable, "exc_value", None)
    if str(exc) not in _ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES:
        return False
    obj = getattr(unraisable, "object", None)
    module = getattr(obj, "__module__", "")
    qualname = getattr(obj, "__qualname__", "")
    return module in _ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES and qualname.endswith(".__del__")


def _install_asyncio_shutdown_noise_filter():
    """Suppress known Windows asyncio transport __del__ noise during GUI shutdown."""
    global _asyncio_shutdown_noise_filter_installed
    if _asyncio_shutdown_noise_filter_installed or not hasattr(sys, "unraisablehook"):
        return
    previous_hook = sys.unraisablehook

    def _desktop2stereo_unraisable_hook(unraisable):
        if _is_asyncio_shutdown_unraisable(unraisable):
            return
        previous_hook(unraisable)

    sys.unraisablehook = _desktop2stereo_unraisable_hook
    _asyncio_shutdown_noise_filter_installed = True


def _is_key_console_output(data):
    text = str(data or "").strip()
    if not text:
        return True
    lower = text.lower()
    if any(text.startswith(prefix) for prefix in _NOISY_CONSOLE_PREFIXES):
        return False
    if any(lower.startswith(prefix) for prefix in _DEBUG_CONSOLE_PREFIXES):
        return False
    if lower.startswith("[diag]") and not any(token in lower for token in ("error", "failed", "exception", "exited")):
        return False
    return True


def _setup_console_logging():
    """Write full stdout/stderr to the rolling log and keep console concise."""
    global _console_logging_installed
    if _console_logging_installed:
        _install_asyncio_shutdown_noise_filter()
        return

    import datetime
    import threading
    from utils.logging_setup import configure_debug_file_logging

    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.isfile(path) and os.path.abspath(path) != os.path.abspath(LOG_FILE):
                try:
                    os.remove(path)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Desktop2Stereo log started {datetime.datetime.now().isoformat(timespec='seconds')} ===\n")
    except Exception:
        pass

    configure_debug_file_logging(LOG_FILE)
    _install_asyncio_shutdown_noise_filter()

    lock = threading.Lock()

    class _TeeStream:
        def __init__(self, original, label):
            self.original = original
            self.label = label

        def write(self, data):
            if data:
                try:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    with lock:
                        with open(LOG_FILE, "a", encoding="utf-8") as f:
                            for line in str(data).splitlines():
                                stripped = line.rstrip()
                                if stripped:
                                    f.write(f"[{ts}] [{self.label}] {stripped}\n")
                except Exception:
                    pass
            if _is_key_console_output(data):
                try:
                    self.original.write(data)
                except Exception:
                    pass
            return len(data or "")

        def flush(self):
            try:
                self.original.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return self.original.isatty()
            except Exception:
                return False

        def fileno(self):
            return self.original.fileno()

    sys.stdout = _TeeStream(sys.stdout, "out")
    sys.stderr = _TeeStream(sys.stderr, "err")
    _console_logging_installed = True


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
        pass


# Disable Quick Edit while the worker is running.
_set_console_quick_edit(False)


class GUIProcessMixin:
    """Mixin providing process lifecycle, ESC monitoring, and status for Desktop2StereoGUI."""

    def set_status(self, msg, key=None):
        self.status_text.value = msg
        if key is not None:
            self._status_key = key
        self._safe_update(self.status_text)

    def _set_running_ui(self, running: bool):
        self.run_btn.disabled = running
        self.stop_btn.disabled = not running
        self._safe_update(self.run_btn, self.stop_btn)

    def _diag(self, msg, error=False):
        import datetime
        os.makedirs(LOG_DIR, exist_ok=True)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [diag] {msg}\n")
        except Exception:
            pass
        if error:
            try:
                original = getattr(sys.stdout, "original", sys.stdout)
                original.write(f"[diag] {msg}\n")
                original.flush()
            except Exception:
                pass

    # ── save & run ──

    def _validate_config_before_run(self):
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
                for w in windows)
            if not exists:
                return False, UI_MESSAGES[self.locale]["The selected window no longer exists. Please refresh and select a valid window."]
        return True, ""

    def save_and_run(self, e):
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
        self._collect_config()
        ok, err = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))
            self._starting = False
            self._set_running_ui(False)
            return
        self.set_status(UI_MESSAGES[self.locale]["Countdown"], key="Countdown")
        self.page.update()
        asyncio.create_task(self._countdown_and_run(0.5))

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
            print(f"[Main] Initializing Desktop2Stereo {self.run_mode_key}...")
            shutdown_event.clear()
            try:
                if os.path.exists(STOP_REQUEST_FILE):
                    os.remove(STOP_REQUEST_FILE)
            except Exception:
                pass
            child_args = [
                sys.executable,
                "-u",
                "-X",
                "faulthandler",
                os.path.join(BASE_DIR, "main.py"),
            ]
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["D2S_FORCE_TQDM"] = "1"
            if OS_NAME == "Windows":
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    env=child_env,
                )
            else:
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
            self._diag(f"process started, pid={self.process.pid}, log={LOG_FILE}")
            asyncio.create_task(self._pump_child_output(self.process))
            self.set_status(UI_MESSAGES[self.locale]["Running"], key="Running")
            self.page.update()
            asyncio.create_task(self._monitor_process_task())
            self._diag("monitor_task created")
            for _ in range(8):
                await asyncio.sleep(1)
                if self.process and self.process.returncode is not None:
                    self._diag(f"process exited during wait, code={self.process.returncode}")
                    break
            self._config["Recompile TensorRT"] = False
            self._config["Recompile MIGraphX"] = False
            self._config["Recompile CoreML"] = False
            self._config["Recompile OpenVINO"] = False
            self._config["Stereo Preset"] = "cinema"
            save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        except Exception as e:
            self._diag(f"_countdown_and_run failed:\n{traceback.format_exc()}", error=True)
            self.set_status(UI_MESSAGES[self.locale]["err_start_failed"].format(e))
            self.page.update()
        finally:
            self._starting = False

    async def _pump_child_output(self, proc):
        try:
            stream = proc.stdout
            if stream is None:
                return
            while True:
                raw = await stream.read(4096)
                if not raw:
                    break
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    text = repr(raw)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
        except Exception as e:
            self._diag(f"_pump_child_output exception: {e}\n{traceback.format_exc()}", error=True)


    async def _monitor_process_task(self):
        proc = self.process
        if not proc:
            self._diag("monitor_task: proc is None, return")
            return
        self._diag(f"monitor_task started, pid={proc.pid}")
        try:
            await proc.wait()
            self._diag(f"proc.wait returned, rc={proc.returncode}")
        except Exception as e:
            self._diag(f"proc.wait() exception: {e}", error=True)
        finally:
            self._diag(f"finally: process is proc={self.process is proc}, returncode={proc.returncode}")
            if self.process is proc:
                self.process = None
            self._starting = False
            code = proc.returncode if proc else None
            if code and code != 0:
                self._diag(f"child exited rc={code}; see {LOG_FILE} for details", error=True)
                self.set_status(UI_MESSAGES[self.locale]["exited_with_code"].format(code))
            else:
                self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
            _set_console_quick_edit(True)
            self._set_running_ui(False)
            self._diag("monitor_task done, status updated")

    # ── stop ──

    def stop_process(self, e=None):
        future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    async def _on_page_close(self, e=None):
        self._closed = True
        if hasattr(self, '_esc_task') and self._esc_task and not self._esc_task.done():
            self._esc_task.cancel()
        await self._async_stop()

    async def _async_stop(self):
        if self._stopping:
            return
        self._stopping = True
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
                    try:
                        if OS_NAME == "Windows":
                            os.makedirs(LOG_DIR, exist_ok=True)
                            with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as f:
                                f.write(str(saved_pid))
                        else:
                            import signal
                            os.killpg(os.getpgid(saved_pid), signal.SIGINT)
                    except Exception:
                        self._diag(f"graceful stop failed:\n{traceback.format_exc()}", error=True)
                        try:
                            proc.terminate()
                        except Exception:
                            self._diag(f"proc.terminate() failed:\n{traceback.format_exc()}", error=True)
                self.process = None

            if saved_pid and proc:
                exited_cleanly = False
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1)
                    exited_cleanly = True
                except asyncio.TimeoutError:
                    exited_cleanly = False
                except Exception:
                    self._diag(f"proc.wait() exception:\n{traceback.format_exc()}", error=True)
                    exited_cleanly = True
                if not exited_cleanly:
                    try:
                        proc.kill()
                    except Exception:
                        pass
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

    # ── reset ──

    def reset_defaults(self, e):
        current_locale = self.locale
        current_device_label = self.device_dd.value
        current_device_idx = self.device_label_to_index.get(current_device_label, DEFAULTS["Computing Device"])
        current_primary = get_primary_monitor_index()
        is_nvidia_cuda = "CUDA" in (current_device_label or "") and not devices_module.IS_ROCM
        dynamic_defaults = DEFAULTS.copy()
        dynamic_defaults["Monitor Index"] = current_primary
        dynamic_defaults["Depth Model"] = default_base_depth_model()
        dynamic_defaults["XR Preview Window"] = False
        if is_nvidia_cuda:
            dynamic_defaults["torch.compile"] = True
            dynamic_defaults["TensorRT"] = True
        self.apply_config(dynamic_defaults, keep_optional=False)
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

    # ── URL actions ──

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
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=url, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run("pbcopy", input=url, text=True)
            self.set_status(UI_MESSAGES[self.locale]["url_copied"], key="url_copied")
            asyncio.create_task(self._fade_status(2.0))

    async def _fade_status(self, delay):
        await asyncio.sleep(delay)
        self.set_status("", key="")

    # ── ESC long-press monitoring ──

    VK_ESC = 0x1B

    async def _esc_poll_task(self):
        if OS_NAME != "Windows":
            return
        user32 = ctypes.windll.user32
        try:
            while not self._closed:
                await asyncio.sleep(0.2)
                if self._closed:
                    break
                if user32.GetAsyncKeyState(self.VK_ESC) & 0x8000:
                    if self._esc_down is None:
                        self._esc_down = time.time()
                    elif not self._esc_stopped and (time.time() - self._esc_down >= 3.0):
                        self._esc_stopped = True
                        self._esc_down = None
                        self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                        asyncio.ensure_future(self._async_stop())
                else:
                    if self._esc_down is not None:
                        self._esc_down = None
                        self._esc_stopped = False
        except asyncio.CancelledError:
            pass

    def _on_key(self, e: ft.KeyboardEvent):
        if e.key != "Esc" or self._esc_stopped or OS_NAME == "Windows":
            return
        now = time.time()
        if self._esc_down is None:
            self._esc_down = now
            asyncio.create_task(self._esc_watch_task())
        elif now - self._esc_down >= 3.0:
            self._esc_stopped = True
            self._esc_down = None
            self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
            asyncio.ensure_future(self._async_stop())

    async def _esc_watch_task(self):
        try:
            for _ in range(60):
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
