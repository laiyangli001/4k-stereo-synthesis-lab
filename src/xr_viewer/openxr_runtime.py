from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
from typing import Callable


def _noop(*args, **kwargs):
    return None


@dataclass
class OpenXRRuntimeConfig:
    depth_strength: float
    convergence: float
    fps: int
    show_fps: bool
    controller_model: str
    environment_model: str
    screen_width: float
    screen_distance: float
    show_preview_window: bool
    capture_mode: str
    monitor_index: int
    frame_size: tuple[int, int] = (1280, 720)


@dataclass
class OpenXRRuntimeCallbacks:
    update_runtime_config: Callable
    render_active_set: Callable
    render_active_clear: Callable
    source_active_set: Callable
    wait_idle_clear: Callable
    bootstrap_done_set: Callable
    breakdown_inc: Callable = _noop
    breakdown_add_time: Callable = _noop
    breakdown_add_value: Callable = _noop
    breakdown_set_latest: Callable = _noop
    render_active_event: object | None = None
    source_active_event: object | None = None
    idle_active_event: object | None = None


def use_environment_viewer(environment_model):
    env_name = str(environment_model or "").strip()
    return bool(env_name) and env_name.lower() != "none"


def frame_size_from_eye(first_eye):
    import torch

    if isinstance(first_eye, torch.Tensor):
        if first_eye.ndim == 4:
            return first_eye.shape[3], first_eye.shape[2]
        return first_eye.shape[2], first_eye.shape[1]
    return first_eye.shape[1], first_eye.shape[0]


def frame_size_from_runtime_result(runtime_result):
    display_size = getattr(runtime_result, "output_display_size", None)
    if display_size is None:
        debug = getattr(runtime_result, "debug_info", None) or {}
        display_size = _parse_size_text(debug.get("runtime_output_display_size"))
    if display_size is not None:
        return display_size
    return frame_size_from_eye(runtime_result.left_eye)


def _parse_size_text(value):
    if value is None:
        return None
    parts = str(value).strip().lower().split("x", 1)
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def load_openxr_viewer(environment_model):
    env_name = str(environment_model or "").strip()
    use_env = use_environment_viewer(env_name)
    print(
        "[OpenXRViewer] Viewer selection: "
        f"environment_model={env_name!r} use_environment={use_env}",
        flush=True,
    )
    if use_env:
        from xr_viewer.environment import OPENXR_AVAILABLE, OpenXRViewer
    else:
        from xr_viewer.base import OPENXR_AVAILABLE, OpenXRViewer
    if not OPENXR_AVAILABLE:
        raise ImportError("pyopenxr not installed -run: pip install pyopenxr")
    return OpenXRViewer


def run_openxr_mode(runtime_q, config: OpenXRRuntimeConfig, callbacks: OpenXRRuntimeCallbacks):
    OpenXRViewer = load_openxr_viewer(config.environment_model)
    runtime_result = None
    capture_start_time = None
    runtime_result, capture_start_time = runtime_q.get()
    callbacks.breakdown_inc("viewer_get")
    width, height = frame_size_from_runtime_result(runtime_result)
    try:
        viewer = OpenXRViewer(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            frame_size=(width, height),
            fps=config.fps,
            depth_q=runtime_q,
            show_fps=config.show_fps,
            controller_model=config.controller_model,
            environment_model=config.environment_model,
            breath_enabled=False,
            openxr_screen_width=config.screen_width,
            openxr_screen_distance=config.screen_distance,
            show_preview_window=config.show_preview_window,
            capture_mode=config.capture_mode,
            monitor_index=config.monitor_index,
            render_active_event=callbacks.render_active_event,
            source_active_event=callbacks.source_active_event,
            idle_active_event=callbacks.idle_active_event,
            runtime_config_callback=callbacks.update_runtime_config,
        )
        viewer._fps_breakdown_inc = callbacks.breakdown_inc
        viewer._fps_breakdown_add_time = callbacks.breakdown_add_time
        viewer._fps_breakdown_add_value = callbacks.breakdown_add_value
        callbacks.breakdown_set_latest(
            "openxr_async_effects_enabled",
            bool(getattr(viewer, "_openxr_async_effects_enabled", True)),
        )
        callbacks.source_active_set()
        callbacks.render_active_clear()
        callbacks.wait_idle_clear()
        callbacks.bootstrap_done_set()
        viewer.run(first_runtime_result=runtime_result, first_frame_ts=capture_start_time)
        return viewer
    except Exception as exc:
        print(f"[Main] OpenXR Link error: {exc}")
        return None
