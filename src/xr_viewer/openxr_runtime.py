from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class OpenXRRuntimeConfig:
    ipd: float
    depth_strength: float
    convergence: float
    fps: int
    show_fps: bool
    controller_model: str
    environment_model: str
    show_preview_window: bool
    capture_mode: str
    monitor_index: int


@dataclass
class OpenXRRuntimeCallbacks:
    update_runtime_config: Callable
    render_active_set: Callable
    render_active_clear: Callable
    source_active_set: Callable
    wait_idle_clear: Callable
    bootstrap_done_set: Callable


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
    if use_environment_viewer(environment_model):
        from xr_viewer.environment import OPENXR_AVAILABLE, OpenXRViewer
    else:
        from xr_viewer.base import OPENXR_AVAILABLE, OpenXRViewer
    if not OPENXR_AVAILABLE:
        raise ImportError("pyopenxr not installed -run: pip install pyopenxr")
    return OpenXRViewer


def run_openxr_mode(runtime_q, config: OpenXRRuntimeConfig, callbacks: OpenXRRuntimeCallbacks):
    OpenXRViewer = load_openxr_viewer(config.environment_model)
    runtime_result, capture_start_time = runtime_q.get()
    width, height = frame_size_from_runtime_result(runtime_result)
    try:
        viewer = OpenXRViewer(
            ipd=config.ipd,
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            frame_size=(width, height),
            fps=config.fps,
            depth_q=runtime_q,
            show_fps=config.show_fps,
            controller_model=config.controller_model,
            environment_model=config.environment_model,
            breath_enabled=False,
            show_preview_window=config.show_preview_window,
            capture_mode=config.capture_mode,
            monitor_index=config.monitor_index,
            render_active_event=None,
            source_active_event=None,
            idle_active_event=None,
            runtime_config_callback=callbacks.update_runtime_config,
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
