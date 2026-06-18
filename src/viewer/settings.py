from __future__ import annotations

from dataclasses import dataclass

from utils.display import compute_output_resolution, get_fps
from viewer.upscaler import normalize_upscaler, normalize_upscaler_sharpness


@dataclass(frozen=True)
class ViewerSettings:
    monitor_index: int
    display_mode: str
    stereo_display_index: int | None
    stereo_display_selection: bool
    output_resolution: tuple[int, int]
    show_fps: bool
    depth_strength: float
    ipd: float
    convergence: float
    capture_mode: str
    window_title: str | None
    target_fps: int
    fps: int
    fill_16_9: bool
    local_vsync: bool
    upscaler: str
    upscaler_sharpness: float
    controller_model: str
    environment_model: str
    xr_preview_window: bool


def resolve_viewer_settings(settings: dict) -> ViewerSettings:
    monitor_index = settings["Monitor Index"]
    display_mode = settings["Display Mode"]
    stereo_display_index = settings.get("Stereo Output")
    stereo_display_selection = False if not stereo_display_index else True
    output_resolution = compute_output_resolution(
        settings.get("Processing Resolution", "Auto"),
        display_mode,
        monitor_index,
        stereo_display_index,
    )
    capture_mode = settings["Capture Mode"]
    window_title = settings["Window Title"] if capture_mode == "Window" else None
    target_fps = int(settings.get("Target FPS", 0) or 0)
    fps = target_fps if 1 <= target_fps <= 240 else get_fps(window_title, monitor_index)

    return ViewerSettings(
        monitor_index=monitor_index,
        display_mode=display_mode,
        stereo_display_index=stereo_display_index,
        stereo_display_selection=stereo_display_selection,
        output_resolution=output_resolution,
        show_fps=settings["Show FPS"],
        depth_strength=settings["Depth Strength"],
        ipd=settings["IPD"],
        convergence=settings["Convergence"],
        capture_mode=capture_mode,
        window_title=window_title,
        target_fps=target_fps,
        fps=fps,
        fill_16_9=settings["Fill 16:9"],
        local_vsync=settings.get("Local VSync", True),
        upscaler=normalize_upscaler(settings.get("Upscaler", "Off")),
        upscaler_sharpness=normalize_upscaler_sharpness(settings.get("Upscaler Sharpness", 0.35)),
        controller_model=settings["Controller Model"],
        environment_model=settings.get("Environment Model", "Default"),
        xr_preview_window=settings.get("XR Preview Window", True),
    )
