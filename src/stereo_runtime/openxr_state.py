from __future__ import annotations

import threading

from stereo_runtime import OpenXRRenderConfig


class OpenXRStateController:
    def __init__(
        self,
        *,
        run_mode: str,
        ipd: float,
        depth_ratio: float | None = None,
        depth_strength: float | None = None,
        convergence: float = 0.0,
        stereo_scale: float | None = None,
        max_shift_ratio: float | None = None,
    ):
        self.run_mode = run_mode
        self.render_active = threading.Event()
        self.source_active = threading.Event()
        self.wait_idle_active = threading.Event()
        self.bootstrap_done = threading.Event()
        self.runtime_config_lock = threading.Lock()
        if depth_ratio is None:
            depth_ratio = depth_strength
        if depth_ratio is None:
            raise TypeError("OpenXRStateController requires depth_ratio or depth_strength")
        self.runtime_config_state = {
            "ipd": float(ipd),
            "depth_ratio": float(depth_ratio),
            "convergence": float(convergence),
            "stereo_scale": None if stereo_scale is None else float(stereo_scale),
            "max_shift_ratio": None if max_shift_ratio is None else float(max_shift_ratio),
            "screen_roll": 0.0,
        }
        self.source_pause_notice_lock = threading.Lock()
        self.source_pause_noticed = None
        self.wait_idle_notice_lock = threading.Lock()
        self.wait_idle_noticed = None

    def source_paused(self) -> bool:
        paused = (
            self.run_mode == "OpenXR"
            and self.bootstrap_done.is_set()
            and not self.source_active.is_set()
        )
        with self.source_pause_notice_lock:
            if self.source_pause_noticed is not paused:
                self.source_pause_noticed = paused
                if paused:
                    print("[Main] OpenXR source inference paused")
                else:
                    print("[Main] OpenXR source inference resumed")
        return paused

    def hard_idle_active(self, on_enter=None) -> bool:
        idle = (
            self.run_mode == "OpenXR"
            and self.bootstrap_done.is_set()
            and self.wait_idle_active.is_set()
        )
        with self.wait_idle_notice_lock:
            if self.wait_idle_noticed is not idle:
                self.wait_idle_noticed = idle
                if idle:
                    if on_enter is not None:
                        on_enter()
                    print("[Main] OpenXR hard idle entered")
                else:
                    print("[Main] OpenXR hard idle exited")
        return idle

    def update_runtime_config(
        self,
        *,
        ipd=None,
        depth_ratio=None,
        depth_strength=None,
        convergence=None,
        stereo_scale=None,
        max_shift_ratio=None,
        screen_roll=None,
    ) -> None:
        with self.runtime_config_lock:
            if ipd is not None:
                self.runtime_config_state["ipd"] = float(ipd)
            if depth_ratio is None:
                depth_ratio = depth_strength
            if depth_ratio is not None:
                self.runtime_config_state["depth_ratio"] = float(depth_ratio)
            if convergence is not None:
                self.runtime_config_state["convergence"] = float(convergence)
            if stereo_scale is not None:
                self.runtime_config_state["stereo_scale"] = float(stereo_scale)
            if max_shift_ratio is not None:
                self.runtime_config_state["max_shift_ratio"] = float(max_shift_ratio)
            if screen_roll is not None:
                self.runtime_config_state["screen_roll"] = float(screen_roll)

    def current_render_config(self, runtime) -> OpenXRRenderConfig:
        with self.runtime_config_lock:
            state = dict(self.runtime_config_state)
        return OpenXRRenderConfig(
            ipd=state["ipd"],
            ipd_mm=runtime.stereo_config.ipd_mm,
            stereo_scale=(
                runtime.stereo_config.stereo_scale
                if state["stereo_scale"] is None
                else state["stereo_scale"]
            ),
            depth_strength=state["depth_ratio"],
            convergence=state["convergence"],
            max_shift_ratio=(
                runtime.stereo_config.max_shift_ratio
                if state["max_shift_ratio"] is None
                else state["max_shift_ratio"]
            ),
            screen_roll=state["screen_roll"],
        )
