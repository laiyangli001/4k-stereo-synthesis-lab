from __future__ import annotations

import os
import threading


class StereoWarmupTracker:
    def __init__(self, runtime, *, run_mode: str, openxr_runtime_direct: bool):
        self.runtime = runtime
        self.run_mode = run_mode
        self.openxr_runtime_direct = openxr_runtime_direct
        self.lock = threading.Lock()
        self.keys = set()

    def key_for_frame(self, rgb_frame):
        config = self.runtime.stereo_config
        runtime_cfg = self.runtime.config
        shape = tuple(getattr(rgb_frame, "shape", ()))
        return (
            shape,
            str(getattr(rgb_frame, "dtype", "unknown")),
            str(getattr(rgb_frame, "device", "unknown")),
            config.backend,
            runtime_cfg.output_format,
            config.layers,
            config.hole_fill,
            config.edge_dilation,
        )

    def warmup_once_for_frame(self, rgb_frame) -> None:
        if self.run_mode == "OpenXR" and self.openxr_runtime_direct:
            return
        key = self.key_for_frame(rgb_frame)
        with self.lock:
            if key in self.keys:
                return
            self.keys.add(key)
        try:
            print(f"[Main] Stereo warmup start: key={key}", flush=True)
            self.runtime.warmup_stereo_kernels_for_frame(rgb_frame)
        except Exception as exc:
            print(f"[Main] Stereo warmup skipped: {type(exc).__name__}: {exc}", flush=True)


class StereoRuntimeLogger:
    def __init__(self, runtime, *, active_preset_getter):
        self.runtime = runtime
        self.active_preset_getter = active_preset_getter
        self.last_mode_state = None
        self.last_fused_state = None

    def log_mode(self, reason, decision=None, samples=None, motion=None) -> None:
        config = self.runtime.stereo_config
        runtime_cfg = self.runtime.config
        active_preset = self.active_preset_getter()
        preset = active_preset or runtime_cfg.stereo_preset or runtime_cfg.mode
        fused_candidate = (
            config.backend == "fast_plus"
            and config.output_format == "half_sbs"
            and str(os.environ.get("D2S_RUNTIME_OUTPUT_UINT8", "0") or "0").strip().lower()
            in {"1", "true", "yes", "on"}
            and str(os.environ.get("D2S_FAST_PLUS_FUSED", "1") or "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        parts = [
            f"[Main] Stereo mode {reason}:",
            f"preset={preset}",
            f"synthetic_view={config.backend}",
            f"quality_setting={runtime_cfg.stereo_quality}",
            f"output={config.output_format}",
            f"hole_fill={config.hole_fill}",
            f"temporal={config.temporal}",
            f"fast_plus_fused_candidate={int(fused_candidate)}",
            f"runtime_uint8={os.environ.get('D2S_RUNTIME_OUTPUT_UINT8', '0')}",
        ]
        if decision is not None:
            parts.append(f"decision={decision.preset}")
            parts.append(f"reason={decision.reason}")
        if motion is not None:
            parts.append(f"motion={float(motion):.3f}")
        if samples:
            parts.append(f"gpu3d={float(samples.get('gpu_3d_util', 0.0)):.2f}")
            parts.append(f"video={float(samples.get('gpu_video_decode_util', 0.0)):.2f}")
            parts.append(f"input={float(samples.get('input_activity', 0.0)):.2f}")
            parts.append(f"idle={float(samples.get('idle_seconds', 0.0)):.1f}s")
        if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            print(" ".join(parts), flush=True)

    def log_mode_once(self, reason="active") -> None:
        config = self.runtime.stereo_config
        runtime_cfg = self.runtime.config
        state = (
            self.active_preset_getter(),
            config.backend,
            runtime_cfg.stereo_quality,
            config.output_format,
            config.hole_fill,
            config.temporal,
        )
        if state == self.last_mode_state:
            return
        self.last_mode_state = state
        self.log_mode(reason)

    def log_fast_plus_fused_runtime_state(self, runtime_result) -> None:
        debug = getattr(runtime_result, "debug_info", None) or {}
        output_format = str(debug.get("runtime_output_format", "unknown"))
        if output_format == "openxr_eye_views":
            state = (
                str(debug.get("backend", "unknown")),
                output_format,
                str(debug.get("runtime_output_dtype", "unknown")),
                str(debug.get("runtime_output_eye_size", "unknown")),
            )
            if state == self.last_fused_state:
                return
            self.last_fused_state = state
            if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                print(
                    "[Main] Stereo runtime output:"
                    f" backend={state[0]}"
                    f" output={state[1]}"
                    f" dtype={state[2]}"
                    f" eye={state[3]}"
                    f" depth_strength={float(debug.get('openxr_depth_strength', 0.0)):.3f}"
                    f" stereo_scale={float(debug.get('openxr_stereo_scale', 0.0)):.3f}"
                    f" max_shift={float(debug.get('openxr_max_shift_ratio', 0.0)):.3f}"
                    f" convergence={float(debug.get('openxr_convergence', 0.0)):.3f}",
                    flush=True,
                )
            return

        state = (
            str(debug.get("backend", "unknown")),
            output_format,
            str(debug.get("runtime_output_dtype", "unknown")),
            str(debug.get("runtime_output_pack_backend", "n/a")),
            str(debug.get("fast_plus_fused_backend", "n/a")),
            str(debug.get("fast_plus_fused_skip", "n/a")),
            str(debug.get("fast_plus_fused_temporal_bypass", "n/a")),
        )
        if state == self.last_fused_state:
            return
        self.last_fused_state = state
        if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            print(
                "[Main] Stereo runtime output:"
                f" backend={state[0]}"
                f" output={state[1]}"
                f" dtype={state[2]}"
                f" pack={state[3]}"
                f" fast_plus_fused={state[4]}"
                f" fast_plus_fused_skip={state[5]}"
                f" fast_plus_fused_temporal_bypass={state[6]}",
                f" depth_strength={float(debug.get('openxr_depth_strength', 0.0)):.3f}"
                f" stereo_scale={float(debug.get('openxr_stereo_scale', 0.0)):.3f}"
                f" max_shift={float(debug.get('openxr_max_shift_ratio', 0.0)):.3f}"
                f" convergence={float(debug.get('openxr_convergence', 0.0)):.3f}",
                flush=True,
            )
