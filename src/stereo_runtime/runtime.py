from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from dataclasses import replace
import os
import time
from typing import Any

import torch

from .adapter import StereoRuntimeConfig, depth_provider_config_from_runtime, stereo_config_from_runtime
from .depth_postprocess import postprocess_depth
from .depth_provider import DepthProfileResult, create_depth_provider
from .openxr_render import OpenXRRenderConfig, render_openxr_stereo
from .synthesis import StereoResult, synthesize_stereo
from .temporal import TemporalState


@dataclass(frozen=True)
class DepthRuntimeResult:
    depth: torch.Tensor
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)


class DepthRuntime:
    """Persistent host-facing runtime for RGB frame -> depth only."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self._loaded = False
        self._active = True
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def set_inference_active(self, active: bool) -> None:
        self._active = bool(active)

    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = self.depth_config.backend
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        report["inference_active"] = self._active
        return report

    def predict_depth_frame(self, rgb_frame: torch.Tensor) -> DepthRuntimeResult:
        if not self._active:
            raise RuntimeError("DepthRuntime inference is paused")
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        total_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(total_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)
        return DepthRuntimeResult(depth=profile.depth, timing=timing, provider_info=self.provider_report())

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


@dataclass(frozen=True)
class StereoRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenXRRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    source_rgb: torch.Tensor | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)


def openxr_result_from_stereo_result(stereo_result: StereoRuntimeResult) -> OpenXRRuntimeResult:
    debug = dict(stereo_result.debug_info or {})
    left_eye = stereo_result.left_eye
    right_eye = stereo_result.right_eye
    display_size = _runtime_frame_size(left_eye)
    if debug.get("runtime_output_format") == "half_sbs" and _should_split_half_sbs_for_openxr(debug):
        split_eyes = _split_half_sbs_frame(stereo_result.sbs)
        if split_eyes is not None:
            left_eye, right_eye = split_eyes
            display_size = _runtime_frame_size(stereo_result.sbs)
            debug.setdefault("runtime_output_pack_backend", "split_half_sbs")

    debug["runtime_output_format"] = "openxr_full_synthesis_eyes"
    debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
    debug["runtime_output_eye_size"] = _runtime_eye_size(left_eye)
    debug["runtime_output_display_size"] = _runtime_size_text(display_size)
    debug.setdefault("runtime_output_pack_backend", "none")

    return OpenXRRuntimeResult(
        depth=stereo_result.depth,
        left_eye=left_eye,
        right_eye=right_eye,
        source_rgb=None,
        debug_info=debug,
        timing=dict(stereo_result.timing or {}),
        provider_info=dict(stereo_result.provider_info or {}),
    )


def _should_split_half_sbs_for_openxr(debug: dict[str, Any]) -> bool:
    fused_backend = str(debug.get("fast_plus_fused_backend", "") or "").strip().lower()
    sbs_backend = str(debug.get("sbs_backend", "") or "").strip().lower()
    if fused_backend and fused_backend not in {"not_used", "none", "n/a"}:
        return True
    return "fused_half_sbs" in sbs_backend


def _split_half_sbs_frame(frame: Any) -> tuple[Any, Any] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) < 2:
        return None
    if len(shape) == 4 and shape[1] in (1, 3, 4):
        width_dim = 3
    elif len(shape) == 4 and shape[-1] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        width_dim = 1
    else:
        width_dim = len(shape) - 1
    width = int(shape[width_dim])
    half_width = width // 2
    if half_width <= 0:
        return None

    left_slice = [slice(None)] * len(shape)
    right_slice = [slice(None)] * len(shape)
    left_slice[width_dim] = slice(0, half_width)
    right_slice[width_dim] = slice(half_width, half_width * 2)
    left = frame[tuple(left_slice)]
    right = frame[tuple(right_slice)]
    if hasattr(left, "contiguous"):
        left = left.contiguous()
    if hasattr(right, "contiguous"):
        right = right.contiguous()
    return left, right


class RollingRuntimeStats:
    def __init__(self, *, maxlen: int = 300) -> None:
        self.maxlen = int(max(1, maxlen))
        self._samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)
        self._memory_samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)

    @property
    def count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._memory_samples.clear()

    def update(self, timing: dict[str, float], memory: dict[str, float] | None = None) -> None:
        self._samples.append({key: float(value) for key, value in timing.items()})
        if memory:
            self._memory_samples.append({key: float(value) for key, value in memory.items()})

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "window": self.maxlen,
            "count": self.count,
            "stages": {},
            "fps": {},
            "memory": {},
        }
        if not self._samples:
            return report

        keys = sorted({key for sample in self._samples for key in sample})
        for key in keys:
            values = [sample[key] for sample in self._samples if key in sample]
            report["stages"][key] = _series_stats(values)

        total_values = [sample["total_ms"] for sample in self._samples if sample.get("total_ms", 0.0) > 0]
        if total_values:
            total_stats = _series_stats(total_values)
            report["fps"] = {
                "latest": 1000.0 / total_values[-1],
                "mean_from_mean_ms": 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0,
                "p90_from_p90_ms": 1000.0 / total_stats["p90"] if total_stats["p90"] > 0 else 0.0,
                "p99_from_p99_ms": 1000.0 / total_stats["p99"] if total_stats["p99"] > 0 else 0.0,
            }

        if self._memory_samples:
            memory_keys = sorted({key for sample in self._memory_samples for key in sample})
            for key in memory_keys:
                values = [sample[key] for sample in self._memory_samples if key in sample]
                report["memory"][key] = _series_stats(values)
        return report


def _series_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return {"count": 0.0, "latest": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": float(count),
        "latest": float(values[-1]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "median": _percentile_sorted(ordered, 0.50),
        "p90": _percentile_sorted(ordered, 0.90),
        "p99": _percentile_sorted(ordered, 0.99),
    }


def _percentile_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


class StereoRuntime:
    """Persistent host-facing runtime for RGB frame -> depth -> stereo output."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        temporal_state: TemporalState | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.stereo_config = stereo_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self.temporal_state = temporal_state if temporal_state is not None else TemporalState()
        self._openxr_depth_temporal: torch.Tensor | None = None
        self._openxr_rgb_depth_dumped = False
        self._loaded = False
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def reset_temporal(self) -> None:
        self.temporal_state.reset()

    def configure_stereo(self, stereo_config: Any, *, reset_temporal: bool = False) -> None:
        self.stereo_config = stereo_config
        if reset_temporal:
            self.temporal_state.reset_stereo()

    def warmup_stereo_kernels_for_frame(self, rgb_frame: torch.Tensor) -> None:
        """Compile stereo synthesis kernels for the actual runtime frame shape."""
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)
        device = rgb_frame.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        if str(os.environ.get("D2S_DISABLE_STEREO_WARMUP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            return
        if rgb_frame.ndim == 3:
            _, height, width = rgb_frame.shape
        else:
            _, _, height, width = rgb_frame.shape
        rgb = torch.zeros((1, 3, int(height), int(width)), device=device, dtype=torch.float32)
        depth = torch.linspace(0.0, 1.0, int(width), device=device, dtype=torch.float32).view(1, 1, 1, int(width)).expand(1, 1, int(height), int(width)).contiguous()
        base = self.stereo_config
        foreground_scales = {round(float(base.foreground_scale), 3), 0.0, -0.7, 0.5}
        antialias_values = {round(float(base.depth_antialias_strength), 3), 0.0, 2.0}
        configs = []
        for scale in sorted(foreground_scales):
            for antialias in sorted(antialias_values):
                configs.append(
                    replace(
                        base,
                        temporal=False,
                        foreground_scale=float(scale),
                        depth_antialias_strength=float(antialias),
                    )
                )
        start = time.perf_counter()
        seen = set()
        for config in configs:
            key = (config.backend, config.output_format, config.layers, config.hole_fill, config.edge_dilation, round(float(config.foreground_scale), 3), round(float(config.depth_antialias_strength), 3))
            if key in seen:
                continue
            seen.add(key)
            synthesize_stereo(rgb, depth, config, temporal_state=None)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[StereoRuntime] stereo kernel warmup complete: {len(seen)} configs {int(width)}x{int(height)} in {elapsed_ms:.1f}ms",
            flush=True,
        )
    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = self.depth_config.backend
        report["stereo_backend"] = self.stereo_config.backend
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        return report

    def process_rgb_frame(self, rgb_frame: torch.Tensor) -> StereoRuntimeResult:
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        depth_total_ms = (time.perf_counter() - depth_start) * 1000.0
        depth = profile.depth
        stereo_config = self.stereo_config

        synth_start = time.perf_counter()
        fused_sbs, fused_skip = self._try_fast_plus_fused_sbs(rgb_frame, depth, stereo_config)
        if fused_sbs is not None:
            stereo = StereoResult(
                left_eye=rgb_frame,
                right_eye=rgb_frame,
                sbs=fused_sbs,
                debug_info={
                    "backend": stereo_config.backend,
                    "sbs_backend": "triton_fast_plus_fused_half_sbs_uint8",
                    "fast_plus_fused_backend": "triton_half_sbs_uint8",
                    "fast_plus_fused_temporal_bypass": int(bool(stereo_config.temporal)),
                    "occlusion_mask_backend": "triton_fused_radius1",
                    "hole_fill_backend": "triton_fused_directional_4tap",
                },
            )
        else:
            stereo = synthesize_stereo(
                rgb_frame,
                depth,
                stereo_config,
                temporal_state=self.temporal_state,
            )
            stereo.debug_info.setdefault("fast_plus_fused_backend", "not_used")
            stereo.debug_info.setdefault("fast_plus_fused_skip", fused_skip)
        synthesis_ms = (time.perf_counter() - synth_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "synthesis_ms": float(synthesis_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(stereo.debug_info)
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_output_format"] = self.stereo_config.output_format
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        if memory:
            debug.update(memory)

        sbs = stereo.sbs
        pack_start = time.perf_counter()
        if _runtime_output_uint8_enabled() and sbs.is_floating_point():
            fused_uint8 = _try_make_runtime_uint8_sbs(stereo, self.stereo_config.output_format)
            if fused_uint8 is not None:
                sbs = fused_uint8
                debug["runtime_output_pack_backend"] = "triton_half_sbs_uint8"
            else:
                sbs = sbs.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
                debug["runtime_output_pack_backend"] = "torch_float_to_uint8"
            debug["runtime_output_dtype"] = "uint8"
        else:
            if sbs.dtype == torch.uint8:
                debug.setdefault("runtime_output_pack_backend", debug.get("fast_plus_fused_backend", "prepacked_uint8"))
                debug["runtime_output_dtype"] = "uint8"
            else:
                debug["runtime_output_dtype"] = str(sbs.dtype).replace("torch.", "")
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        timing["pack_ms"] = float(pack_ms)
        if total_ms >= float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "200") or "200"):
            print(
                "[StereoRuntime] slow frame:"
                f" total_ms={total_ms:.1f}"
                f" depth_total_ms={depth_total_ms:.1f}"
                f" depth_model_ms={float(profile.model_ms):.1f}"
                f" synthesis_ms={synthesis_ms:.1f}"
                f" pack_ms={pack_ms:.1f}"
                f" backend={debug.get('backend', stereo_config.backend)}"
                f" foreground_scale={stereo_config.foreground_scale:.3f}"
                f" antialias={stereo_config.depth_antialias_strength:.3f}"
                f" output_dtype={debug.get('runtime_output_dtype', sbs.dtype)}"
                f" pack_backend={debug.get('runtime_output_pack_backend', 'n/a')}"
                f" sbs_backend={debug.get('sbs_backend', 'n/a')}"
                f" fast_plus_fused={debug.get('fast_plus_fused_backend', 'n/a')}"
                f" fast_plus_skip={debug.get('fast_plus_fused_skip', 'n/a')}"
                f" stage_depth_shift={float(debug.get('depth_postprocess_shift_ms', 0.0)):.1f}"
                f" stage_warp={float(debug.get('warp_composite_ms', 0.0)):.1f}"
                f" stage_occ={float(debug.get('occlusion_ms', 0.0)):.1f}"
                f" stage_fill={float(debug.get('hole_fill_ms', 0.0)):.1f}"
                f" stage_refine={float(debug.get('refine_ms', 0.0)):.1f}",
                flush=True,
            )

        return StereoRuntimeResult(
            depth=depth,
            left_eye=stereo.left_eye,
            right_eye=stereo.right_eye,
            sbs=sbs,
            debug_info=debug,
            timing=timing,
            provider_info=self.provider_report(),
        )

    def process_openxr_frame(
        self,
        rgb_frame: torch.Tensor,
        openxr_config: OpenXRRenderConfig | None = None,
    ) -> OpenXRRuntimeResult:
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        depth_total_ms = (time.perf_counter() - depth_start) * 1000.0
        depth = profile.depth

        prewarp_eyes = _openxr_prewarp_eyes_enabled()
        openxr_render_ms = 0.0
        pack_ms = 0.0
        pack_backend = "none"
        source_rgb = rgb_frame
        raw_depth = depth
        if prewarp_eyes:
            render_start = time.perf_counter()
            openxr = render_openxr_stereo(rgb_frame, depth, openxr_config)
            openxr_render_ms = (time.perf_counter() - render_start) * 1000.0

            pack_start = time.perf_counter()
            left_eye = openxr.left_eye
            right_eye = openxr.right_eye
            if _openxr_runtime_output_uint8_enabled() and (left_eye.is_floating_point() or right_eye.is_floating_point()):
                left_eye = left_eye.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                right_eye = right_eye.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                pack_backend = "torch_float_eye_to_uint8"
            pack_ms = (time.perf_counter() - pack_start) * 1000.0
            output_format = "openxr_eye_views"
            render_backend = dict(openxr.debug_info)
        else:
            depth = self._prepare_openxr_rgb_depth(depth)
            self._maybe_dump_openxr_rgb_depth(source_rgb=source_rgb, raw_depth=raw_depth, prepared_depth=depth)
            left_eye = rgb_frame
            right_eye = rgb_frame
            output_format = "openxr_rgb_depth"
            render_backend = {"backend": "openxr_viewer_shader_dibr"}
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "openxr_render_ms": float(openxr_render_ms),
            "pack_ms": float(pack_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(render_backend)
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_output_format"] = output_format
        debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
        debug["runtime_output_eye_size"] = _runtime_eye_size(left_eye)
        debug["runtime_output_pack_backend"] = pack_backend
        if openxr_config is not None:
            debug["openxr_ipd"] = float(openxr_config.ipd)
            debug["openxr_depth_strength"] = float(openxr_config.depth_strength)
            debug["openxr_stereo_scale"] = float(openxr_config.stereo_scale)
            debug["openxr_max_shift_ratio"] = float(openxr_config.max_shift_ratio)
            debug["openxr_convergence"] = float(openxr_config.convergence)
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        if memory:
            debug.update(memory)
        if total_ms >= float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "120") or "120"):
            print(
                "[StereoRuntime] slow openxr frame:"
                f" total_ms={total_ms:.1f}"
                f" depth_total_ms={depth_total_ms:.1f}"
                f" depth_model_ms={float(profile.model_ms):.1f}"
                f" depth_postprocess_ms={float(profile.postprocess_ms):.1f}"
                f" openxr_render_ms={openxr_render_ms:.1f}"
                f" pack_ms={pack_ms:.1f}"
                f" output_dtype={debug.get('runtime_output_dtype', 'n/a')}"
                f" eye_size={debug.get('runtime_output_eye_size', 'n/a')}"
                f" render_backend={debug.get('backend', debug.get('openxr_backend', 'n/a'))}"
                f" depth_backend={debug.get('runtime_depth_backend', 'n/a')}",
                flush=True,
            )

        return OpenXRRuntimeResult(
            depth=depth,
            left_eye=left_eye,
            right_eye=right_eye,
            source_rgb=source_rgb,
            debug_info=debug,
            timing=timing,
            provider_info=self.provider_report(),
        )

    def _prepare_openxr_rgb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
        depth = _openxr_rgb_depth_percentile_normalize(depth, percentile=_openxr_rgb_depth_percentile())
        gamma = _openxr_rgb_depth_gamma()
        if abs(gamma - 1.0) > 1e-4:
            depth = depth.pow(gamma)
        depth = postprocess_depth(
            depth,
            foreground_scale=float(getattr(self.stereo_config, "foreground_scale", 0.0)),
            antialias_strength=float(getattr(self.stereo_config, "depth_antialias_strength", 0.0)),
        )
        return self._stabilize_openxr_rgb_depth(depth)

    def _maybe_dump_openxr_rgb_depth(
        self,
        *,
        source_rgb: torch.Tensor,
        raw_depth: torch.Tensor,
        prepared_depth: torch.Tensor,
    ) -> None:
        dump_dir = os.environ.get("D2S_OPENXR_RGB_DEPTH_DUMP_DIR", "").strip()
        if not dump_dir or self._openxr_rgb_depth_dumped:
            return
        try:
            from pathlib import Path

            from .io import save_depth, save_rgb

            out_dir = Path(dump_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(source_rgb.detach().float().clamp(0.0, 1.0).cpu(), out_dir / "source_rgb.png")
            save_depth(raw_depth.detach().float().clamp(0.0, 1.0).cpu(), out_dir / "raw_depth.png")
            save_depth(prepared_depth.detach().float().clamp(0.0, 1.0).cpu(), out_dir / "prepared_depth.png")
            print(f"[StereoRuntime] OpenXR rgb_depth dump saved: {out_dir}", flush=True)
            self._openxr_rgb_depth_dumped = True
        except Exception as exc:
            print(f"[StereoRuntime] OpenXR rgb_depth dump failed: {type(exc).__name__}: {exc}", flush=True)

    def _stabilize_openxr_rgb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        alpha = _openxr_rgb_depth_temporal_alpha()
        depth = depth.detach().contiguous().float()
        if alpha <= 0.0:
            self._openxr_depth_temporal = None
            return depth
        prev = self._openxr_depth_temporal
        if prev is None or prev.shape != depth.shape or prev.device != depth.device:
            self._openxr_depth_temporal = depth
            return depth
        out = prev.mul(alpha).add(depth, alpha=(1.0 - alpha))
        self._openxr_depth_temporal = out.detach()
        return out

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)


    def _try_fast_plus_fused_sbs(self, rgb_frame: torch.Tensor, depth: torch.Tensor, stereo_config: Any) -> tuple[torch.Tensor | None, str]:
        if not _fast_plus_fused_enabled():
            return None, "disabled"
        if stereo_config.backend != "fast_plus":
            return None, f"backend={stereo_config.backend}"
        if stereo_config.output_format != "half_sbs":
            return None, f"format={stereo_config.output_format}"
        if not _runtime_output_uint8_enabled():
            return None, "runtime_uint8_off"
        if bool(getattr(stereo_config, "cross_eyed", False)):
            return None, "cross_eyed"
        if bool(getattr(stereo_config, "debug_output", False)):
            return None, "debug_output"
        try:
            from .fast_plus_fused_triton import can_use_fast_plus_fused_half_sbs_uint8, make_fast_plus_fused_half_sbs_uint8
            from .output import match_depth
        except Exception as exc:
            return None, f"import_failed:{type(exc).__name__}"
        depth = match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1])
        if not can_use_fast_plus_fused_half_sbs_uint8(rgb_frame, depth):
            return None, f"unsupported_tensor:rgb={tuple(rgb_frame.shape)}/{rgb_frame.dtype}/{rgb_frame.device};depth={tuple(depth.shape)}/{depth.dtype}/{depth.device}"
        ipd_mm = getattr(stereo_config, "ipd_mm", None)
        if ipd_mm is None:
            effective_ipd_m = max(0.0, float(getattr(stereo_config, "ipd", 0.064)))
        else:
            effective_ipd_m = max(0.0, float(ipd_mm)) / 1000.0 * max(0.0, float(getattr(stereo_config, "stereo_scale", 0.4)))
        try:
            return make_fast_plus_fused_half_sbs_uint8(
                rgb_frame,
                depth,
                depth_strength=float(getattr(stereo_config, "depth_strength", 2.0)),
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                effective_ipd_m=effective_ipd_m,
                max_shift_ratio=float(getattr(stereo_config, "max_shift_ratio", 0.05)),
                edge_threshold=0.03,
            ), "used"
        except Exception as exc:
            return None, f"kernel_failed:{type(exc).__name__}"

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


StereoLabRuntime = StereoRuntime
StereoLabRuntimeResult = StereoRuntimeResult
StereoLabOpenXRRuntimeResult = OpenXRRuntimeResult
StereoLabDepthRuntime = DepthRuntime
StereoLabDepthRuntimeResult = DepthRuntimeResult


def _provider_report(depth_provider: Any) -> dict[str, Any]:
    info = getattr(depth_provider, "info", None)
    if info is None:
        return {}
    to_report = getattr(info, "to_report", None)
    if callable(to_report):
        return to_report()
    if isinstance(info, dict):
        return dict(info)
    return {"info": str(info)}


def _validate_runtime_rgb_frame(rgb_frame: Any) -> torch.Tensor:
    """Validate the capture/runtime boundary without doing capture adaptation."""
    if not isinstance(rgb_frame, torch.Tensor):
        raise TypeError("rgb_frame must be a torch.Tensor prepared by the capture layer")
    if rgb_frame.ndim not in (3, 4):
        raise ValueError(f"rgb_frame must be CHW or BCHW, got shape {tuple(rgb_frame.shape)}")
    channel_dim = 0 if rgb_frame.ndim == 3 else 1
    if rgb_frame.shape[channel_dim] != 3:
        raise ValueError(f"rgb_frame must be RGB with 3 channels in CHW/BCHW layout, got shape {tuple(rgb_frame.shape)}")
    if not rgb_frame.is_floating_point():
        raise TypeError(f"rgb_frame must be float 0..1; got dtype {rgb_frame.dtype}")
    return rgb_frame



def _runtime_frame_size(frame) -> tuple[int, int] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return int(shape[2]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return int(shape[1]), int(shape[0])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    return None


def _runtime_size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "unknown"
    return f"{int(size[0])}x{int(size[1])}"


def _runtime_eye_dtype(left_eye, right_eye) -> str:
    left_dtype = str(getattr(left_eye, "dtype", "unknown")).replace("torch.", "")
    right_dtype = str(getattr(right_eye, "dtype", "unknown")).replace("torch.", "")
    if left_dtype == right_dtype:
        return left_dtype
    return f"left={left_dtype},right={right_dtype}"


def _runtime_eye_size(eye) -> str:
    shape = tuple(getattr(eye, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3 and shape[-1] >= 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"
def _runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_RUNTIME_OUTPUT_UINT8", "0")


def _openxr_runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", os.environ.get("D2S_RUNTIME_OUTPUT_UINT8", "1"))


def _openxr_prewarp_eyes_enabled() -> bool:
    return _env_flag("D2S_OPENXR_PREWARP_EYES", "0")


def _openxr_rgb_depth_temporal_alpha() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    try:
        return max(0.0, min(0.98, float(raw)))
    except Exception:
        return 0.9


def _openxr_rgb_depth_gamma() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_GAMMA", "1.2")
    try:
        return max(0.1, min(4.0, float(raw)))
    except Exception:
        return 1.2


def _openxr_rgb_depth_percentile() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")
    try:
        return max(0.0, min(20.0, float(raw)))
    except Exception:
        return 0.0


def _openxr_rgb_depth_percentile_normalize(depth: torch.Tensor, *, percentile: float) -> torch.Tensor:
    depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
    if percentile <= 0.0:
        return depth
    flat = depth.flatten(start_dim=2)
    lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
    hi_q = 1.0 - lo_q
    count = flat.shape[-1]
    if count <= 1:
        return depth
    lo_idx = min(count - 1, max(0, int(round(lo_q * (count - 1)))))
    hi_idx = min(count - 1, max(0, int(round(hi_q * (count - 1)))))
    sorted_vals = torch.sort(flat, dim=-1).values
    lo = sorted_vals[..., lo_idx].view(depth.shape[0], 1, 1, 1)
    hi = sorted_vals[..., hi_idx].view(depth.shape[0], 1, 1, 1)
    return ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _env_flag(name: str, default: object = "0") -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _fast_plus_fused_enabled() -> bool:
    return str(os.environ.get("D2S_FAST_PLUS_FUSED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}


def _try_make_runtime_uint8_sbs(stereo: StereoResult, output_format: str) -> torch.Tensor | None:
    if output_format != "half_sbs":
        return None
    try:
        from .output_triton import can_use_triton_half_sbs, make_half_sbs_uint8
    except Exception:
        return None
    left = stereo.left_eye
    right = stereo.right_eye
    if not can_use_triton_half_sbs(left, right):
        return None
    try:
        return make_half_sbs_uint8(left, right)
    except Exception:
        return None
