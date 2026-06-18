from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from dataclasses import replace
import time
from typing import Any

import torch

from .adapter import StereoRuntimeConfig, depth_provider_config_from_runtime, stereo_config_from_runtime
from .depth_safety import apply_depth_safety
from .depth_provider import DepthProfileResult, create_depth_provider
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
        depth_safety_report: dict[str, Any] | None = None
        if self._depth_safety_enabled():
            depth, decision = apply_depth_safety(rgb_frame, depth)
            depth_safety_report = decision.to_report()
            if decision.depth_strength_scale != 1.0:
                stereo_config = replace(
                    stereo_config,
                    depth_strength=stereo_config.depth_strength * decision.depth_strength_scale,
                )

        synth_start = time.perf_counter()
        stereo = synthesize_stereo(
            rgb_frame,
            depth,
            stereo_config,
            temporal_state=self.temporal_state,
        )
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
        if depth_safety_report is not None:
            debug["depth_safety"] = depth_safety_report

        return StereoRuntimeResult(
            depth=depth,
            left_eye=stereo.left_eye,
            right_eye=stereo.right_eye,
            sbs=stereo.sbs,
            debug_info=debug,
            timing=timing,
            provider_info=self.provider_report(),
        )

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

    def _depth_safety_enabled(self) -> bool:
        if self.config.depth_safety is not None:
            return bool(self.config.depth_safety)
        return self.config.mode == "image"

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
