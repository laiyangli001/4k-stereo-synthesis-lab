"""Phase-0 Panda3D glTF animation screenshot diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from pygltflib import GLTF2

from xr_viewer.panda3d_node_animation import GltfNodeAnimationPlayer, GltfNodeAnimationRuntime
from xr_viewer.panda3d_probe import _animation_sample_times


@dataclass(frozen=True)
class Panda3DAnimationScreenshotFrame:
    time_seconds: float
    screenshot_path: str
    screenshot_sha256: str
    screenshot_byte_length: int


@dataclass(frozen=True)
class Panda3DAnimationScreenshotProbeReport:
    asset_path: str
    output_dir: str
    width: int
    height: int
    duration_seconds: float
    sampled_node_name: str
    sample_times_seconds: tuple[float, ...]
    transform_changed: bool
    frame_count: int
    frames: tuple[Panda3DAnimationScreenshotFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Panda3DAnimationScreenshotProbeError(RuntimeError):
    """Raised when Panda3D cannot render animation diagnostic screenshots."""


def panda3d_animation_screenshot_probe_available() -> bool:
    return bool(importlib.util.find_spec("panda3d")) and bool(
        importlib.util.find_spec("gltf")
    )


def _matrix_fingerprint(node_path: Any) -> tuple[float, ...]:
    matrix = node_path.get_mat()
    return tuple(round(matrix.get_cell(row, col), 6) for row in range(4) for col in range(4))


def _first_bound_node(runtime: GltfNodeAnimationRuntime) -> Any | None:
    for target_node in runtime.target_node_ids:
        node_path = runtime.get_bound_node_path(target_node)
        if node_path is not None:
            return node_path
    return None


def _frame_camera(base: Any, root: Any) -> None:
    min_point, max_point = root.get_tight_bounds()
    if min_point is None or max_point is None:
        base.cam.set_pos(0, -10, 3)
        base.cam.look_at(0, 0, 0)
        return
    center = (min_point + max_point) * 0.5
    extent = max_point - min_point
    radius = max(1.0, extent.length() * 0.5)
    base.cam.set_pos(center.x, center.y - radius * 2.8, center.z + radius * 0.25)
    base.cam.look_at(center)


def inspect_panda3d_animation_screenshots(
    asset_path: str | Path,
    output_dir: str | Path,
    *,
    width: int = 512,
    height: int = 512,
) -> Panda3DAnimationScreenshotProbeReport:
    path = Path(asset_path).resolve()
    if not path.is_file():
        raise Panda3DAnimationScreenshotProbeError(f"GLB asset does not exist: {path}")
    if width <= 0 or height <= 0:
        raise Panda3DAnimationScreenshotProbeError("Screenshot dimensions must be positive")
    if not panda3d_animation_screenshot_probe_available():
        raise Panda3DAnimationScreenshotProbeError("Panda3D animation screenshot dependencies are unavailable")

    try:
        import gltf
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import Filename, NodePath, load_prc_file_data
    except ImportError as exc:  # pragma: no cover - guarded above for diagnostics
        raise Panda3DAnimationScreenshotProbeError("Panda3D animation screenshot imports failed") from exc

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    load_prc_file_data(
        "d2s-panda-animation-screenshot-probe",
        "\n".join(
            [
                "window-type offscreen",
                "load-display pandagl",
                "audio-library-name null",
                "sync-video false",
                "show-frame-rate-meter false",
                "notify-level-display error",
                f"win-size {width} {height}",
            ]
        ),
    )

    base: Any | None = None
    try:
        base = ShowBase(windowType="offscreen")
        if not base.win:
            raise Panda3DAnimationScreenshotProbeError("Panda3D did not create an offscreen window")

        root = NodePath(gltf.load_model(str(path)))
        root.reparent_to(base.render)
        _frame_camera(base, root)
        base.set_background_color(1.0, 1.0, 1.0, 1.0)

        runtime = GltfNodeAnimationRuntime(GLTF2().load(str(path)), root)
        sample_times = _animation_sample_times(runtime.duration_seconds)
        if not sample_times:
            raise Panda3DAnimationScreenshotProbeError("Asset has no glTF node animation samples")
        sampled_node = _first_bound_node(runtime)
        if sampled_node is None:
            raise Panda3DAnimationScreenshotProbeError("Animation runtime bound no Panda NodePath")

        player = GltfNodeAnimationPlayer(runtime, loop=False)
        fingerprints: list[tuple[float, ...]] = []
        frames: list[Panda3DAnimationScreenshotFrame] = []
        for sample_time in sample_times:
            player.set_time_seconds(sample_time)
            fingerprints.append(_matrix_fingerprint(sampled_node))
            base.graphicsEngine.render_frame()
            base.graphicsEngine.render_frame()
            screenshot_path = output / f"{path.stem}_animation_{sample_time:05.2f}s.png"
            saved = base.win.save_screenshot(Filename.from_os_specific(str(screenshot_path)))
            if not saved or not screenshot_path.is_file():
                raise Panda3DAnimationScreenshotProbeError(f"Failed to save screenshot: {screenshot_path}")
            screenshot_bytes = screenshot_path.read_bytes()
            frames.append(
                Panda3DAnimationScreenshotFrame(
                    time_seconds=float(sample_time),
                    screenshot_path=str(screenshot_path),
                    screenshot_sha256=hashlib.sha256(screenshot_bytes).hexdigest(),
                    screenshot_byte_length=len(screenshot_bytes),
                )
            )

        return Panda3DAnimationScreenshotProbeReport(
            asset_path=str(path),
            output_dir=str(output),
            width=width,
            height=height,
            duration_seconds=runtime.duration_seconds,
            sampled_node_name=sampled_node.get_name(),
            sample_times_seconds=sample_times,
            transform_changed=len(set(fingerprints)) > 1,
            frame_count=len(frames),
            frames=tuple(frames),
        )
    except Panda3DAnimationScreenshotProbeError:
        raise
    except Exception as exc:
        raise Panda3DAnimationScreenshotProbeError(
            f"Panda3D animation screenshot probe failed: {exc}"
        ) from exc
    finally:
        if base is not None:
            base.destroy()


def animation_screenshot_report_as_json(report: Panda3DAnimationScreenshotProbeReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
