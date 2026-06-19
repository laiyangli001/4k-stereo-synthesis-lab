from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(SRC)

import glfw
import moderngl

from viewer.viewer import StereoWindow

OUT_DIR = ROOT / "outputs" / "visual_regression"
DEFAULT_EYE_SIZE = (3840, 2160)


@dataclass(frozen=True)
class CaseSpec:
    output_format: str
    display_mode: str
    packed_size: tuple[int, int]
    filename_slug: str


CASES = (
    CaseSpec(
        output_format="half_sbs",
        display_mode="Half-SBS",
        packed_size=DEFAULT_EYE_SIZE,
        filename_slug="half_sbs",
    ),
    CaseSpec(
        output_format="full_sbs",
        display_mode="Full-SBS",
        packed_size=(DEFAULT_EYE_SIZE[0] * 2, DEFAULT_EYE_SIZE[1]),
        filename_slug="full_sbs",
    ),
)


def make_eye_pattern(width: int, height: int, *, eye: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx = np.broadcast_to(x[None, :], (height, width))
    yy = np.broadcast_to(y[:, None], (height, width))

    if eye == "left":
        img[..., 0] = np.maximum(xx, 48)
        img[..., 1] = yy // 3
        img[..., 2] = 40
        frame_color = (255, 32, 32)
        landmark_color = (255, 180, 0)
        text_color = (255, 255, 255)
        label = "LEFT 4K EYE"
    elif eye == "right":
        img[..., 0] = 28
        img[..., 1] = np.maximum(xx, 48)
        img[..., 2] = np.maximum(255 - yy // 2, 96)
        frame_color = (32, 220, 255)
        landmark_color = (0, 96, 255)
        text_color = (0, 0, 0)
        label = "RIGHT 4K EYE"
    else:
        raise ValueError(f"unknown eye: {eye}")

    border = 64
    img[:border, :, :] = frame_color
    img[-border:, :, :] = frame_color
    img[:, :border, :] = frame_color
    img[:, -border:, :] = frame_color

    # High-contrast landmarks make cropping, scaling, or eye swaps obvious.
    cx, cy = width // 2, height // 2
    img[cy - 48:cy + 48, cx - 48:cx + 48, :] = landmark_color
    img[height // 4 - 32:height // 4 + 32, width // 4 - 32:width // 4 + 32, :] = (255, 255, 255)
    img[3 * height // 4 - 32:3 * height // 4 + 32, 3 * width // 4 - 32:3 * width // 4 + 32, :] = (0, 0, 0)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((border + 28, border + 28), f"{label} {width}x{height}", fill=text_color)
    draw.text((width // 2 - 150, height - border - 72), "FRAME EDGES MUST REMAIN VISIBLE", fill=text_color)
    return np.asarray(pil, dtype=np.uint8)


def pack_half_sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width, _ = left.shape
    left_half = Image.fromarray(left).resize((width // 2, height), Image.Resampling.BOX)
    right_half = Image.fromarray(right).resize((width - width // 2, height), Image.Resampling.BOX)
    return np.concatenate(
        [np.asarray(left_half, dtype=np.uint8), np.asarray(right_half, dtype=np.uint8)],
        axis=1,
    )


def pack_full_sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)


def make_packed_frame(left: np.ndarray, right: np.ndarray, output_format: str) -> np.ndarray:
    if output_format == "half_sbs":
        return pack_half_sbs(left, right)
    if output_format == "full_sbs":
        return pack_full_sbs(left, right)
    raise ValueError(f"unsupported output format: {output_format}")


def read_fbo_rgb(fbo: moderngl.Framebuffer, width: int, height: int) -> np.ndarray:
    raw = fbo.read(components=3, alignment=1)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return np.flipud(arr).copy()


def diff_stats(captured: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(captured.astype(np.int16) - expected.astype(np.int16))
    return {
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "p95_abs_diff": round(float(np.percentile(diff, 95)), 4),
        "p99_abs_diff": round(float(np.percentile(diff, 99)), 4),
    }


def sample_stats(img: np.ndarray) -> dict:
    h, w, _ = img.shape
    half = w // 2
    samples = {
        "top_edge_mean": img[:64].mean(axis=(0, 1)),
        "bottom_edge_mean": img[-64:].mean(axis=(0, 1)),
        "left_edge_mean": img[:, :64].mean(axis=(0, 1)),
        "right_edge_mean": img[:, -64:].mean(axis=(0, 1)),
        "center_seam_mean": img[:, half - 32:half + 32].mean(axis=(0, 1)),
        "left_center_landmark_mean": img[h // 2 - 32:h // 2 + 32, w // 4 - 32:w // 4 + 32].mean(axis=(0, 1)),
        "right_center_landmark_mean": img[h // 2 - 32:h // 2 + 32, 3 * w // 4 - 32:3 * w // 4 + 32].mean(axis=(0, 1)),
    }
    return {key: [round(float(v), 2) for v in value] for key, value in samples.items()}


def timed_gpu_resident_output_fps(
    window: StereoWindow,
    fbo: moderngl.Framebuffer,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> dict:
    for _ in range(warmup_frames):
        fbo.use()
        window._render_scene(defer_overlay=True)

    frame_ms: list[float] = []
    for _ in range(measure_frames):
        start = time.perf_counter()
        fbo.use()
        window._render_scene(defer_overlay=True)
        window.ctx.finish()
        frame_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = sum(frame_ms)
    return {
        "metric": "gpu_resident_output",
        "notes": "GPU-resident texture render + GPU sync only; excludes CPU upload, CPU readback, display present, and vsync.",
        "frames": int(measure_frames),
        "total_ms": round(float(total_ms), 4),
        "fps": round(float(1000.0 * measure_frames / total_ms), 3) if total_ms > 0 else 0.0,
        "mean_ms": round(float(statistics.fmean(frame_ms)), 4),
        "median_ms": round(float(statistics.median(frame_ms)), 4),
        "min_ms": round(float(min(frame_ms)), 4),
        "max_ms": round(float(max(frame_ms)), 4),
    }

def timed_cpu_upload_output_fps(
    window: StereoWindow,
    fbo: moderngl.Framebuffer,
    tex: moderngl.Texture,
    packed: np.ndarray,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> dict:
    frame_bytes = packed.tobytes()
    for _ in range(warmup_frames):
        tex.write(frame_bytes)
        fbo.use()
        window._render_scene(defer_overlay=True)
        window.ctx.finish()

    frame_ms: list[float] = []
    for _ in range(measure_frames):
        start = time.perf_counter()
        tex.write(frame_bytes)
        fbo.use()
        window._render_scene(defer_overlay=True)
        window.ctx.finish()
        frame_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = sum(frame_ms)
    return {
        "metric": "cpu_upload_output",
        "notes": "Per-frame CPU packed-frame upload + GPU-resident viewer render + GPU sync; excludes CPU readback, display present, and vsync.",
        "frames": int(measure_frames),
        "total_ms": round(float(total_ms), 4),
        "fps": round(float(1000.0 * measure_frames / total_ms), 3) if total_ms > 0 else 0.0,
        "mean_ms": round(float(statistics.fmean(frame_ms)), 4),
        "median_ms": round(float(statistics.median(frame_ms)), 4),
        "min_ms": round(float(min(frame_ms)), 4),
        "max_ms": round(float(max(frame_ms)), 4),
        "input_mb_per_frame": round(float(packed.nbytes / (1024.0 * 1024.0)), 3),
    }
def create_window(width: int, height: int, display_mode: str) -> StereoWindow:
    return StereoWindow(
        capture_mode="Monitor",
        monitor_index=1,
        display_mode=display_mode,
        fill_16_9=True,
        show_fps=False,
        use_3d=False,
        fix_aspect=False,
        stream_mode="MJPEG",
        specify_display=False,
        frame_size=(width, height),
        use_cuda=False,
        local_vsync=False,
    )


def run_case(
    window: StereoWindow,
    case: CaseSpec,
    packed: np.ndarray,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> dict:
    width, height = case.packed_size
    if packed.shape != (height, width, 3):
        raise ValueError(f"{case.output_format} expected {(height, width, 3)}, got {packed.shape}")

    tex = None
    fbo_tex = None
    fbo = None
    try:
        tex = window.ctx.texture((width, height), 3, dtype="f1")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.write(packed.tobytes())
        fbo_tex = window.ctx.texture((width, height), 3, dtype="f1")
        fbo_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fbo = window.ctx.framebuffer(color_attachments=[fbo_tex])

        window.color_tex = tex
        window.depth_tex = None
        window._texture_size = (width, height)
        window._runtime_direct_output = True
        window._runtime_output_format = case.output_format
        window._scene_render_size_override = (width, height)
        window.display_mode = case.display_mode

        fbo.use()
        window._render_scene(defer_overlay=True)
        window.ctx.finish()
        captured = read_fbo_rgb(fbo, width, height)
        diff = np.abs(captured.astype(np.int16) - packed.astype(np.int16)).astype(np.uint8)
        gpu_resident_output_fps = timed_gpu_resident_output_fps(
            window,
            fbo,
            warmup_frames=warmup_frames,
            measure_frames=measure_frames,
        )
        cpu_upload_output_fps = timed_cpu_upload_output_fps(
            window,
            fbo,
            tex,
            packed,
            warmup_frames=warmup_frames,
            measure_frames=measure_frames,
        )
    finally:
        window._scene_render_size_override = None
        if fbo is not None:
            fbo.release()
        if fbo_tex is not None:
            fbo_tex.release()
        if tex is not None:
            tex.release()

    packed_path = OUT_DIR / f"local_4k_{case.filename_slug}_packed_input.png"
    capture_path = OUT_DIR / f"local_4k_{case.filename_slug}_render_capture.png"
    diff_path = OUT_DIR / f"local_4k_{case.filename_slug}_render_diff.png"
    Image.fromarray(packed).save(packed_path)
    Image.fromarray(captured).save(capture_path)
    Image.fromarray(diff).save(diff_path)

    return {
        "output_format": case.output_format,
        "display_mode": case.display_mode,
        "packed_size": [int(width), int(height)],
        "packed_input": str(packed_path),
        "capture": str(capture_path),
        "diff": str(diff_path),
        "capture_shape": list(captured.shape),
        "visual_diff": diff_stats(captured, packed),
        "capture_samples": sample_stats(captured),
        "gpu_resident_output_fps": gpu_resident_output_fps,
        "cpu_upload_output_fps": cpu_upload_output_fps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local 4K half-SBS and full-SBS visual regression plus output FPS measurement."
    )
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--measure-frames", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.measure_frames <= 0:
        raise ValueError("--measure-frames must be > 0")

    global OUT_DIR
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eye_w, eye_h = DEFAULT_EYE_SIZE
    left = make_eye_pattern(eye_w, eye_h, eye="left")
    right = make_eye_pattern(eye_w, eye_h, eye="right")
    left_path = OUT_DIR / "local_4k_left_eye_input.png"
    right_path = OUT_DIR / "local_4k_right_eye_input.png"
    Image.fromarray(left).save(left_path)
    Image.fromarray(right).save(right_path)

    max_w = max(case.packed_size[0] for case in CASES)
    max_h = max(case.packed_size[1] for case in CASES)
    window = create_window(max_w, max_h, "Half-SBS")
    try:
        reports = []
        for case in CASES:
            packed = make_packed_frame(left, right, case.output_format)
            reports.append(
                run_case(
                    window,
                    case,
                    packed,
                    warmup_frames=args.warmup_frames,
                    measure_frames=args.measure_frames,
                )
            )
    finally:
        try:
            window.cleanup_cuda()
        except Exception:
            pass
        try:
            glfw.destroy_window(window.window)
        except Exception:
            pass
        glfw.terminate()

    report = {
        "input_eye_size": [eye_w, eye_h],
        "left_input": str(left_path),
        "right_input": str(right_path),
        "warmup_frames": int(args.warmup_frames),
        "measure_frames": int(args.measure_frames),
        "cases": reports,
    }
    report_path = OUT_DIR / "local_4k_sbs_visual_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




