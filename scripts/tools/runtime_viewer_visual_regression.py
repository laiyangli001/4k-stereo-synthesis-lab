from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import glfw
import moderngl
from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glFinish, glReadPixels

from viewer.viewer import StereoWindow


OUT_DIR = ROOT / "outputs" / "visual_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_half_sbs_pattern(width: int = 3840, height: int = 2160) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    half = width // 2

    # Left eye: red horizontal gradient with hard border markers.
    x_left = np.linspace(40, 230, half, dtype=np.uint8)
    img[:, :half, 0] = x_left[None, :]
    img[:, :half, 1] = 32
    img[:, :half, 2] = 48

    # Right eye: green/blue horizontal gradient with hard border markers.
    x_right = np.linspace(40, 230, width - half, dtype=np.uint8)
    img[:, half:, 0] = 32
    img[:, half:, 1] = x_right[None, :]
    img[:, half:, 2] = 210

    border = 48
    seam = 32
    # Outer frame must remain visible if the whole packed image is shown.
    img[:border, :, :] = (255, 255, 0)       # top yellow
    img[-border:, :, :] = (255, 0, 255)      # bottom magenta
    img[:, :border, :] = (255, 0, 0)         # left red
    img[:, -border:, :] = (0, 255, 255)      # right cyan

    # SBS center seam: white band with black edges.
    img[:, half - seam:half + seam, :] = 255
    img[:, half - seam - 8:half - seam, :] = 0
    img[:, half + seam:half + seam + 8, :] = 0

    # Per-eye inner landmarks; if a half is cropped these disappear or move.
    img[height // 2 - 24:height // 2 + 24, half // 2 - 24:half // 2 + 24, :] = (255, 128, 0)
    img[height // 2 - 24:height // 2 + 24, half + half // 2 - 24:half + half // 2 + 24, :] = (0, 128, 255)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((border + 20, border + 20), "LEFT EYE - OUTER EDGES MUST SHOW", fill=(255, 255, 255))
    draw.text((half + border + 20, border + 20), "RIGHT EYE - OUTER EDGES MUST SHOW", fill=(0, 0, 0))
    draw.text((half - 180, height - border - 60), "CENTER SEAM", fill=(0, 0, 0))
    return np.asarray(pil, dtype=np.uint8)


def read_screen_rgb(width: int, height: int) -> np.ndarray:
    glFinish()
    raw = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return np.flipud(arr).copy()


def edge_stats(img: np.ndarray) -> dict:
    h, w, _ = img.shape
    half = w // 2
    samples = {
        "top_mean": img[:48].mean(axis=(0, 1)).tolist(),
        "bottom_mean": img[-48:].mean(axis=(0, 1)).tolist(),
        "left_edge_mean": img[:, :48].mean(axis=(0, 1)).tolist(),
        "right_edge_mean": img[:, -48:].mean(axis=(0, 1)).tolist(),
        "center_seam_mean": img[:, half - 24:half + 24].mean(axis=(0, 1)).tolist(),
        "left_landmark_mean": img[h // 2 - 20:h // 2 + 20, w // 4 - 20:w // 4 + 20].mean(axis=(0, 1)).tolist(),
        "right_landmark_mean": img[h // 2 - 20:h // 2 + 20, half + w // 4 - 20:half + w // 4 + 20].mean(axis=(0, 1)).tolist(),
    }
    return {k: [round(float(v), 2) for v in values] for k, values in samples.items()}


def main() -> int:
    width, height = 3840, 2160
    pattern = make_half_sbs_pattern(width, height)
    input_path = OUT_DIR / "viewer_half_sbs_input_pattern.png"
    Image.fromarray(pattern).save(input_path)

    window = StereoWindow(
        capture_mode="Monitor",
        monitor_index=1,
        display_mode="Half-SBS",
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
    try:
        window.ctx.screen.use()
        window.color_tex = window.ctx.texture((width, height), 3, dtype="f1")
        window.color_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        window.depth_tex = window.ctx.texture((width, height), 1, dtype="f4")
        window.depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        window.depth_tex.write(np.zeros((height, width), dtype=np.float32).tobytes())
        window.color_tex.write(pattern.tobytes())
        window._texture_size = (width, height)
        window._runtime_direct_output = True
        window._runtime_output_format = "half_sbs"
        window._render_scene(defer_overlay=True)
        glfw.swap_buffers(window.window)
        window._render_scene(defer_overlay=True)
        fb_w, fb_h = glfw.get_framebuffer_size(window.window)
        captured = read_screen_rgb(fb_w, fb_h)
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

    output_path = OUT_DIR / "viewer_half_sbs_render_capture.png"
    Image.fromarray(captured).save(output_path)
    diff = np.abs(captured.astype(np.int16) - pattern.astype(np.int16)).astype(np.uint8)
    diff_path = OUT_DIR / "viewer_half_sbs_render_diff.png"
    Image.fromarray(diff).save(diff_path)

    report = {
        "input": str(input_path),
        "capture": str(output_path),
        "diff": str(diff_path),
        "framebuffer": [int(fb_w), int(fb_h)],
        "input_shape": list(pattern.shape),
        "capture_shape": list(captured.shape),
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "capture_edge_stats": edge_stats(captured),
    }
    report_path = OUT_DIR / "viewer_half_sbs_render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())