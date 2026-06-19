from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import glfw
import moderngl
from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glFinish, glReadPixels

from viewer.viewer import StereoWindow
from scripts.tools.runtime_viewer_visual_regression import make_half_sbs_pattern, edge_stats

OUT_DIR = ROOT / "outputs" / "visual_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_screen_rgb(width: int, height: int) -> np.ndarray:
    glFinish()
    raw = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return np.flipud(arr).copy()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    width, height = 3840, 2160
    pattern = make_half_sbs_pattern(width, height)
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
        use_cuda=True,
        cuda_device_id=0,
        local_vsync=False,
    )
    try:
        window.ctx.screen.use()
        window._color_tex_components = 4
        window.color_tex = window.ctx.texture((width, height), 4, dtype="f1")
        window.color_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        window.depth_tex = window.ctx.texture((width, height), 1, dtype="f4")
        window.depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        window.depth_tex.write(np.zeros((height, width), dtype=np.float32).tobytes())
        window._texture_size = (width, height)
        window._init_cuda_pbos(width, height)
        gpu = torch.from_numpy(pattern.copy()).cuda(non_blocking=False).contiguous()
        torch.cuda.synchronize()
        window._upload_color_cuda_image(gpu)
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

    capture_path = OUT_DIR / "viewer_half_sbs_cuda_texture_capture.png"
    Image.fromarray(captured).save(capture_path)

    h = min(captured.shape[0], pattern.shape[0])
    w = min(captured.shape[1], pattern.shape[1])
    diff = np.abs(captured[:h, :w].astype(np.int16) - pattern[:h, :w].astype(np.int16)).astype(np.uint8)
    diff_path = OUT_DIR / "viewer_half_sbs_cuda_texture_diff.png"
    Image.fromarray(diff).save(diff_path)

    report = {
        "capture": str(capture_path),
        "diff": str(diff_path),
        "framebuffer": [int(fb_w), int(fb_h)],
        "capture_shape": list(captured.shape),
        "max_abs_diff_common_area": int(diff.max()),
        "mean_abs_diff_common_area": round(float(diff.mean()), 4),
        "capture_edge_stats": edge_stats(captured),
    }
    report_path = OUT_DIR / "viewer_half_sbs_cuda_texture_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
