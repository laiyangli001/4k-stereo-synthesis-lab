from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate roll-adaptive OpenXR stereo preview images.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--out-dir", default="outputs/openxr_preview")
    parser.add_argument("--device", default=None)
    parser.add_argument("--screen-roll-deg", type=float, default=0.0)
    parser.add_argument("--depth-strength", type=float, default=2.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--ipd", type=float, default=0.064)
    parser.add_argument("--max-shift-ratio", type=float, default=0.05)
    args = parser.parse_args()

    import torch

    from stereo_lab.io import load_depth, load_rgb, save_rgb
    from stereo_lab.openxr_render import OpenXRRenderConfig, is_pyopenxr_available, render_openxr_stereo

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rgb = load_rgb(args.rgb, device=device)
    depth = load_depth(args.depth, device=device)
    roll_rad = math.radians(args.screen_roll_deg)
    config = OpenXRRenderConfig(
        depth_strength=args.depth_strength,
        convergence=args.convergence,
        ipd=args.ipd,
        max_shift_ratio=args.max_shift_ratio,
        screen_roll=roll_rad,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        result = render_openxr_stereo(rgb, depth, config)
        save_rgb(result.left_eye.cpu(), out_dir / "openxr_left_eye.png")
        save_rgb(result.right_eye.cpu(), out_dir / "openxr_right_eye.png")

    report = {
        "rgb": str(args.rgb),
        "depth": str(args.depth),
        "device": str(device),
        "pyopenxr_available": is_pyopenxr_available(),
        "pyopenxr_import_name": "xr",
        "screen_roll_deg": args.screen_roll_deg,
        "screen_roll_rad": roll_rad,
        "left_shape": list(result.left_eye.shape),
        "right_shape": list(result.right_eye.shape),
        "debug": {k: v for k, v in result.debug_info.items() if isinstance(v, (float, int, str))},
    }
    (out_dir / "openxr_preview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
