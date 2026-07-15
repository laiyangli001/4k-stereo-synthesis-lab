"""Render Phase-0 Panda3D animation diagnostic screenshots for a GLB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_viewer.panda3d_animation_screenshot_probe import (  # noqa: E402
    Panda3DAnimationScreenshotProbeError,
    animation_screenshot_report_as_json,
    inspect_panda3d_animation_screenshots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Panda3D glTF animation screenshots at 0/mid/end sample times."
    )
    parser.add_argument("asset", type=Path, help="Path to an animated .glb or .gltf asset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG screenshots")
    parser.add_argument("--width", type=int, default=512, help="Screenshot width")
    parser.add_argument("--height", type=int, default=512, help="Screenshot height")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_panda3d_animation_screenshots(
            args.asset,
            args.output_dir,
            width=args.width,
            height=args.height,
        )
    except Panda3DAnimationScreenshotProbeError as exc:
        print(f"[Panda3DAnimationScreenshotProbe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(animation_screenshot_report_as_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
