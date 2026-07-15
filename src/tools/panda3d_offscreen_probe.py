"""Run the Panda3D OpenGL offscreen Phase-0 probe."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_viewer.panda3d_offscreen_probe import (  # noqa: E402
    Panda3DOffscreenProbeError,
    inspect_panda3d_offscreen,
    offscreen_report_as_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Panda3D OpenGL offscreen rendering compatibility probe."
    )
    parser.add_argument("--width", type=int, default=64, help="Offscreen buffer width")
    parser.add_argument("--height", type=int, default=64, help="Offscreen buffer height")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            report = inspect_panda3d_offscreen(args.width, args.height)
    except Panda3DOffscreenProbeError as exc:
        print(f"[Panda3DOffscreenProbe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(offscreen_report_as_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
