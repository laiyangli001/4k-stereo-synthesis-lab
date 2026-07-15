"""Inspect a GLB with Panda3D before enabling the Panda3D OpenXR renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_viewer.panda3d_probe import (  # noqa: E402
    Panda3DProbeError,
    inspect_panda3d_asset,
    report_as_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Panda3D glTF Phase-0 asset compatibility probe."
    )
    parser.add_argument("asset", type=Path, help="Path to a .glb or .gltf asset")
    parser.add_argument(
        "--strict-animation",
        action="store_true",
        help="Return exit code 2 when an animated glTF exposes no Panda3D animation runtime nodes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_panda3d_asset(args.asset)
    except Panda3DProbeError as exc:
        print(f"[Panda3DProbe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(report_as_json(report))
    return 2 if args.strict_animation and not report.animation_runtime_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
