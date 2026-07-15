"""Inspect glTF material semantics before enabling the Panda3D OpenXR renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_viewer.panda3d_material_probe import (  # noqa: E402
    Panda3DMaterialProbeError,
    inspect_panda3d_materials,
    material_report_as_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Panda3D glTF Phase-0 material semantics probe."
    )
    parser.add_argument("asset", type=Path, help="Path to a .glb or .gltf asset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_panda3d_materials(args.asset)
    except Panda3DMaterialProbeError as exc:
        print(f"[Panda3DMaterialProbe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(material_report_as_json(report))
    return 0 if report.material_semantics_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
