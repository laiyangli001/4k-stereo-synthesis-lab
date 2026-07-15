"""Run the Panda3D OpenGL to D3D11 NV_DX_interop readiness probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_viewer.panda3d_d3d11_interop_probe import (  # noqa: E402
    Panda3DD3D11InteropProbeError,
    d3d11_interop_report_as_json,
    inspect_panda3d_d3d11_interop,
)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Run the Panda3D OpenGL to D3D11 NV_DX_interop readiness probe."
    )


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = inspect_panda3d_d3d11_interop()
    except Panda3DD3D11InteropProbeError as exc:
        print(f"[Panda3DD3D11InteropProbe] ERROR: {exc}", file=sys.stderr)
        return 1
    print(d3d11_interop_report_as_json(report))
    return 0 if report.readiness_status == "ready_for_swapchain_texture_registration" else 2


if __name__ == "__main__":
    raise SystemExit(main())
