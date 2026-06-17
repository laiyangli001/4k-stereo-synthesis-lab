from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sync_if_cuda(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _synthetic_rgb_depth(height: int, width: int, device: str):
    import torch

    yy = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    red = xx.expand(1, 1, height, width)
    green = yy.expand(1, 1, height, width)
    blue = (0.55 + 0.35 * torch.sin(xx * 16.0) * torch.cos(yy * 12.0)).expand(1, 1, height, width)
    rgb = torch.cat([red, green, blue], dim=1).clamp(0, 1)
    depth = (0.15 + 0.85 * (1.0 - yy) + 0.08 * torch.sin(xx * 20.0)).clamp(0, 1)
    return rgb, depth


def _load_rgb(path: Path, device: str):
    from stereo_lab.io import load_rgb

    return load_rgb(path, device=device)


def _make_depth(args, rgb):
    if not args.auto_depth:
        return None, None

    from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider

    provider = create_depth_provider(
        DepthProviderConfig(
            backend=args.depth_backend,
            device=args.device,
            onnx_path=args.onnx,
            engine_path=args.trt_engine,
            local_files_only=not args.online,
        )
    )
    provider.load()
    return provider.predict(rgb), provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the public host API contract.")
    parser.add_argument("--rgb", default=None, help="Optional RGB image. If omitted, a synthetic tensor is used.")
    parser.add_argument("--auto-depth", action="store_true", help="Use a persistent depth provider instead of synthetic depth.")
    parser.add_argument("--depth-backend", default="tensorrt_native")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    parser.add_argument("--online", action="store_true", help="Allow online model loading for PyTorch fallback providers.")
    parser.add_argument("--preset", default="cinema", choices=["auto", "cinema", "game_low_latency", "still_image_hq", "debug_export"])
    parser.add_argument("--output-format", default="half_sbs", choices=["half_sbs", "full_sbs", "half_tab", "full_tab", "mono", "depth_map", "anaglyph", "interleaved", "leia"])
    parser.add_argument("--openxr", action="store_true", help="Smoke-test the OpenXR per-eye core instead of packed stereo output.")
    parser.add_argument("--screen-roll", type=float, default=0.0, help="OpenXR screen roll in radians when --openxr is used.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", default="outputs/host_api_smoke.json", help="JSON report path, or '-' to print without writing.")
    args = parser.parse_args()

    import torch

    from stereo_lab import openxr_config_for_preset, render_openxr_stereo, stereo_config_for_preset, synthesize_stereo
    from stereo_lab.temporal import TemporalState

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.rgb:
        rgb = _load_rgb(Path(args.rgb), device)
    else:
        rgb, synthetic_depth = _synthetic_rgb_depth(args.height, args.width, device)

    depth_provider = None
    if args.auto_depth:
        depth, depth_provider = _make_depth(args, rgb)
    else:
        if args.rgb:
            _, synthetic_depth = _synthetic_rgb_depth(rgb.shape[-2], rgb.shape[-1], device)
        depth = synthetic_depth

    stereo_config = stereo_config_for_preset(args.preset, output_format=args.output_format)
    openxr_config = openxr_config_for_preset(args.preset, screen_roll=args.screen_roll)
    temporal_state = TemporalState()

    timings = []
    result = None
    for _ in range(max(1, args.iters)):
        _sync_if_cuda(device)
        start = time.perf_counter()
        if args.openxr:
            result = render_openxr_stereo(rgb, depth, openxr_config)
        else:
            result = synthesize_stereo(rgb, depth, stereo_config, temporal_state=temporal_state)
        _sync_if_cuda(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    sbs_shape = None if args.openxr else list(result.sbs.shape)
    report = {
        "mode": "openxr" if args.openxr else "stereo",
        "preset": args.preset,
        "output_format": None if args.openxr else args.output_format,
        "screen_roll": args.screen_roll if args.openxr else None,
        "device": device,
        "auto_depth": bool(args.auto_depth),
        "depth_provider": getattr(getattr(depth_provider, "info", None), "depth_backend", None),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "left_eye_shape": list(result.left_eye.shape),
        "right_eye_shape": list(result.right_eye.shape),
        "sbs_shape": sbs_shape,
        "timings_ms": timings,
        "mean_render_ms": sum(timings) / len(timings),
        "debug_info": {k: v for k, v in result.debug_info.items() if isinstance(v, (float, int, str))},
    }

    text = json.dumps(report, indent=2)
    if args.out != "-":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
