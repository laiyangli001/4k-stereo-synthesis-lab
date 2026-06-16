from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--out", default="outputs/synthesis_profile_4k/profile.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["fast", "quality_4k", "hq_4k"], default="quality_4k")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output-format", choices=["half_sbs", "full_sbs"], default="half_sbs")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    import torch

    from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_lab.io import load_rgb
    from stereo_lab.baseline_shift import ShiftParams, compute_shift_px, warp_horizontal
    from stereo_lab.hole_fill import edge_aware_fill
    from stereo_lab.layers import composite_layers, make_depth_layers
    from stereo_lab.occlusion import make_occlusion_mask
    from stereo_lab.output import make_sbs, match_depth
    from stereo_lab.synthesis import StereoConfig, synthesize_stereo

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rgb = load_rgb(args.rgb, device=device)
    provider = create_depth_provider(DepthProviderConfig(backend="tensorrt_native", device=device))
    provider.load()
    depth = provider.predict(rgb)
    config = StereoConfig(backend=args.backend, layers=args.layers, output_format=args.output_format, temporal=False)

    def sync() -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    timings = []
    breakdown = {
        "baseline_shift": [],
        "make_layers": [],
        "warp_layers": [],
        "composite": [],
        "occlusion": [],
        "hole_fill": [],
        "make_sbs": [],
    }

    def profile_quality_once():
        params = ShiftParams()
        local_depth = match_depth(depth, rgb.shape[-2], rgb.shape[-1])
        sync()
        start = time.perf_counter()
        base_shift = compute_shift_px(local_depth, rgb.shape[-1], params)
        sync()
        baseline_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        weights = make_depth_layers(local_depth, layers=args.layers)
        sync()
        layers_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        left_layers = []
        right_layers = []
        for layer_idx in range(args.layers):
            layer_shift = base_shift * (0.75 + 0.25 * (layer_idx + 1) / args.layers)
            left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=-1.0))
            right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
        sync()
        warp_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        left = composite_layers(left_layers, weights)
        right = composite_layers(right_layers, weights)
        sync()
        composite_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        mask = make_occlusion_mask(local_depth, base_shift)
        sync()
        occlusion_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        eyes = torch.cat([left, right], dim=0)
        fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
        eyes = edge_aware_fill(eyes, fill_mask, radius=3, strength=1.0)
        left, right = eyes.chunk(2, dim=0)
        sync()
        fill_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        sbs = make_sbs(left, right, args.output_format)
        sync()
        sbs_ms = (time.perf_counter() - start) * 1000.0
        return sbs, {
            "baseline_shift": baseline_ms,
            "make_layers": layers_ms,
            "warp_layers": warp_ms,
            "composite": composite_ms,
            "occlusion": occlusion_ms,
            "hole_fill": fill_ms,
            "make_sbs": sbs_ms,
        }
    with torch.inference_mode():
        synthesize_stereo(rgb, depth, config)
        for idx in range(args.iters):
            sync()
            start = time.perf_counter()
            if args.backend == "quality_4k":
                sbs, parts = profile_quality_once()
                result = type("ProfileResult", (), {"sbs": sbs})()
                for key, value in parts.items():
                    breakdown[key].append(value)
            else:
                result = synthesize_stereo(rgb, depth, config)
            sync()
            elapsed = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed)
            print(f"iter={idx + 1} synthesis_ms={elapsed:.3f}", flush=True)

    report = {
        "rgb": str(args.rgb),
        "input_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "backend": args.backend,
        "layers": args.layers,
        "output_format": args.output_format,
        "output_shape": list(result.sbs.shape),
        "timings_ms": timings,
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "breakdown_mean_ms": {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in breakdown.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
