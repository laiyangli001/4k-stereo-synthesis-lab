from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed visual regression set for stereo synthesis.")
    parser.add_argument("--rgb", required=True, help="Input RGB image.")
    parser.add_argument("--depth", help="Optional depth image. If omitted, --auto-depth is required.")
    parser.add_argument("--auto-depth", action="store_true", help="Estimate depth from RGB when --depth is omitted.")
    parser.add_argument(
        "--depth-backend",
        choices=["tensorrt_native", "onnx_cuda_dlpack", "onnx_cuda_iobinding", "pytorch_cuda", "luma"],
        default="tensorrt_native",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    parser.add_argument("--out-dir", default="outputs/visual_regression")
    parser.add_argument("--depth-strength", type=float, default=3.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--ipd", type=float, default=0.064)
    parser.add_argument("--max-shift-ratio", type=float, default=0.05)
    parser.add_argument("--quality-layers", type=int, default=2)
    parser.add_argument("--no-fused", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--temporal-strength", type=float, default=0.85)
    parser.add_argument("--auto-reset-temporal", action="store_true")
    parser.add_argument("--scene-reset-threshold", type=float, default=0.22)
    parser.add_argument("--reset-cooldown-frames", type=int, default=3)
    parser.add_argument("--foreground-scale", type=float, default=0.0)
    parser.add_argument("--depth-antialias-strength", type=float, default=0.0)
    parser.add_argument("--edge-dilation", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--cross-eyed", action="store_true")
    parser.add_argument("--anaglyph-method", choices=["red_cyan", "green_magenta", "amber_blue", "gray"], default="red_cyan")
    args = parser.parse_args()

    print("[1/5] importing torch and stereo_lab ...", flush=True)
    import torch

    from stereo_lab.auto_depth import estimate_luma_depth
    from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_lab.io import load_depth, load_rgb, save_depth, save_rgb
    from stereo_lab.output import make_sbs
    from stereo_lab.report import absdiff, basic_image_metrics, make_contact_sheet, make_labeled_contact_sheet, write_json
    from stereo_lab.synthesis import StereoConfig, synthesize_stereo

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/5] loading RGB on {device} ...", flush=True)
    rgb = load_rgb(args.rgb, device=device)

    depth_info: dict[str, object]
    if args.depth:
        print("[3/5] loading fixed depth file ...", flush=True)
        depth = load_depth(args.depth, device=device)
        depth_info = {"provider": "file", "path": str(args.depth)}
    elif args.auto_depth:
        print(f"[3/5] estimating fixed depth via {args.depth_backend} ...", flush=True)
        if args.depth_backend == "luma":
            depth = estimate_luma_depth(rgb)
            depth_info = {"provider": "luma_pseudo_depth"}
        else:
            provider = create_depth_provider(
                DepthProviderConfig(
                    backend=args.depth_backend,
                    device=str(device),
                    onnx_path=args.onnx,
                    engine_path=args.trt_engine,
                )
            )
            provider.load()
            depth = provider.predict(rgb)
            info = provider.info() if callable(provider.info) else provider.info
            depth_info = info.to_report()
    else:
        raise SystemExit("missing --depth. Provide a depth image or pass --auto-depth.")

    base_config = {
        "depth_strength": args.depth_strength,
        "convergence": args.convergence,
        "ipd": args.ipd,
        "max_shift_ratio": args.max_shift_ratio,
        "temporal": args.temporal,
        "temporal_strength": args.temporal_strength,
        "auto_reset_temporal": args.auto_reset_temporal,
        "scene_reset_threshold": args.scene_reset_threshold,
        "reset_cooldown_frames": args.reset_cooldown_frames,
        "foreground_scale": args.foreground_scale,
        "depth_antialias_strength": args.depth_antialias_strength,
        "edge_dilation": args.edge_dilation,
        "edge_threshold": args.edge_threshold,
        "cross_eyed": args.cross_eyed,
        "anaglyph_method": args.anaglyph_method,
        "debug_output": True,
        "fused": not args.no_fused,
    }
    configs = {
        "baseline": StereoConfig(backend="fast", output_format="full_sbs", **base_config),
        "quality_4k": StereoConfig(
            backend="quality_4k",
            layers=args.quality_layers,
            output_format="full_sbs",
            **base_config,
        ),
    }

    report: dict[str, object] = {
        "rgb": str(args.rgb),
        "depth": str(args.depth or f"auto_{args.depth_backend}"),
        "depth_info": depth_info,
        "device": str(device),
        "input_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "params": {
            "depth_strength": args.depth_strength,
            "convergence": args.convergence,
            "ipd": args.ipd,
            "max_shift_ratio": args.max_shift_ratio,
            "quality_layers": args.quality_layers,
            "temporal": args.temporal,
            "temporal_strength": args.temporal_strength,
            "auto_reset_temporal": args.auto_reset_temporal,
            "scene_reset_threshold": args.scene_reset_threshold,
            "reset_cooldown_frames": args.reset_cooldown_frames,
            "foreground_scale": args.foreground_scale,
            "depth_antialias_strength": args.depth_antialias_strength,
            "edge_dilation": args.edge_dilation,
            "edge_threshold": args.edge_threshold,
            "cross_eyed": args.cross_eyed,
            "anaglyph_method": args.anaglyph_method,
            "fused": not args.no_fused,
        },
        "methods": {},
        "comparisons": {},
    }

    print("[4/5] synthesizing baseline and quality_4k ...", flush=True)
    results = {}
    timings = {}
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        save_rgb(rgb.cpu(), out_dir / "input_rgb.png")
        save_depth(depth.cpu(), out_dir / "used_depth.png")

        for name, config in configs.items():
            start = time.perf_counter()
            result = synthesize_stereo(rgb, depth, config)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results[name] = result
            timings[name] = elapsed_ms

            half_sbs = make_sbs(result.left_eye, result.right_eye, "half_sbs", anaglyph_method=args.anaglyph_method)
            full_sbs = make_sbs(result.left_eye, result.right_eye, "full_sbs", anaglyph_method=args.anaglyph_method)
            half_tab = make_sbs(result.left_eye, result.right_eye, "half_tab", anaglyph_method=args.anaglyph_method)
            full_tab = make_sbs(result.left_eye, result.right_eye, "full_tab", anaglyph_method=args.anaglyph_method)
            mono = make_sbs(result.left_eye, result.right_eye, "mono", anaglyph_method=args.anaglyph_method)
            depth_map = make_sbs(result.left_eye, result.right_eye, "depth_map", depth=depth, anaglyph_method=args.anaglyph_method)
            anaglyph = make_sbs(result.left_eye, result.right_eye, "anaglyph", anaglyph_method=args.anaglyph_method)
            interleaved = make_sbs(result.left_eye, result.right_eye, "interleaved", anaglyph_method=args.anaglyph_method)
            leia = make_sbs(result.left_eye, result.right_eye, "leia", anaglyph_method=args.anaglyph_method)
            save_rgb(result.left_eye.cpu(), out_dir / f"{name}_left.png")
            save_rgb(result.right_eye.cpu(), out_dir / f"{name}_right.png")
            save_rgb(half_sbs.cpu(), out_dir / f"{name}_half_sbs.png")
            save_rgb(full_sbs.cpu(), out_dir / f"{name}_full_sbs.png")
            save_rgb(half_tab.cpu(), out_dir / f"{name}_half_tab.png")
            save_rgb(full_tab.cpu(), out_dir / f"{name}_full_tab.png")
            save_rgb(mono.cpu(), out_dir / f"{name}_mono.png")
            save_rgb(depth_map.cpu(), out_dir / f"{name}_depth_map.png")
            save_rgb(anaglyph.cpu(), out_dir / f"{name}_anaglyph.png")
            save_rgb(interleaved.cpu(), out_dir / f"{name}_interleaved.png")
            save_rgb(leia.cpu(), out_dir / f"{name}_leia.png")

            shift = result.debug_info.get("shift_px")
            if shift is not None:
                shift_vis = _normalize_debug_map(shift)
                save_depth(shift_vis.cpu(), out_dir / f"{name}_shift_px.png")
            mask = result.debug_info.get("occlusion_mask")
            if mask is not None:
                save_depth(mask.cpu(), out_dir / f"{name}_occlusion_mask.png")

            report["methods"][name] = {
                "backend": config.backend,
                "layers": int(result.debug_info.get("layers", 1 if config.backend == "fast" else config.layers)),
                "synthesis_ms": elapsed_ms,
                "left_shape": list(result.left_eye.shape),
                "right_shape": list(result.right_eye.shape),
                "half_sbs_shape": list(half_sbs.shape),
                "full_sbs_shape": list(full_sbs.shape),
                "half_tab_shape": list(half_tab.shape),
                "full_tab_shape": list(full_tab.shape),
                "mono_shape": list(mono.shape),
                "depth_map_shape": list(depth_map.shape),
                "anaglyph_shape": list(anaglyph.shape),
                "interleaved_shape": list(interleaved.shape),
                "leia_shape": list(leia.shape),
            }
            print(f"  {name}: {elapsed_ms:.3f} ms", flush=True)

        baseline = results["baseline"]
        quality = results["quality_4k"]
        for target in ("left_eye", "right_eye"):
            base_eye = getattr(baseline, target)
            quality_eye = getattr(quality, target)
            diff = absdiff(base_eye.cpu(), quality_eye.cpu())
            save_rgb(diff, out_dir / f"baseline_vs_quality_4k_{target}_absdiff.png")
            report["comparisons"][f"baseline_vs_quality_4k_{target}"] = basic_image_metrics(base_eye.cpu(), quality_eye.cpu())

        baseline_half = make_sbs(baseline.left_eye, baseline.right_eye, "half_sbs").cpu()
        quality_half = make_sbs(quality.left_eye, quality.right_eye, "half_sbs").cpu()
        baseline_full = make_sbs(baseline.left_eye, baseline.right_eye, "full_sbs").cpu()
        quality_full = make_sbs(quality.left_eye, quality.right_eye, "full_sbs").cpu()
        half_diff = absdiff(baseline_half, quality_half)
        full_diff = absdiff(baseline_full, quality_full)
        save_rgb(half_diff, out_dir / "baseline_vs_quality_4k_half_sbs_absdiff.png")
        save_rgb(full_diff, out_dir / "baseline_vs_quality_4k_full_sbs_absdiff.png")
        report["comparisons"]["baseline_vs_quality_4k_half_sbs"] = basic_image_metrics(baseline_half, quality_half)
        report["comparisons"]["baseline_vs_quality_4k_full_sbs"] = basic_image_metrics(baseline_full, quality_full)

        sheet_items = [
            rgb.cpu(),
            depth.repeat(1, 3, 1, 1).cpu(),
            baseline_half,
            quality_half,
            half_diff,
        ]
        quality_mask = quality.debug_info.get("occlusion_mask")
        if quality_mask is not None:
            sheet_items.append(quality_mask.repeat(1, 3, 1, 1).cpu())
        contact = make_contact_sheet(sheet_items, columns=2)
        save_rgb(contact, out_dir / "contact_sheet.png")
        labeled_items = [
            ("input_rgb", rgb.cpu()),
            ("used_depth", depth.repeat(1, 3, 1, 1).cpu()),
            ("baseline_half_sbs", baseline_half),
            ("quality_4k_half_sbs", quality_half),
            ("baseline_vs_quality_4k_half_absdiff", half_diff),
        ]
        if quality_mask is not None:
            labeled_items.append(("quality_4k_occlusion_mask", quality_mask.repeat(1, 3, 1, 1).cpu()))
        labeled_contact = make_labeled_contact_sheet(labeled_items, columns=2)
        save_rgb(labeled_contact, out_dir / "contact_sheet_labeled.png")
        write_json(report, out_dir / "visual_regression_report.json")

    print(f"[5/5] wrote visual regression set: {out_dir}", flush=True)


def _normalize_debug_map(tensor):
    import torch

    x = tensor.detach().float()
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return x.clamp(0, 1)


if __name__ == "__main__":
    main()
