from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def colorize_depth(depth):
    import torch

    from stereo_lab.output import ensure_b1hw

    depth = ensure_b1hw(depth).float().clamp(0, 1)
    near = depth
    mid = (1.0 - (depth - 0.5).abs() * 2.0).clamp(0, 1)
    far = 1.0 - depth
    return torch.cat([near, mid, far], dim=1).clamp(0, 1)


def main() -> None:
    print("[1/6] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--reference-depth", default=None)
    parser.add_argument("--out-dir", default="outputs/depth_compare")
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-cache-dir", default=None)
    parser.add_argument("--depth-local-only", action="store_true")
    parser.add_argument("--depth-force-download", action="store_true")
    args = parser.parse_args()

    print("[2/6] importing torch ...", flush=True)
    import torch

    print("[3/6] importing stereo_lab ...", flush=True)
    from stereo_lab.depth_provider import estimate_distill_any_depth_base_518
    from stereo_lab.io import load_depth, load_rgb, save_depth, save_rgb
    from stereo_lab.output import match_depth
    from stereo_lab.report import absdiff, basic_image_metrics, make_contact_sheet, write_json

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    print("[info] depth model: Distill-Any-Depth-Base @ 518", flush=True)
    print("[info] model id: lc700x/Distill-Any-Depth-Base-hf", flush=True)

    print("[4/6] loading RGB ...", flush=True)
    rgb = load_rgb(args.rgb, device=device)

    print("[5/6] estimating depth ...", flush=True)
    with torch.inference_mode():
        depth, provider_info = estimate_distill_any_depth_base_518(
            rgb,
            device=device,
            cache_dir=args.depth_cache_dir,
            local_files_only=args.depth_local_only,
            force_download=args.depth_force_download,
        )

        save_rgb(rgb.cpu(), out_dir / "input_rgb.png")
        save_depth(depth.cpu(), out_dir / "distill_base_518_depth.png")
        save_rgb(colorize_depth(depth).cpu(), out_dir / "distill_base_518_depth_color.png")

        sheet_items = [rgb.cpu(), depth.repeat(1, 3, 1, 1).cpu(), colorize_depth(depth).cpu()]
        report = {
            "rgb": str(args.rgb),
            "depth_provider": provider_info.to_report(),
            "outputs": {
                "depth": "distill_base_518_depth.png",
                "depth_color": "distill_base_518_depth_color.png",
                "depth_shape": list(depth.shape),
            },
            "comparisons": {},
        }

        if args.reference_depth:
            reference = load_depth(args.reference_depth, device=device)
            reference = match_depth(reference, depth.shape[-2], depth.shape[-1])
            depth_rgb = depth.repeat(1, 3, 1, 1)
            reference_rgb = reference.repeat(1, 3, 1, 1)
            diff = absdiff(reference_rgb.cpu(), depth_rgb.cpu())
            save_depth(reference.cpu(), out_dir / "reference_depth_matched.png")
            save_rgb(diff, out_dir / "reference_vs_distill_absdiff.png")
            sheet_items.extend([reference_rgb.cpu(), diff])
            report["reference_depth"] = str(args.reference_depth)
            report["comparisons"]["reference_vs_distill"] = basic_image_metrics(reference_rgb.cpu(), depth_rgb.cpu())

        contact = make_contact_sheet(sheet_items, columns=2)
        save_rgb(contact, out_dir / "depth_contact_sheet.png")
        write_json(report, out_dir / "depth_report.json")

    print(f"[6/6] depth output written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
