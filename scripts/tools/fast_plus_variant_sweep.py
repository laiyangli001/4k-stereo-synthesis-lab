from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(SRC)

from stereo_runtime.baseline_shift import ShiftParams, synthesize_baseline
from stereo_runtime.hole_fill import edge_aware_fill
from stereo_runtime.io import load_depth, load_rgb, save_depth, save_rgb
from stereo_runtime.occlusion import make_occlusion_mask
from stereo_runtime.output import ensure_bchw, make_sbs, match_depth
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

OUT_DIR = ROOT / "outputs" / "visual_regression" / "fast_plus_sweep"


@dataclass(frozen=True)
class Variant:
    name: str
    fill: str
    edge_threshold: float
    dilation: int
    radius: int
    strength: float
    taps: int = 0
    direction: str = "none"


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, repeats: int, warmup: int = 1) -> tuple[object, float, list[float]]:
    result = None
    samples: list[float] = []
    for _ in range(max(0, warmup)):
        result = fn()
        sync_cuda()
    for _ in range(max(1, repeats)):
        sync_cuda()
        start = time.perf_counter()
        result = fn()
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, statistics.fmean(samples), samples


def make_synthetic_4k(width: int, height: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.0, 1.0, height, device=device).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, width)
    xx = x.expand(1, height, width)
    yy = y.expand(1, height, width)

    base = torch.cat([
        0.18 + 0.55 * xx,
        0.22 + 0.48 * yy,
        0.28 + 0.28 * torch.sin(xx * math.pi * 4.0) * torch.cos(yy * math.pi * 3.0),
    ], dim=0).clamp(0, 1)
    depth = (0.2 + 0.55 * (1.0 - yy) + 0.12 * torch.sin(xx * math.pi * 2.0)).clamp(0, 1)

    def rect(x0: float, y0: float, x1: float, y1: float, color: tuple[float, float, float], d: float) -> None:
        nonlocal base, depth
        m = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        color_t = torch.tensor(color, device=device).view(3, 1, 1)
        base = torch.where(m.expand_as(base), color_t, base)
        depth = torch.where(m, torch.full_like(depth, d), depth)

    def circle(cx: float, cy: float, r: float, color: tuple[float, float, float], d: float) -> None:
        nonlocal base, depth
        m = (xx - cx).square() + (yy - cy).square() <= r * r
        color_t = torch.tensor(color, device=device).view(3, 1, 1)
        base = torch.where(m.expand_as(base), color_t, base)
        depth = torch.where(m, torch.full_like(depth, d), depth)

    rect(0.08, 0.18, 0.35, 0.82, (0.92, 0.22, 0.18), 0.88)
    rect(0.48, 0.12, 0.76, 0.52, (0.18, 0.78, 0.92), 0.74)
    rect(0.66, 0.50, 0.93, 0.86, (0.94, 0.82, 0.20), 0.62)
    circle(0.40, 0.54, 0.14, (0.75, 0.30, 0.95), 0.93)

    # Thin high-contrast bars expose edge cracking and small hole fill behavior.
    for i in range(10):
        x0 = 0.12 + i * 0.045
        rect(x0, 0.08, x0 + 0.010, 0.94, (0.96, 0.96, 0.96), 0.70 + (i % 3) * 0.07)
    for i in range(7):
        y0 = 0.18 + i * 0.080
        rect(0.78, y0, 0.95, y0 + 0.012, (0.05, 0.05, 0.05), 0.82)

    return base.unsqueeze(0).contiguous(), depth.unsqueeze(0).contiguous()


def shift_x(x: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return x
    if offset > 0:
        pad = x[..., :, :1].expand(*x.shape[:-1], offset)
        return torch.cat([pad, x[..., :, :-offset]], dim=-1)
    offset = abs(offset)
    pad = x[..., :, -1:].expand(*x.shape[:-1], offset)
    return torch.cat([x[..., :, offset:], pad], dim=-1)


def directional_fill_one(eye: torch.Tensor, mask: torch.Tensor, *, eye_sign: int, taps: int, strength: float, direction: str) -> torch.Tensor:
    if taps <= 0:
        return eye
    if direction == "both":
        offsets = [t for t in range(1, taps + 1)] + [-t for t in range(1, taps + 1)]
    else:
        sign = eye_sign if direction == "same" else -eye_sign
        offsets = [sign * t for t in range(1, taps + 1)]
    accum = torch.zeros_like(eye)
    for offset in offsets:
        accum = accum + shift_x(eye, offset)
    filled = accum / float(len(offsets))
    blend = (mask * strength).clamp(0, 1)
    return torch.lerp(eye, filled, blend)


def synthesize_fast_plus(rgb: torch.Tensor, depth: torch.Tensor, config: StereoConfig, variant: Variant) -> tuple[torch.Tensor, dict]:
    params = ShiftParams(
        depth_strength=config.depth_strength,
        convergence=config.convergence,
        ipd=config.ipd,
        max_shift_ratio=config.max_shift_ratio,
        ipd_mm=config.ipd_mm,
        stereo_scale=config.stereo_scale,
    )
    left, right, shift_px = synthesize_baseline(rgb, depth, params)
    depth_matched = match_depth(depth, left.shape[-2], left.shape[-1])
    shift_matched = match_depth(shift_px, left.shape[-2], left.shape[-1])
    mask = make_occlusion_mask(
        depth_matched,
        shift_matched,
        edge_threshold=variant.edge_threshold,
        dilation=variant.dilation,
        fused=config.fused,
        screen_edge_suppression=config.screen_edge_mask_suppression,
    )
    if variant.fill == "blur":
        eyes = torch.cat([left, right], dim=0)
        fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
        eyes = edge_aware_fill(eyes, fill_mask, radius=variant.radius, strength=variant.strength, fused=config.fused)
        left, right = eyes.chunk(2, dim=0)
    elif variant.fill == "directional":
        left = directional_fill_one(left, mask, eye_sign=-1, taps=variant.taps, strength=variant.strength, direction=variant.direction)
        right = directional_fill_one(right, mask, eye_sign=1, taps=variant.taps, strength=variant.strength, direction=variant.direction)
    elif variant.fill != "none":
        raise ValueError(f"unknown fill type: {variant.fill}")
    if config.cross_eyed:
        left, right = right, left
    sbs = make_sbs(left, right, config.output_format, fused=config.fused, depth=depth_matched, anaglyph_method=config.anaglyph_method)
    return sbs, {"mask": mask, "shift_px": shift_px}


def sampled_quantile(x: torch.Tensor, q: float, max_items: int = 2_000_000) -> float:
    flat = x.detach().flatten()
    if flat.numel() > max_items:
        step = max(1, flat.numel() // max_items)
        flat = flat[::step]
    return float(torch.quantile(flat.float(), q).item())
def image_metrics(candidate: torch.Tensor, reference: torch.Tensor, edge_mask: torch.Tensor | None = None) -> dict[str, float]:
    candidate = ensure_bchw(candidate, name="candidate").float().clamp(0, 1)
    reference = ensure_bchw(reference, name="reference").float().clamp(0, 1)
    diff = (candidate - reference).abs()
    mse = (candidate - reference).square().mean().item()
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    out = {
        "mae": float(diff.mean().item()),
        "rmse": float(math.sqrt(mse)),
        "psnr": float(psnr),
        "p95_abs": sampled_quantile(diff, 0.95),
    }
    if edge_mask is not None:
        mask = edge_mask.float()
        if mask.shape[-2:] != diff.shape[-2:]:
            mask = F.interpolate(mask, size=diff.shape[-2:], mode="nearest")
        if mask.shape[1] == 1:
            mask = mask.expand(diff.shape[0], diff.shape[1], diff.shape[2], diff.shape[3])
        denom = mask.sum().clamp_min(1.0)
        out["edge_mae"] = float((diff * mask).sum().item() / denom.item())
        out["edge_coverage"] = float(mask.mean().item())
    return out


def save_diff(candidate: torch.Tensor, reference: torch.Tensor, path: Path, scale: float = 4.0) -> None:
    diff = (ensure_bchw(candidate, name="candidate") - ensure_bchw(reference, name="reference")).abs().clamp(0, 1)
    save_rgb((diff * scale).clamp(0, 1), path)


def build_variants() -> list[Variant]:
    variants: list[Variant] = []
    for edge_threshold in (0.03, 0.04):
        for dilation in (0, 1):
            for strength in (0.45, 0.60):
                variants.append(Variant(
                    name=f"blur_r1_s{strength:.2f}_e{edge_threshold:.2f}_d{dilation}",
                    fill="blur",
                    edge_threshold=edge_threshold,
                    dilation=dilation,
                    radius=1,
                    strength=strength,
                ))
            for taps in (2, 4):
                for direction in ("opposite", "both"):
                    variants.append(Variant(
                        name=f"dir{taps}_{direction}_s0.55_e{edge_threshold:.2f}_d{dilation}",
                        fill="directional",
                        edge_threshold=edge_threshold,
                        dilation=dilation,
                        radius=1,
                        strength=0.55,
                        taps=taps,
                        direction=direction,
                    ))
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fast_plus stereo-runtime variants against quality_4k visual baseline.")
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--depth", type=Path, default=None)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-format", choices=["half_sbs", "full_sbs"], default="half_sbs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-variants", type=int, default=0, help="0 means run all built-in variants")
    parser.add_argument("--depth-strength", type=float, default=2.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--save-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rgb is not None and not args.rgb.is_absolute():
        args.rgb = ROOT / args.rgb
    if args.depth is not None and not args.depth.is_absolute():
        args.depth = ROOT / args.depth
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.rgb:
        rgb = load_rgb(args.rgb, device=device)
        if args.depth:
            depth = load_depth(args.depth, device=device)
        else:
            # Fallback depth for arbitrary RGB: luminance plus vertical gradient. Prefer --depth for real comparisons.
            lum = rgb.mean(dim=1, keepdim=True)
            grad = torch.linspace(1.0, 0.0, rgb.shape[-2], device=device).view(1, 1, rgb.shape[-2], 1)
            depth = (0.65 * lum + 0.35 * grad).clamp(0, 1)
    else:
        rgb, depth = make_synthetic_4k(args.width, args.height, device)
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    save_rgb(rgb, args.out_dir / "input_rgb.png")
    save_depth(depth, args.out_dir / "input_depth.png")

    base_config = StereoConfig(
        backend="fast",
        output_format=args.output_format,
        depth_strength=args.depth_strength,
        convergence=args.convergence,
        temporal=False,
        debug_output=True,
        hole_fill="edge_aware",
        fused=True,
    )
    quality_config = StereoConfig(**{**base_config.__dict__, "backend": "quality_4k", "layers": 2})

    quality_result, quality_ms, quality_samples = time_ms(lambda: synthesize_stereo(rgb, depth, quality_config), args.repeats, args.warmup)
    fast_result, fast_ms, fast_samples = time_ms(lambda: synthesize_stereo(rgb, depth, base_config), args.repeats, args.warmup)
    quality_sbs = quality_result.sbs
    fast_sbs = fast_result.sbs
    edge_mask = quality_result.debug_info.get("occlusion_mask")

    save_rgb(quality_sbs, args.out_dir / f"quality_4k_{args.output_format}.png")
    save_rgb(fast_sbs, args.out_dir / f"fast_{args.output_format}.png")
    save_diff(fast_sbs, quality_sbs, args.out_dir / f"fast_vs_quality_diff_x4.png")
    if isinstance(edge_mask, torch.Tensor):
        save_depth(edge_mask, args.out_dir / "quality_occlusion_mask.png")

    rows: list[dict] = []
    rows.append({
        "name": "fast",
        "kind": "baseline",
        "time_ms": round(float(fast_ms), 4),
        "fps": round(1000.0 / fast_ms, 3),
        **{k: round(v, 6) for k, v in image_metrics(fast_sbs, quality_sbs, edge_mask if isinstance(edge_mask, torch.Tensor) else None).items()},
    })

    variants = build_variants()
    if args.max_variants > 0:
        variants = variants[: args.max_variants]

    for variant in variants:
        (sbs, debug), elapsed_ms, samples = time_ms(lambda v=variant: synthesize_fast_plus(rgb, depth, base_config, v), args.repeats, args.warmup)
        metrics = image_metrics(sbs, quality_sbs, edge_mask if isinstance(edge_mask, torch.Tensor) else None)
        row = {
            "name": variant.name,
            "kind": variant.fill,
            "edge_threshold": variant.edge_threshold,
            "dilation": variant.dilation,
            "radius": variant.radius,
            "strength": variant.strength,
            "taps": variant.taps,
            "direction": variant.direction,
            "time_ms": round(float(elapsed_ms), 4),
            "fps": round(1000.0 / elapsed_ms, 3),
            "samples_ms": [round(float(x), 4) for x in samples],
            **{k: round(v, 6) for k, v in metrics.items()},
        }
        rows.append(row)
        if args.save_all:
            save_rgb(sbs, args.out_dir / f"{variant.name}_{args.output_format}.png")
            save_diff(sbs, quality_sbs, args.out_dir / f"{variant.name}_vs_quality_diff_x4.png")

    rows_sorted = sorted(rows, key=lambda r: (r.get("edge_mae", r["mae"]), r["mae"], r["time_ms"]))
    top = rows_sorted[: min(6, len(rows_sorted))]
    for row in top:
        if row["name"] != "fast":
            variant = next(v for v in variants if v.name == row["name"])
            sbs, _ = synthesize_fast_plus(rgb, depth, base_config, variant)
            sync_cuda()
            save_rgb(sbs, args.out_dir / f"top_{row['name']}_{args.output_format}.png")
            save_diff(sbs, quality_sbs, args.out_dir / f"top_{row['name']}_vs_quality_diff_x4.png")

    report = {
        "config": {
            "output_format": args.output_format,
            "image_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "quality_time_ms": round(float(quality_ms), 4),
            "quality_fps": round(1000.0 / quality_ms, 3),
            "quality_samples_ms": [round(float(x), 4) for x in quality_samples],
        },
        "ranking": rows_sorted,
    }
    (args.out_dir / "fast_plus_variant_sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out_dir / "fast_plus_variant_sweep_report.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows_sorted for key in row.keys() if key != "samples_ms"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: v for k, v in row.items() if k != "samples_ms"})

    print(json.dumps({"config": report["config"], "top": top}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




