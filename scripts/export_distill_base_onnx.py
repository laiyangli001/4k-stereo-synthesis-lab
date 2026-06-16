from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORCE_FP32_KEYWORDS = (
    "depthpro",
    "zoedepth",
    "infinidepth-large",
)

def default_output_path(model_id: str, dtype_name: str, height: int, width: int) -> Path:
    repo_dir = "models--" + model_id.replace("/", "--")
    return (
        ROOT
        / "models"
        / repo_dir
        / f"model_{dtype_name}_{height}x{width}.onnx"
    )


def choose_export_dtype(model_id: str, device, requested: str):
    import torch

    if requested == "fp16":
        return torch.float16, "fp16", "requested fp16"
    if requested == "fp32":
        return torch.float32, "fp32", "requested fp32"
    model_lower = model_id.lower()
    if device.type != "cuda":
        return torch.float32, "fp32", "auto: non-CUDA device"
    if any(keyword in model_lower for keyword in FORCE_FP32_KEYWORDS):
        return torch.float32, "fp32", "auto: model requires fp32"
    return torch.float16, "fp16", "auto: CUDA default"


def _extract_depth_output(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, dict) and "predicted_depth" in output:
        return output["predicted_depth"]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"unsupported model output type: {type(output).__name__}")


def probe_model_dtype(model, *, device, dtype, height: int, width: int) -> tuple[bool, str]:
    import torch

    dummy = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    try:
        model.eval()
        with torch.no_grad():
            output = model(pixel_values=dummy)
        depth = _extract_depth_output(output).detach().float()
        if depth.numel() == 0:
            return False, "empty output"
        if not torch.isfinite(depth).all().item():
            return False, "output contains NaN or Inf"
        abs_max = float(depth.abs().max().item())
        value_range = float((depth.max() - depth.min()).abs().item())
        if abs_max == 0.0:
            return False, "output is all zero"
        if value_range < 1e-7:
            return False, f"output dynamic range too small: {value_range:.3e}"
        return True, f"ok: shape={tuple(depth.shape)} abs_max={abs_max:.6g} range={value_range:.6g}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def load_model_for_dtype(AutoModelForDepthEstimation, model_id: str, *, dtype, device, cache_dir: Path, force_download: bool):
    import torch

    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        dtype=dtype,
        cache_dir=str(cache_dir),
        weights_only=True,
        force_download=force_download,
    ).to(device)
    if dtype == torch.float16:
        model.half()
    else:
        model.float()
    model.eval()
    return model


def main() -> None:
    print("[1/7] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--model-id", default="lc700x/Distill-Any-Depth-Base-hf")
    parser.add_argument("--model-name", default="Distill-Any-Depth-Base")
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--no-force-download", action="store_true")
    args = parser.parse_args()

    print("[2/7] importing torch and transformers ...", flush=True)
    import torch
    from transformers import AutoModelForDepthEstimation

    model_id = args.model_id
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype, dtype_name, dtype_reason = choose_export_dtype(model_id, device, args.dtype)
    output_path = Path(args.output) if args.output else default_output_path(model_id, dtype_name, args.height, args.width)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.parent.parent

    print(f"[info] model: {args.model_name}", flush=True)
    print(f"[info] model id: {model_id}", flush=True)
    print(f"[info] onnx input: 1x3x{args.height}x{args.width}", flush=True)
    print(f"[info] dtype: {dtype_name}", flush=True)
    print(f"[info] dtype reason: {dtype_reason}", flush=True)
    print(f"[info] device: {device}", flush=True)
    print(f"[info] cache dir: {cache_dir}", flush=True)
    print(f"[info] output: {output_path}", flush=True)
    print(f"[info] force download: {not args.no_force_download}", flush=True)

    print("[3/7] downloading/loading model from Hugging Face ...", flush=True)
    model = load_model_for_dtype(
        AutoModelForDepthEstimation,
        model_id,
        dtype=dtype,
        device=device,
        cache_dir=cache_dir,
        force_download=not args.no_force_download,
    )

    print(f"[4/7] probing {dtype_name} export dtype ...", flush=True)
    ok, probe_reason = probe_model_dtype(model, device=device, dtype=dtype, height=args.height, width=args.width)
    print(f"[info] {dtype_name} probe: {probe_reason}", flush=True)
    if args.dtype == "auto" and not ok and dtype == torch.float16:
        print("[warn] fp16 probe failed; falling back to fp32", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dtype = torch.float32
        dtype_name = "fp32"
        dtype_reason = f"auto: fp16 probe failed ({probe_reason}); fallback fp32"
        output_path = Path(args.output) if args.output else default_output_path(model_id, dtype_name, args.height, args.width)
        cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.parent.parent
        model = load_model_for_dtype(
            AutoModelForDepthEstimation,
            model_id,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
            force_download=False,
        )
        ok, probe_reason = probe_model_dtype(model, device=device, dtype=dtype, height=args.height, width=args.width)
        print(f"[info] fp32 probe: {probe_reason}", flush=True)
    if not ok:
        raise RuntimeError(f"export dtype probe failed for {dtype_name}: {probe_reason}")
    print(f"[info] final dtype: {dtype_name}", flush=True)
    print(f"[info] final dtype reason: {dtype_reason}", flush=True)
    print(f"[info] final output: {output_path}", flush=True)

    print("[5/7] creating dummy input ...", flush=True)
    dummy_input = torch.randn(1, 3, args.height, args.width, device=device, dtype=dtype)

    print("[6/7] exporting ONNX with Desktop2Stereo-compatible settings ...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[7/7] ONNX written: {output_path}", flush=True)
    print(f"[info] size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
