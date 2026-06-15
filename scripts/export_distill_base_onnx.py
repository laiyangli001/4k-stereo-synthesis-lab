from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def default_output_path() -> Path:
    return (
        ROOT
        / "models"
        / "models--lc700x--Distill-Any-Depth-Base-hf"
        / "model_fp16_294x518.onnx"
    )


def main() -> None:
    print("[1/7] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(default_output_path()))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--no-force-download", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.parent.parent

    print("[2/7] importing torch and transformers ...", flush=True)
    import torch
    from transformers import AutoModelForDepthEstimation

    model_id = "lc700x/Distill-Any-Depth-Base-hf"
    dtype = torch.float16
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print(f"[info] model: Distill-Any-Depth-Base", flush=True)
    print(f"[info] model id: {model_id}", flush=True)
    print(f"[info] onnx input: 1x3x{args.height}x{args.width}", flush=True)
    print(f"[info] dtype: fp16", flush=True)
    print(f"[info] device: {device}", flush=True)
    print(f"[info] cache dir: {cache_dir}", flush=True)
    print(f"[info] output: {output_path}", flush=True)
    print(f"[info] force download: {not args.no_force_download}", flush=True)

    print("[3/7] downloading/loading model from Hugging Face ...", flush=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        dtype=dtype,
        cache_dir=str(cache_dir),
        weights_only=True,
        force_download=not args.no_force_download,
    ).to(device)

    print("[4/7] converting model to fp16 eval mode ...", flush=True)
    model.half()
    model.eval()

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
