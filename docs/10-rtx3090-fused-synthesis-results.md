# RTX 3090 Fused Synthesis Results - 2026-06-17

## Hardware

- GPU: NVIDIA GeForce RTX 3090 24GB
- Driver: 610.47
- Input: `4K.jpg`, 3840x2160
- Base depth model: `lc700x/Distill-Any-Depth-Base-hf`
- Large depth model: `xingyang1/Distill-Any-Depth-Large-hf`
- Depth backend: Native TensorRT, fp16, 294x518

## Implemented Optimizations

### Depth

Base Native TensorRT is resident and remains the recommended depth backend.

### Synthesis

Three optional fused Triton paths are now available:

- `triton_radius3` hole fill for CUDA float32, `B x 3 x H x W` image, `B x 1 x H x W` mask, `radius=3`, `strength=1.0`.
- `triton_warp_composite2` for CUDA float32, 2-layer, symmetric, single-frame `quality_4k` warp + composite.
- `triton_occlusion_radius2` for CUDA float32, single-frame occlusion mask generation with `edge_threshold=0.04` and `dilation=2`.

Both paths are guarded by strict shape/type/config checks and fall back to the original PyTorch implementation when unsupported.

Fused paths are enabled by default and can be disabled with:

- `StereoConfig(fused=False)`
- CLI flag `--no-fused`
- Environment variable `STEREO_LAB_DISABLE_TRITON=1`

## End-to-End Results

### Base

Command:

```powershell
.\python3\python.exe -B scripts\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\rtx3090_end_to_end_base_quality_occlusion_fused_repeat.json --warmup 5 --iters 20 --backend quality_4k --layers 2 --depth-backend tensorrt_native --onnx models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.onnx --trt-engine models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.trt --output-format half_sbs --output-format full_sbs
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 6.253 | 6.088 | 12.342 | 81.02 |
| Full-SBS | 6.884 | 6.088 | 12.973 | 77.08 |

### Large

Command:

```powershell
.\python3\python.exe -B scripts\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\rtx3090_end_to_end_large_quality_final_fused.json --warmup 3 --iters 10 --backend quality_4k --layers 2 --depth-backend tensorrt_native --onnx models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.onnx --trt-engine models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.trt --output-format half_sbs --output-format full_sbs
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 13.588 | 7.732 | 21.322 | 46.90 |
| Full-SBS | 12.845 | 7.441 | 20.287 | 49.29 |

The Large model is visually usable for quality comparison on RTX 3090, but it is not a 4K 60 FPS end-to-end target in this current path because depth inference is the bottleneck.

Provider metadata for explicit Large ONNX/TRT paths is inferred from the Hugging Face cache-style directory name, so JSON reports should show `xingyang1/Distill-Any-Depth-Large-hf` instead of the Base default.

## Progression

| Version | Half-SBS FPS | Full-SBS FPS |
|---|---:|---:|
| RTX 3090 initial PyTorch synthesis | 40.47 | 41.32 |
| Triton hole fill only | 46.57 | 50.50 |
| Triton warp+composite + Triton hole fill | 74.55 | 72.58 |
| Triton warp+composite + occlusion + hole fill | 81.02 | 77.08 |

## Visual Regression

Latest set:

```text
outputs/visual_regression/rtx3090_base_quality_occlusion_fused
```

Large set:

```text
outputs/visual_regression/rtx3090_large_engine_quality_final_fused
```

Compared with `rtx3090_base_quality_triton_holefill`:

- `used_depth.png`: identical
- `quality_4k_occlusion_mask.png`: identical
- `quality_4k_left.png`: max uint8 diff 1, mean 0.000501
- `quality_4k_right.png`: max uint8 diff 1, mean 0.000498
- `quality_4k_full_sbs.png`: max uint8 diff 1, mean 0.000499
- `quality_4k_half_sbs.png`: max uint8 diff 1, mean 0.017743

The fused warp path has tiny interpolation/rounding differences from `torch.grid_sample`; no structural mask/depth changes were observed.

Compared with `rtx3090_base_quality_final_fused`, the fused occlusion path produced byte-identical output:

- `used_depth.png`: identical
- `quality_4k_occlusion_mask.png`: identical
- `quality_4k_left.png`: identical
- `quality_4k_right.png`: identical
- `quality_4k_half_sbs.png`: identical
- `quality_4k_full_sbs.png`: identical

Manual visual inspection of `contact_sheet_labeled.png` for both Base and Large found:

- Input/depth/left/right/SBS outputs are present and complete.
- Occlusion masks primarily follow object/background boundaries.
- Difference images are concentrated around edges and depth discontinuities.
- No obvious full-frame corruption, black holes, broken SBS seam, or large tearing was observed.

## Important Notes

- Visual regression script timings can include first-run Triton compile overhead. Use `bench_end_to_end_4k.py` for performance claims.
- `generate_visual_regression_set.py` accepts `--onnx` and `--trt-engine`; pass these explicitly for Large or any non-default engine. Otherwise the default Base engine may be used.
- Use `--no-fused` on benchmark/profile/visual-regression scripts to force the PyTorch fallback path for A/B comparison.
- `profile_synthesis_4k.py` now reports both:
  - `end_to_end_mean_ms`, which uses `synthesize_stereo` and includes fused backends when available.
  - `breakdown_mean_ms`, which uses the manual unfused breakdown path for component attribution.
- Fused warp/composite is not used for `hq_4k` 3+ layers, asymmetric mode, CPU, non-float32 tensors, or unsupported shapes.
- Fused occlusion is not used for non-default threshold/dilation, CPU, non-float32 tensors, or unsupported shapes.
- `bench_end_to_end_4k.py` records `synthesis_debug.warp_composite_backend`, `synthesis_debug.occlusion_mask_backend`, and `synthesis_debug.hole_fill_backend`; the latest Base run reports `triton_warp_composite2`, `triton_occlusion_radius2`, and `triton_radius3`.

## Next Verification Targets

1. Re-run the same benchmark after a fresh Python process warmup to confirm stability.
2. Run Base and Large depth model end-to-end comparisons with fused synthesis.
3. Re-test on RTX 5070 or final target deployment hardware.
