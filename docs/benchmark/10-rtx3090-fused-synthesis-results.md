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

Eleven optional fused Triton paths are now available:

- `triton_radius3` hole fill for CUDA float32, `B x 3 x H x W` image, `B x 1 x H x W` mask, `radius=3`, `strength=1.0`.
- `triton_warp_composite2` for CUDA float32, 2-layer, symmetric, single-frame `quality_4k` warp + composite.
- `triton_occlusion_radius2` for CUDA float32, single-frame occlusion mask generation with `edge_threshold=0.04` and `dilation=2`.
- `triton_half_sbs` for CUDA float32, single-frame `B=1, C=3`, even-width Half-SBS resize + pack.
- `triton_full_sbs` for CUDA float32, single-frame `B=1, C=3`, Full-SBS copy + pack.
- `triton_half_tab` for CUDA float32, single-frame `B=1, C=3`, even-height Half-TAB resize + pack.
- `triton_full_tab` for CUDA float32, single-frame `B=1, C=3`, Full-TAB copy + pack.
- `triton_depth_map` for CUDA float32, single-frame `B=1, C=1` depth repeated to RGB channels.
- `triton_anaglyph` for CUDA float32, single-frame `B=1, C=3`, red channel from left and green/blue channels from right.
- `triton_interleaved` for CUDA float32, single-frame `B=1, C=3`, even rows from left and odd rows from right.
- `triton_leia` for CUDA float32, single-frame `B=1, C=3`, even columns from left and odd columns from right.

All paths are guarded by strict shape/type/config checks and fall back to the original PyTorch implementation when unsupported.

RTX 3090 4K probe timing for Full-SBS showed `triton_full_sbs` and `torch.cat` within probe noise, with `triton_full_sbs` numerically identical:

| Full-SBS pack path | Mean ms | Min ms | Max ms |
|---|---:|---:|---:|
| `torch_cat` | 0.579 | 0.520 | 1.422 |
| `triton_full_sbs` | 0.597 | 0.544 | 1.309 |

Fused paths are enabled by default and can be disabled with:

- `StereoConfig(fused=False)`
- CLI flag `--no-fused`
- Environment variable `STEREO_LAB_DISABLE_TRITON=1`

## Stereo Output Formats

The core output API now covers the file/API formats that map directly from synthesized left/right/depth tensors:

- `half_sbs`: same size as input, left/right resized to half width and packed side by side.
- `full_sbs`: full-width left/right packed side by side.
- `half_tab`: same size as input, left/right resized to half height and packed top/bottom.
- `full_tab`: full-height left/right packed top/bottom.
- `mono`: left-eye output, useful for fallback and debugging.
- `depth_map`: matched output depth repeated to RGB channels, useful for debug/export. In `debug_output=True`, the exact tensor is also available as `debug_info["output_depth"]`.
- `anaglyph`: red channel from left, green/blue channels from right.
- `interleaved`: even rows from left, odd rows from right.
- `leia`: even columns from left, odd columns from right.

`mono` remains a direct left-eye return and does not need a Triton kernel.

Desktop2Stereo also exposes `Anaglyph`, `Interleaved`, and `Leia` as display/viewer shader modes. The lab implementation is intentionally a file/API post-processing path from already synthesized left/right tensors, not full viewer shader DIBR parity.

Smoke verification for these three display formats:

```text
outputs/stereo_display_formats_triton_smoke.json
```

Confirmed `synthesis_debug.sbs_backend` values:

- `anaglyph`: `triton_anaglyph`
- `interleaved`: `triton_interleaved`
- `leia`: `triton_leia`

## End-to-End Results

### Base

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\rtx3090_end_to_end_base_quality_full_sbs_triton.json --warmup 5 --iters 20 --backend quality_4k --layers 2 --depth-backend tensorrt_native --onnx models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.onnx --trt-engine models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.trt --output-format half_sbs --output-format full_sbs
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 6.088 | 5.601 | 11.691 | 85.54 |
| Full-SBS | 5.931 | 5.892 | 11.823 | 84.58 |

This latest Base run includes `triton_half_sbs` for Half-SBS output packing:

```text
outputs/rtx3090_end_to_end_base_quality_full_sbs_triton.json
```

Backend fields:

- Half-SBS: `triton_warp_composite2`, `triton_occlusion_radius2`, `triton_radius3`, `triton_half_sbs`
- Full-SBS: `triton_warp_composite2`, `triton_occlusion_radius2`, `triton_radius3`, `triton_full_sbs`

### Large

Command:

```powershell
.\python3\python.exe -B scripts\benchmark\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\rtx3090_end_to_end_large_quality_half_sbs_fused.json --warmup 5 --iters 20 --backend quality_4k --layers 2 --depth-backend tensorrt_native --onnx models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.onnx --trt-engine models\models--xingyang1--Distill-Any-Depth-Large-hf\model_fp16_294x518.trt --output-format half_sbs --output-format full_sbs
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 12.441 | 5.561 | 18.003 | 55.55 |
| Full-SBS | 12.543 | 5.819 | 18.363 | 54.46 |

The Large model is visually usable for quality comparison on RTX 3090 and now lands near 55 FPS in this path. It is still below the 4K 60 FPS target because depth inference is the bottleneck.

Provider metadata for explicit Large ONNX/TRT paths is inferred from the Hugging Face cache-style directory name, so JSON reports should show `xingyang1/Distill-Any-Depth-Large-hf` instead of the Base default.

## Progression

| Version | Half-SBS FPS | Full-SBS FPS |
|---|---:|---:|
| RTX 3090 initial PyTorch synthesis | 40.47 | 41.32 |
| Triton hole fill only | 46.57 | 50.50 |
| Triton warp+composite + Triton hole fill | 74.55 | 72.58 |
| Triton warp+composite + occlusion + hole fill | 81.02 | 77.08 |
| Triton warp+composite + occlusion + hole fill + Half-SBS | 83.14 | 86.78 |
| Triton warp+composite + occlusion + hole fill + Half/Full-SBS | 85.54 | 84.58 |

## Visual Regression

Latest set:

```text
outputs/visual_regression/rtx3090_base_quality_occlusion_fused
```

Latest Half-SBS fused set:

```text
outputs/visual_regression/rtx3090_base_quality_half_sbs_fused
```

Latest Full-SBS Triton set:

```text
outputs/visual_regression/rtx3090_base_quality_full_sbs_triton
```

Large set:

```text
outputs/visual_regression/rtx3090_large_quality_half_sbs_fused
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

Compared with `rtx3090_base_quality_occlusion_fused`, the fused Half-SBS path preserved all non-Half-SBS outputs byte-identically:

- `quality_4k_left.png`: identical
- `quality_4k_right.png`: identical
- `quality_4k_full_sbs.png`: identical
- `quality_4k_occlusion_mask.png`: identical
- `baseline_half_sbs.png`: max uint8 diff 1, mean 0.000016
- `quality_4k_half_sbs.png`: max uint8 diff 1, mean 0.000026

The Half-SBS differences are limited to 1/255-level rounding differences from the fused linear resize path.

The same pattern holds for Large when compared with `rtx3090_large_engine_quality_final_fused`:

- `used_depth.png`: identical
- `quality_4k_left.png`: identical
- `quality_4k_right.png`: identical
- `quality_4k_full_sbs.png`: identical
- `quality_4k_occlusion_mask.png`: identical
- `baseline_half_sbs.png`: max uint8 diff 1, mean 0.000015
- `quality_4k_half_sbs.png`: max uint8 diff 1, mean 0.000033

Compared with `rtx3090_base_quality_half_sbs_fused`, the default `triton_full_sbs` path produced byte-identical visual regression outputs:

- `used_depth.png`: identical
- `quality_4k_left.png`: identical
- `quality_4k_right.png`: identical
- `quality_4k_half_sbs.png`: identical
- `quality_4k_full_sbs.png`: identical
- `quality_4k_occlusion_mask.png`: identical

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
- `bench_end_to_end_4k.py` records `synthesis_debug.warp_composite_backend`, `synthesis_debug.occlusion_mask_backend`, `synthesis_debug.hole_fill_backend`, and `synthesis_debug.sbs_backend`; output backends include `triton_half_sbs`, `triton_full_sbs`, `triton_half_tab`, `triton_full_tab`, `triton_depth_map`, `triton_anaglyph`, `triton_interleaved`, `triton_leia`, and `torch_mono_left`.

## Next Verification Targets

1. Re-run the same benchmark after a fresh Python process warmup to confirm stability.
2. Run Base and Large depth model end-to-end comparisons with fused synthesis.
3. Re-test on RTX 5070 or final target deployment hardware.
