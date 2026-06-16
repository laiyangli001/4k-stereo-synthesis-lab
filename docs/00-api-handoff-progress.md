# Handoff - 2026-06-16

## Project

Repo:

```text
D:\AI_2D_to_3D\4.LC700X_Desktop2Stereo\4k-stereo-synthesis-lab
```

GitHub:

```text
https://github.com/laiyangli001/4k-stereo-synthesis-lab
```

Latest pushed commit at handoff:

```text
f3569c6 docs: update handoff with auto mode plan
```

Important docs:

- `docs/07-depth-backend-benchmark-2026-06-16.md`
- `docs/08-synthesis-optimization-log-2026-06-16.md`
- `docs/10-rtx3090-fused-synthesis-results-2026-06-17.md`
- `docs/11-visual-regression-guide.md`
- `docs/12-openxr-stereo-runtime-plan.md`
- `docs/13-realtime-stereo-parameter-guide.md`
- This file: `docs/00-api-handoff-progress.md`

Note:

- `docs/06-api-handoff-progress.md` contains useful history but may display mojibake in some viewers. Prefer this UTF-8 handoff plus docs `07/08`.

## Hard Boundaries

Do not change these unless the user explicitly approves a separate quality evaluation:

- Do not lower depth inference resolution.
- Do not change `294x518` / `depth_resolution=518` for the current Base model path.
- Do not change RGB resize mode, antialias behavior, or ImageNet normalize semantics.
- Do not change the depth model as a "performance optimization".
- Do not write model artifacts into Desktop2Stereo's model directory.
- Do not upload runtime artifacts:
  - `models/`
  - `outputs/`
  - `python3/`
  - `python-cu13/`
  - `downloads/`
  - `.codegraph/`
  - `4K.jpg`

## Current Status

### 2026-06-17 RTX 3090 Update

RTX 3090 formal fused-synthesis pass is complete. Current 4K `quality_4k` no longer has synthesis as the primary bottleneck for the Base model.

Latest detailed result doc:

```text
docs/10-rtx3090-fused-synthesis-results-2026-06-17.md
```

Visual regression guide:

```text
docs/11-visual-regression-guide.md
```

Final Base Native TensorRT + `quality_4k` + 2 layers on RTX 3090:

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 6.088 | 5.601 | 11.691 | 85.54 |
| Full-SBS | 5.931 | 5.892 | 11.823 | 84.58 |

Final Large Native TensorRT + `quality_4k` + 2 layers on RTX 3090:

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 12.441 | 5.561 | 18.003 | 55.55 |
| Full-SBS | 12.543 | 5.819 | 18.363 | 54.46 |

Current fused synthesis backends:

- `triton_warp_composite2`
- `triton_occlusion_radius2`
- `triton_radius3`
- `triton_half_sbs` for Half-SBS output
- `triton_full_sbs` for Full-SBS output
- `triton_half_tab` for Half-TAB output
- `triton_full_tab` for Full-TAB output
- `triton_depth_map` for depth-map output
- `triton_anaglyph` for anaglyph display output
- `triton_interleaved` for row-interleaved display output
- `triton_leia` for column-interleaved Leia-style display output

Current core output formats:

- `half_sbs`
- `full_sbs`
- `half_tab`
- `full_tab`
- `mono`
- `depth_map`
- `anaglyph`
- `interleaved`
- `leia`

Realtime parameter status:

- P0 GUI/API parameters are implemented with defaults: `depth_strength`, `convergence`, `ipd`, `max_shift_ratio`, `temporal_strength`, `auto_reset_temporal`, `scene_reset_threshold`, `reset_cooldown_frames`, `edge_dilation`, `edge_threshold`, and OpenXR roll-adaptive core parameters.
- P1 still-image/HQ parameters are implemented where executable: `foreground_scale`, `depth_antialias_strength`, `cross_eyed`, and `anaglyph_method`.
- `synthetic_view` should be represented by `backend=fast/quality_4k/hq_4k` or OpenXR mode, not a separate no-op parameter.
- Offline P2 items are intentionally excluded from realtime: offline video lookahead, TransNetV2 scene detection, and HDR/video codec pipeline.
- GUI mode taxonomy is now: `Auto`, `Cinema`, `Game / Low Latency`, `Still Image / HQ`, and `Debug / Export`.
- `Auto Mode` is the recommended GUI default. It should classify by frame motion, still duration, foreground process/window type, frame-rate/latency pressure, OpenXR state, and user export/debug actions.
- Auto mode must use hysteresis: require consecutive frames before switching, hold mode for `2-5` seconds, blend parameters instead of jumping, and quickly downgrade to `Game / Low Latency` on scene reset or violent motion.
- `Still Image / HQ` is for static images, screenshots, paused frames, and single-image 2D-to-3D generation. It should disable `temporal` and `auto_reset_temporal`.
- Details: `docs/13-realtime-stereo-parameter-guide.md`

4K is the stress/performance target, not a functional input-size limit. The output API and fast synthesis path are covered by tests for 720p, 1080p, portrait, and odd-size inputs. Unsupported Triton cases fall back to PyTorch instead of restricting input resolution.

OpenXR note: the local environment has pyopenxr available as the `xr` module, but this lab does not yet include a full OpenXR session/swapchain runtime. The rotation-adaptive stereo core has been added in `src/stereo_lab/openxr_render.py`; it accepts arbitrary `screen_roll` angles in radians and should be used by a future runtime integration instead of fixed SBS output. `scripts/generate_openxr_stereo_preview.py` can generate roll-adaptive left/right preview images from RGB+depth inputs.

`depth_map` is the matched output depth repeated to RGB channels. With `debug_output=True`, the exact tensor is also available as `debug_info["output_depth"]`.

`mono` remains a direct left-eye return and does not need a Triton kernel.

Desktop2Stereo also has `Anaglyph`, `Interleaved`, and `Leia`. The lab now exposes these as file/API post-processing formats from already synthesized left/right tensors:

- `anaglyph`: default red/cyan mode uses red channel from left and green/blue channels from right. Additional `anaglyph_method` values are available for compatibility: `green_magenta`, `amber_blue`, and `gray`.
- `interleaved`: even rows from left, odd rows from right.
- `leia`: even columns from left, odd columns from right.

This is not full Desktop2Stereo viewer shader parity because the reference viewer performs display-time shader DIBR from RGB+depth for these modes.

Confirmed in benchmark JSON under:

```text
formats.<format>.synthesis_debug.warp_composite_backend
formats.<format>.synthesis_debug.occlusion_mask_backend
formats.<format>.synthesis_debug.hole_fill_backend
formats.<format>.synthesis_debug.sbs_backend
```

Fused control:

- `StereoConfig(fused=True)` enables fused paths by default.
- `StereoConfig(fused=False)` forces PyTorch fallback.
- CLI scripts support `--no-fused` for comparison/fallback.
- Environment variable `STEREO_LAB_DISABLE_TRITON=1` disables Triton fused paths globally.

Important:

- First Triton execution includes compile overhead. Do not use visual regression script smoke timing for performance claims.
- Use `bench_end_to_end_4k.py` for performance claims.
- `profile_synthesis_4k.py` reports `end_to_end_mean_ms` for real fused `synthesize_stereo` timing and `breakdown_mean_ms` for manual unfused component attribution.
- `generate_visual_regression_set.py` now accepts `--onnx` and `--trt-engine`; pass them explicitly for Large or other non-default engines.

Latest verification:

```text
61 passed, 1 warning
compileall syntax ok
```

Latest key outputs:

```text
outputs/rtx3090_end_to_end_base_quality_full_sbs_triton.json
outputs/rtx3090_end_to_end_large_quality_half_sbs_fused.json
outputs/stereo_output_formats_triton_smoke.json
outputs/stereo_display_formats_triton_smoke.json
outputs/visual_regression/rtx3090_base_quality_full_sbs_triton
outputs/visual_regression/rtx3090_large_quality_half_sbs_fused
```

### Depth Backend

Current recommended backend order:

```text
Native TensorRT -> ONNX CUDA DLPack -> ONNX CUDA IOBinding -> PyTorch CUDA
```

Best current Base @ 518 result on RTX 2060:

| Backend | Mean ms | Median ms | Mean FPS |
|---|---:|---:|---:|
| Native TensorRT | 16.425 | 15.641 | 60.88 |
| ONNX CUDA DLPack | 27.706 | 26.287 | 36.09 |
| ONNX CUDA IOBinding | 35.905 | 35.308 | 27.85 |
| PyTorch CUDA | 42.324 | 41.060 | 23.63 |

Source:

```text
outputs/depth_backend_final_compare_4k/depth_backend_bench.json
docs/07-depth-backend-benchmark-2026-06-16.md
```

Important implementation files:

- `src/stereo_lab/depth_provider.py`
- `src/stereo_lab/depth_onnx_provider.py`
- `src/stereo_lab/depth_trt_provider.py`
- `src/stereo_lab/depth_trt_native_provider.py`

Native TensorRT engine path:

```text
models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.trt
```

Current judgment:

- Depth backend is no longer the main bottleneck on RTX 2060 when using Native TensorRT Base @ 518.
- Provider/session/engine must stay resident. Do not construct providers per frame.
- ONNX CUDA DLPack is the preferred ONNX fallback.
- CUDA Graph probe exists but was slower than normal native TensorRT, so it is not default.

### Distill-Any-Depth-Large

Model:

```text
xingyang1/Distill-Any-Depth-Large-hf
input: 1x3x294x518
```

RTX 2060 result:

| Backend | Mean ms | Mean FPS |
|---|---:|---:|
| Large Native TensorRT | 33.616 | 29.75 |
| Large ONNX CUDA DLPack | 62.590 | 15.98 |

Current judgment:

- Large @ 518 is useful for offline/quality evaluation on RTX 2060.
- It is not a 4K 60 FPS end-to-end target on RTX 2060.
- Must be retested on RTX 3090 / RTX 5070.

### Stereo Synthesis

Implemented backends:

- `fast`: Desktop2Stereo-like depth-shift baseline.
- `quality_4k`: 2-layer occlusion-aware prototype.
- `hq_4k`: at least 3 layers, still prototype.

Key files:

- `src/stereo_lab/synthesis.py`
- `src/stereo_lab/baseline_shift.py`
- `src/stereo_lab/layers.py`
- `src/stereo_lab/occlusion.py`
- `src/stereo_lab/hole_fill.py`
- `src/stereo_lab/output.py`

Latest 4K end-to-end on RTX 2060, Native TensorRT + `quality_4k`:

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.593 | 54.697 | 71.292 | 14.03 |
| Full-SBS | 15.550 | 54.482 | 70.034 | 14.28 |

Source:

```text
outputs/end_to_end_4k/quality_native_grid_components.json
```

Important caveat:

- RTX 2060 is the entry-level 4K baseline, not the final high-end ceiling.
- RTX 3090 / RTX 5070 must rerun formal benchmarks before making high-end performance claims.

## Completed Optimizations

Detailed optimization history:

```text
docs/08-synthesis-optimization-log-2026-06-16.md
```

Highlights:

- TensorRT DLL PATH discovery.
- Provider/session benchmark separation.
- ONNX CUDA IOBinding.
- Native TensorRT `data_ptr()` provider.
- Native TensorRT output dtype detection from engine.
- Native TensorRT output preallocation.
- ONNX CUDA DLPack lower-copy fallback.
- Safe preprocess cache with identical depth output.
- Smart ONNX `--dtype auto` export probe.
- Base grid cache and grid component cache.
- Removed redundant baseline warp from `quality_4k`.
- Hole fill kernel cache and left/right batch fill.
- Separable box blur.
- In-place composite.
- In-place depth edge accumulation.
- Rejected PyTorch batch warp because it was slower at 4K.
- Rejected Native TensorRT CUDA Graph as default because it was slower in current form.

## Visual Regression

New fixed visual regression script:

```text
scripts/generate_visual_regression_set.py
```

Example:

```powershell
.\python3\python.exe -B scripts\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\4k_native_base_quality
```

Verified real 4K output:

```text
outputs/visual_regression/4k_native_base_quality/
```

Key files:

- `input_rgb.png`
- `used_depth.png`
- `baseline_left.png`
- `baseline_right.png`
- `baseline_half_sbs.png`
- `baseline_full_sbs.png`
- `quality_4k_left.png`
- `quality_4k_right.png`
- `quality_4k_half_sbs.png`
- `quality_4k_full_sbs.png`
- `quality_4k_occlusion_mask.png`
- `baseline_vs_quality_4k_*_absdiff.png`
- `contact_sheet.png`
- `contact_sheet_labeled.png`
- `visual_regression_report.json`

Important:

- `generate_visual_regression_set.py` timing is only a coarse smoke value.
- Use `profile_synthesis_4k.py` and `bench_end_to_end_4k.py` for performance claims.
- The assistant can visually inspect generated images with the local image viewer.

## Verification Commands

Syntax:

```powershell
.\python3\python.exe -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax ok', len(files), 'files')"
```

Tests:

```powershell
.\python3\python.exe -B -m pytest -q
```

Current latest result:

```text
61 passed, 1 warning
compileall syntax ok
```

Synthesis profile:

```powershell
.\python3\python.exe -B scripts\profile_synthesis_4k.py --rgb 4K.jpg --out outputs\synthesis_profile_4k\<name>.json --backend quality_4k --layers 2 --output-format half_sbs --iters 5
```

End-to-end:

```powershell
.\python3\python.exe -B scripts\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\end_to_end_4k\<name>.json --warmup 2 --iters 5 --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```

Visual regression:

```powershell
.\python3\python.exe -B scripts\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\<name>
```

## Recommended Next Steps

1. Implement GUI-side `Auto Mode` state machine:
   - `frame_motion_score`
   - `scene_cut_score`
   - `still_duration`
   - `foreground_process`
   - `fullscreen_state`
   - `openxr_active`
   - `user_export_action`
2. Map `Auto Mode` output to `StereoModePreset`, `StereoConfig`, and `OpenXRRenderConfig`.
3. Generate visual regression sets for representative Cinema, Game / Low Latency, Still Image / HQ, and Debug / Export samples.
4. Use visual regression to finalize default preset values.
5. Optimize `hole_fill` only with visual regression checks.
6. Re-run formal benchmarks on RTX 3090 / RTX 5070 when available:
   - `bench_depth_backends.py`
   - `profile_synthesis_4k.py`
   - `bench_end_to_end_4k.py`
   - `generate_visual_regression_set.py`
7. Add true iw3 same-scene comparison later, after fixed RGB/depth/output format is locked.

## Current Bottleneck

Depth is usable. The main bottleneck is now stereo synthesis:

- `hole_fill`
- `warp_layers`
- memory bandwidth in 4K PyTorch tensor operations

The next major jump likely requires:

- fused CUDA/shader warp + composite, or
- a semantically equivalent lower-bandwidth hole fill, or
- stronger hardware such as RTX 3090 / RTX 5070 for the current prototype path.

