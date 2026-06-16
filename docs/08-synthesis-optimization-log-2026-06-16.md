# Optimization Log - 2026-06-16

## Scope

This document records optimization-related work across the whole 4K stereo lab, including workflow, depth runtime, TensorRT / ONNX backends, and stereo synthesis.

Strict boundary:

- Did not change the depth model.
- Did not change inference resolution.
- Did not change RGB resize mode, antialias behavior, or normalization semantics.
- Did not change ONNX / TensorRT export parameters.
- Did not change the generated depth values.

The synthesis changes are limited to stereo generation after RGB + depth are already available.

## Optimization Timeline

| Commit | Area | Summary |
|---|---|---|
| `dfde924` | workflow/report | Added real input comparison reports and visible compare helpers. |
| `5c5ac7b` | depth workflow | Added Distill depth pipeline, auto-depth mode, ONNX export entry, visible bat helpers. |
| `b289d3d` | depth runtime | Added NVIDIA depth workflow, ONNX CUDA provider, CUDA 13 experiment environment, batch depth reports. |
| `2a2bb30` | benchmark/runtime | Added TensorRT EP, DLL probing, provider/session benchmark separation, python3 vs python-cu13 comparison. |
| `2f1ab45` | depth runtime/synthesis | Added native TensorRT, ONNX DLPack, smart ONNX dtype auto, Large model benchmark, 4K end-to-end/profile scripts, initial synthesis safe optimizations. |
| `49da1fc` | synthesis | Reduced quality synthesis allocations. |
| `1e2a171` | synthesis | Reduced depth edge allocations. |
| `5866878` | docs | Added this optimization log. |
| `85166be` | synthesis | Cache grid components for horizontal warp and avoid cloning the full base grid. |
| `pending` | evaluation | Add fixed visual regression set generator for baseline vs quality_4k. |

## Workflow And Evaluation Optimizations

These changes improved repeatability and debugging. They are not speed optimizations by themselves, but they are required to avoid false conclusions.

### Real Input Comparison Reports

Commit:

```text
dfde924 feat: add real input comparison reports
```

Files:

- `scripts/compare_methods.py`
- `scripts/run_visible_compare_demo.bat`
- `scripts/run_visible_compare_files.bat`
- `src/stereo_lab/report.py`

What changed:

- Added real RGB/depth file comparison workflow.
- Added report metrics and visible batch helpers.
- Made it easier to compare baseline vs quality outputs on actual user images.

Why it matters:

- Prevents judging stereo quality only from synthetic/demo images.
- Makes future visual regressions easier to reproduce.

### Auto Depth And Visible Depth Pipeline

Commit:

```text
5c5ac7b feat: add distill depth pipeline
```

Files:

- `src/stereo_lab/depth_provider.py`
- `src/stereo_lab/auto_depth.py`
- `scripts/generate_depth_map.py`
- `scripts/compare_methods.py`
- `scripts/export_distill_base_onnx.py`
- `scripts/run_visible_generate_depth.bat`
- `scripts/run_visible_compare_rgb_auto_depth.bat`
- `scripts/run_visible_export_distill_base_onnx.bat`

What changed:

- Added Distill-Any-Depth-Base @ 518 pipeline.
- Added `--auto-depth` mode for comparisons when no depth image is provided.
- Added visible bat entry points for slow first-run model import/export on low-end machines.
- Kept model outputs under this project's `models/`, not Desktop2Stereo's model directory.

Why it matters:

- Enables RGB-only testing without manually preparing depth.
- Keeps long-running first import/export visible to the user.
- Keeps project artifacts isolated and excluded from Git.

### Batch Depth Reports And CUDA 13 Experiment Line

Commit:

```text
b289d3d feat: add nvidia depth workflow
```

Files:

- `scripts/batch_generate_depth_maps.py`
- `scripts/test_distill_base_onnx.py`
- `scripts/setup_cuda13_nightly_env.ps1`
- `scripts/run_cuda13_onnx_smoke.ps1`
- `src/stereo_lab/depth_onnx_provider.py`
- `src/stereo_lab/report.py`

What changed:

- Added batch depth generation and structured reports.
- Added ONNX CUDA provider smoke tests.
- Added isolated `python-cu13/` experiment environment scripts.
- Added provider/report metadata for backend, runtime, ONNX path, execution provider, and fallback reason.

Why it matters:

- Allows `python3` stable line vs `python-cu13` experiment line comparisons.
- Makes depth-provider changes traceable in JSON reports.

## Depth Runtime Optimizations

Primary reference:

```text
docs/07-depth-backend-benchmark-2026-06-16.md
```

### TensorRT DLL Path Discovery

Commit:

```text
2a2bb30 feat: add tensorrt depth backend benchmark
```

Files:

- `src/stereo_lab/depth_trt_provider.py`
- `scripts/probe_tensorrt_runtime.py`
- `scripts/bench_depth_backends.py`

Problem:

```text
TensorRT provider did not activate; active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Root cause:

```text
onnxruntime_providers_tensorrt.dll depends on nvinfer_10.dll
```

Actual DLL location:

```text
python3/Lib/site-packages/tensorrt_libs/
```

Optimization/fix:

- Auto-discover and prepend the TensorRT DLL folder to `PATH` before creating ORT TensorRT sessions.

Why it matters:

- Makes TensorRT EP actually activate instead of silently falling back to CUDA/CPU providers.

### Provider/Session Reuse Benchmark Fix

Commit:

```text
2a2bb30 feat: add tensorrt depth backend benchmark
```

Files:

- `scripts/bench_depth_backends.py`
- `scripts/compare_python_env_depth_backends.py`

Problem:

- Earlier numbers accidentally included TensorRT build/session creation in per-frame inference.
- The first TensorRT build took about `309s`.
- A later `3.4s` warm run was still session/engine load overhead, not true per-frame inference.

Fix:

- Benchmarks now separate:

```text
setup_ms     = provider/session creation + TensorRT engine build/load
warmup_ms    = first calls using the same provider/session
inference_ms = repeated inference using the same provider/session
```

Correct runtime pattern:

```python
provider.load()  # once
for frame in frames:
    depth = provider.predict(frame)
```

Why it matters:

- Prevents falsely concluding TensorRT is unusably slow.
- Establishes provider/session persistence as a hard requirement for realtime API design.

### ONNX CUDA IOBinding

Commit:

```text
2a2bb30 feat: add tensorrt depth backend benchmark
```

Files:

- `src/stereo_lab/depth_onnx_provider.py`
- `tests/test_depth_onnx_provider.py`

What changed:

- Added ONNX CUDA provider path with IOBinding.
- `distill_base_nvidia` fallback chain became:

```text
TensorRT EP -> ONNX CUDA IOBinding -> PyTorch CUDA
```

4K reference result:

| Backend | Mean ms |
|---|---:|
| TensorRT EP | 24.971 |
| ONNX CUDA IOBinding | 37.483 |
| PyTorch CUDA | 45.812 |

Why it matters:

- ONNX CUDA became a usable fallback when TensorRT fails.

### Native TensorRT Data Pointer Provider

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/depth_trt_native_provider.py`
- `scripts/build_native_tensorrt_engine.py`
- `scripts/run_visible_build_native_tensorrt_engine.bat`
- `scripts/check_native_tensorrt_consistency.py`

What changed:

- Added native TensorRT runtime provider using PyTorch CUDA tensor `data_ptr()`.
- Input and output stay on CUDA tensor memory.
- Output dtype is detected from the TensorRT engine instead of hardcoding fp16.
- Expected engine path:

```text
models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.trt
```

Why it matters:

- Avoids ORT TensorRT EP's expensive Python/ORT/numpy-style handoff.
- Mirrors the native Desktop2Stereo approach, but lives entirely inside this project.

4K result:

| Backend | Mean ms |
|---|---:|
| Native TensorRT | 19.697 |
| ONNX Runtime TensorRT EP | 25.813 |
| ONNX CUDA IOBinding | 39.217 |
| PyTorch CUDA | 41.939 |

### Native TensorRT Output Preallocation

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/depth_trt_native_provider.py`
- `scripts/check_native_tensorrt_consistency.py`

What changed:

- Reused output CUDA tensors instead of allocating `torch.empty(...)` every frame.

Correctness:

```text
outputs/native_tensorrt_consistency.json
depth_absdiff_mean = 0.0
depth_absdiff_max  = 0.0
```

4K result:

| Backend | Mean ms | Median ms | P90 ms |
|---|---:|---:|---:|
| Native TensorRT prealloc | 17.775 | 17.933 | 18.851 |

Why it matters:

- Reduces per-frame allocation overhead.
- Confirmed identical depth output in consistency check.

### ONNX CUDA DLPack Lower-Copy Fallback

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/depth_onnx_provider.py`
- `scripts/bench_depth_backends.py`

What changed:

- Added DLPack path:

```python
ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
torch.utils.dlpack.from_dlpack(output_ort)
```

4K result:

| Backend | Mean ms | Postprocess ms |
|---|---:|---:|
| ONNX CUDA DLPack | 32.215 | 1.197 |
| ONNX CUDA IOBinding | 39.379 | 5.999 |
| Native TensorRT | 15.287 | 0.970 |

Why it matters:

- ONNX CUDA fallback becomes meaningfully faster by reducing output copy overhead.
- Still slower than native TensorRT, so it is fallback, not primary path.

### Native TensorRT CUDA Graph Probe

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Status:

```text
experimental, not default
```

Result:

| Backend | Mean ms | Median ms |
|---|---:|---:|
| Native TensorRT CUDA Graph | 19.156 | 18.617 |
| Native TensorRT | 16.143 | 16.257 |

Decision:

- CUDA Graph capture runs, but current implementation is slower.
- Do not make it default.
- Revisit only after fixed input/output buffers and possibly fused preprocess are available.

Why it matters:

- Records a rejected optimization path so future agents do not repeat it blindly.

### Safe Preprocess Cache

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/depth_provider.py`
- `src/stereo_lab/depth_trt_native_provider.py`

What changed:

- Cached ImageNet mean/std CUDA tensors.
- Cached fixed model input shape calculation for a given source frame shape.

Strictly unchanged:

```text
4K RGB -> bicubic+antialias resize -> 294x518 -> ImageNet mean/std normalize -> fp16
```

Correctness:

```text
outputs/native_tensorrt_consistency_preprocess_cache.json
depth_absdiff_mean = 0.0
depth_absdiff_max  = 0.0
```

4K result:

| Backend | Mean ms | Preprocess ms |
|---|---:|---:|
| Native TensorRT + preprocess cache | 17.586 | 5.679 |

Why it matters:

- This is a safe optimization because model resolution and preprocessing semantics are unchanged.
- Gain is modest because 4K bicubic+antialias resize remains the expensive part.

### Smart ONNX Dtype Auto

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `scripts/export_distill_base_onnx.py`
- `tests/test_export_onnx_dtype.py`

What changed:

- `--dtype auto` now does a dummy forward probe.
- It tries fp16 first.
- If fp16 output has exception / NaN / Inf / all-zero / too-small dynamic range, it falls back to fp32.
- If fp32 also fails, export stops and reports the failure.

Why it matters:

- Avoids forcing fp16 on models that cannot safely run fp16.
- Supports future mixed or fp32-only models more safely.

### Distill-Any-Depth-Large 4K Limit Test

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Model:

```text
xingyang1/Distill-Any-Depth-Large-hf
input: 1x3x294x518
dtype: fp16
```

Result:

| Backend | Mean ms | Mean FPS |
|---|---:|---:|
| Large Native TensorRT | 33.616 | 29.75 |
| Large ONNX CUDA DLPack | 62.590 | 15.98 |

Decision on RTX 2060:

- Usable for offline/quality evaluation.
- Not suitable for 4K 60 FPS end-to-end SBS on RTX 2060.
- Must be retested on RTX 3090 / RTX 5070 class GPUs.

### Final Depth Backend Priority

Final 4K comparison:

```text
outputs/depth_backend_final_compare_4k/depth_backend_bench.json
```

| Backend | Mean ms | Median ms | Mean FPS |
|---|---:|---:|---:|
| Native TensorRT | 16.425 | 15.641 | 60.88 |
| ONNX CUDA DLPack | 27.706 | 26.287 | 36.09 |
| ONNX CUDA IOBinding | 35.905 | 35.308 | 27.85 |
| PyTorch CUDA | 42.324 | 41.060 | 23.63 |

Current recommended order:

```text
Native TensorRT -> ONNX CUDA DLPack -> ONNX CUDA IOBinding -> PyTorch CUDA
```

## Stereo Synthesis Optimizations

All optimizations in this section happen after depth is already produced.

### Initial 4K Synthesis Profiling And Benchmark Scripts

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `scripts/bench_end_to_end_4k.py`
- `scripts/profile_synthesis_4k.py`

What changed:

- Added 4K end-to-end benchmark:

```text
RGB -> depth -> stereo -> Half-SBS / Full-SBS
```

- Added synthesis-only stage profiling:

```text
baseline_shift / make_layers / warp_layers / composite / occlusion / hole_fill / make_sbs
```

Why it matters:

- Showed that after native TensorRT depth became fast, the bottleneck moved from depth to synthesis.
- Prevented optimizing depth further while the actual end-to-end bottleneck was already synthesis.

### Fixed Visual Regression Set Generator

Commit:

```text
pending
```

File:

- `scripts/generate_visual_regression_set.py`

What changed:

- Added a repeatable visual regression generator for a fixed RGB + fixed depth pair.
- It emits both `baseline` and `quality_4k` outputs from the same depth:
  - `left`
  - `right`
  - `half_sbs`
  - `full_sbs`
  - `occlusion_mask`
  - `shift_px`
  - absdiff images
  - contact sheet
  - `visual_regression_report.json`

Why it matters:

- Future synthesis optimizations can be checked against the same visual baseline.
- This prevents optimizing only for milliseconds while silently hurting edges, holes, UI text, or left/right consistency.
- The script defaults `temporal=False` so single-frame regression output is deterministic.
- The script timing is only a coarse smoke value because it also performs first-use work and image output; use `profile_synthesis_4k.py` and `bench_end_to_end_4k.py` for performance claims.

Example:

```powershell
.\python3\python.exe -B scripts\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\4k_native_base_quality
```

Verified output set:

```text
outputs/visual_regression/4k_native_base_quality/
```

Key outputs:

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
- `visual_regression_report.json`

### Base Grid Cache

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

File:

- `src/stereo_lab/baseline_shift.py`

What changed:

- Cached the fixed 4K base grid by `(batch, height, width, device, dtype)`.

Why this is safe:

- The base normalized grid is deterministic for the same shape/device/dtype.
- Shift values are still computed per frame from the current depth.
- No depth inference path is touched.

### Removed Redundant Baseline Warp From Quality Path

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/synthesis.py`
- `src/stereo_lab/baseline_shift.py`

What changed:

- `quality_4k` no longer computes baseline left/right eyes just to get `shift_px`.
- It now computes `base_shift = compute_shift_px(...)` directly.

Why this is safe:

- The shift formula is unchanged.
- Removed unused full-frame warps.
- No depth inference path is touched.

Profile after this earlier optimization:

```text
outputs/synthesis_profile_4k/quality_half_profile_no_baseline_warp.json
```

This became the reference profile for later synthesis work.

### Hole Fill Kernel Cache And Batched Left/Right Fill

Commit:

```text
2f1ab45 feat: add native tensorrt depth backend
```

Files:

- `src/stereo_lab/hole_fill.py`
- `src/stereo_lab/synthesis.py`

What changed:

- Cached box blur kernels by channel count, kernel size, device, and dtype.
- Filled left/right eyes together as one batch:

```python
eyes = torch.cat([left, right], dim=0)
fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
eyes = edge_aware_fill(eyes, fill_mask, radius=radius, strength=strength)
left, right = eyes.chunk(2, dim=0)
```

Why this is safe:

- Left/right fill formula and mask are unchanged.
- It reduces repeated kernel setup and makes the fill path more consistent.
- No depth inference path is touched.

## Baseline Before This Round

Reference profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_no_baseline_warp.json
```

Quality 4K, 2 layers, Half-SBS, 4K input:

| Stage | Mean ms |
|---|---:|
| baseline_shift | 1.238 |
| make_layers | 4.403 |
| warp_layers | 14.543 |
| composite | 9.745 |
| occlusion | 7.679 |
| hole_fill | 20.658 |
| make_sbs | 1.886 |

Main bottlenecks:

1. `hole_fill`
2. `warp_layers`
3. `composite`
4. `occlusion`

## Kept Optimization 1: Layer Center Cache + In-Place Composite

Commit:

```text
49da1fc perf: reduce quality synthesis allocations
```

Files:

- `src/stereo_lab/layers.py`
- `tests/test_synthesis.py`

Changes:

- Cached `torch.linspace` layer centers by `(layers, device, dtype)`.
- Changed `composite_layers` from repeated `out = out + layer * weight` allocation to `out.addcmul_(...)`.

Why this is safe:

- Layer center values are identical for the same layer count, device, and dtype.
- Composite math is unchanged.
- No depth inference path is touched.

Profile after change:

```text
outputs/synthesis_profile_4k/quality_half_profile_composite_inplace.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| composite | 9.745 | 6.890 |
| synthesis mean | ~59.1 | 58.319 |

## Kept Optimization 2: Separable Box Blur For Hole Fill

Commit:

```text
49da1fc perf: reduce quality synthesis allocations
```

Files:

- `src/stereo_lab/hole_fill.py`
- `tests/test_synthesis.py`

Changes:

- Replaced one `k x k` depthwise box blur with two depthwise passes:
  - horizontal `1 x k`
  - vertical `k x 1`
- Cached horizontal and vertical kernels separately.

Why this is safe:

- A box filter is separable, so the mathematical kernel is the same.
- Added `test_box_blur_matches_2d_kernel`.
- Stereo output semantics are unchanged except for possible floating-point accumulation order noise.
- No depth inference path is touched.

Profile after change:

```text
outputs/synthesis_profile_4k/quality_half_profile_separable_fill.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| hole_fill | 21.331 | 19.434 |
| synthesis mean | 58.319 | 56.162 |

End-to-end after this optimization:

```text
outputs/end_to_end_4k/quality_native_synthesis_safe_opt.json
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.881 | 55.058 | 71.941 | 13.90 |
| Full-SBS | 15.275 | 54.308 | 69.584 | 14.37 |

## Rejected Experiment: Batch Warp

Status:

```text
reverted, not committed
```

Attempt:

- Added a temporary `warp_horizontal_batch`.
- Tried to combine layer/eye warps into one larger batch `grid_sample`.

Reason for rejection:

- 4K memory pressure and input/grid replication outweighed the reduced kernel launch count.
- It did not improve `warp_layers`.

Profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_batch_warp.json
```

| Stage | Previous ms | Batch-warp ms |
|---|---:|---:|
| warp_layers | ~14.25 | 14.483 |
| synthesis mean | 56.162 | 57.335 |

Decision:

- Do not use batch warp in the current PyTorch 4K path.
- Future warp optimization should likely be a fused CUDA/shader/TensorRT-style kernel, not larger PyTorch batch replication.

## Kept Optimization 3: In-Place Depth Edge Accumulation

Commit:

```text
1e2a171 perf: reduce depth edge allocations
```

Files:

- `src/stereo_lab/layers.py`
- `tests/test_synthesis.py`

Changes:

- Replaced padded `dx` + padded `dy` temporary tensors in `depth_edges`.
- New implementation uses one `edges = torch.zeros_like(depth)` and accumulates x/y differences in place.

Why this is safe:

- Added `test_depth_edges_matches_padded_gradient_formula`.
- The test verifies exact equality with the old padded-gradient formula.
- This affects occlusion mask calculation only after depth is already generated.
- No depth inference path is touched.

Profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_depth_edges_inplace_rerun.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| occlusion | ~7.81 | ~7.09 |
| synthesis mean | 56.162 | 56.166 |

End-to-end after this optimization:

```text
outputs/end_to_end_4k/quality_native_synthesis_edges_opt.json
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.200 | 54.951 | 71.152 | 14.05 |
| Full-SBS | 15.201 | 53.022 | 68.224 | 14.66 |

## Kept Optimization 4: Base Grid Component Cache

Commit:

```text
85166be perf: cache warp grid components
```

Files:

- `src/stereo_lab/baseline_shift.py`
- `tests/test_synthesis.py`

Changes:

- Added cached base grid components:

```text
xx, yy = make_base_grid_components(height, width, device, dtype)
```

- `warp_horizontal` now builds the grid from cached x/y components instead of cloning the full `[B,H,W,2]` base grid and overwriting the x channel.

Why this is safe:

- The same normalized x/y grid values are used.
- The shift formula is unchanged.
- `grid_sample` parameters are unchanged:

```text
mode="bilinear", padding_mode="border", align_corners=True
```

- Added `test_warp_horizontal_matches_cached_grid_formula`, comparing against the old cached-grid-clone formula.
- No depth inference path is touched.

Profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_grid_components_rerun.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| warp_layers | ~14.62 | ~14.45 |
| synthesis mean | 56.166 | 55.742 |

End-to-end:

```text
outputs/end_to_end_4k/quality_native_grid_components.json
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.593 | 54.697 | 71.292 | 14.03 |
| Full-SBS | 15.550 | 54.482 | 70.034 | 14.28 |

Judgment:

- Keep as a small safe synthesis optimization.
- Synthesis-only profile improved slightly.
- End-to-end FPS did not show a clear stable gain, likely due to depth/output bandwidth and GPU timing variance.
- Do not overstate this as a major speedup.

## Verification Commands

Syntax:

```powershell
.\python3\python.exe -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax ok', len(files), 'files')"
```

Tests:

```powershell
.\python3\python.exe -B -m pytest -q
```

Latest result:

```text
22 passed
```

Profile:

```powershell
.\python3\python.exe -B scripts\profile_synthesis_4k.py --rgb 4K.jpg --out outputs\synthesis_profile_4k\quality_half_profile_depth_edges_inplace_rerun.json --backend quality_4k --layers 2 --output-format half_sbs --iters 5
```

End-to-end:

```powershell
.\python3\python.exe -B scripts\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\end_to_end_4k\quality_native_synthesis_edges_opt.json --warmup 2 --iters 5 --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```

## Current Bottlenecks

Latest profile indicates the main remaining bottlenecks are:

| Stage | Approx ms | Notes |
|---|---:|---|
| hole_fill | ~20 | Full-frame 4K blur/blend remains bandwidth-heavy |
| warp_layers | ~14-15 | Multiple `grid_sample` calls dominate geometry synthesis |
| occlusion | ~7 | Improved, but still non-trivial |
| composite | ~6-7 | Improved, still bandwidth-bound |

## Next Safe Optimization Candidates

These must still preserve depth model, inference resolution, resize, antialias, and normalization semantics.

1. Profile `hole_fill` at kernel level and evaluate whether mask-limited fill can be implemented without changing visible semantics.
2. Investigate a fused CUDA/Torch extension for warp + composite, with output-equivalence tests against the PyTorch reference.
3. Keep PyTorch batch warp rejected unless a new implementation avoids 4K tensor replication.
4. Add visual regression/contact sheet for baseline vs quality after every synthesis optimization.

## Do Not Repeat Without New Evidence

These items were already tested or constrained by the user requirements.

| Item | Current decision | Reason |
|---|---|---|
| Lowering depth inference resolution | Do not do it | User explicitly rejected quality-affecting resolution changes. |
| Changing resize/antialias/normalize | Do not do it | Could change generated depth and invalidate comparisons. |
| Replacing model for speed | Do not treat as optimization | Smaller models can be evaluated separately, but not as an optimization of the current model path. |
| Native TensorRT CUDA Graph | Not default | Current graph path was slower than normal native TensorRT. |
| PyTorch batch warp | Rejected | 4K tensor replication made it slower. |
| Distill-Any-Depth-Large on RTX 2060 realtime | Not a 60 FPS target | Large native TensorRT depth alone is about 29-30 FPS on RTX 2060. |
| Per-frame provider/session construction | Never use in realtime | It measures setup/build/load overhead, not inference. |
| Writing models into Desktop2Stereo model dir | Do not do it | Project artifacts must stay under this repo's `models/`. |

## Current Overall Status

Depth backend is no longer the main 4K bottleneck on RTX 2060 when using Native TensorRT Base @ 518.

Current best depth backend:

```text
Native TensorRT, Distill-Any-Depth-Base @ 1x3x294x518
mean 16.425 ms, median 15.641 ms
```

Current end-to-end Quality 4K on RTX 2060:

```text
Half-SBS: about 14.05 FPS
Full-SBS: about 14.66 FPS
```

Main remaining bottleneck:

```text
stereo synthesis, especially hole_fill and warp_layers
```

Next meaningful performance jump probably requires one of:

- A fused CUDA/shader implementation for warp + composite.
- A semantically equivalent but lower-bandwidth hole fill implementation.
- Hardware stronger than RTX 2060 for the same PyTorch prototype path.
