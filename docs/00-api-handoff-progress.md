# Handoff - 2026-06-19

## Project

Repo:

```text
D:\AI_2D_to_3D\4.LC700X_Desktop2Stereo\4k-stereo-synthesis-lab
```

GitHub:

```text
https://github.com/laiyangli001/4k-stereo-synthesis-lab
```

Latest pushed commit:

```text
8ca5b07 feat: refine stereo runtime controls
```

Important docs:

- `docs/benchmark/07-depth-backend-benchmark.md`
- `docs/benchmark/08-synthesis-optimization-log.md`
- `docs/benchmark/10-rtx3090-fused-synthesis-results.md`
- `docs/11-visual-regression-guide.md`
- `docs/12-openxr-stereo-runtime-plan.md`
- `docs/13-realtime-stereo-parameter-guide.md`
- `docs/14-host-api-preset-examples.md`
- `docs/15-host-api-contract.md`
- `docs/19-capture-architecture-flow.md`
- This file: `docs/00-api-handoff-progress.md`

Note:

- This file is the single current handoff entry point. Early design documents are archived under `docs/archive/`, and benchmark/optimization reports live under `docs/benchmark/`.

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

## Project Scope

This repository is the algorithm / inference / stereo synthesis core library. It should provide stable external function calls, configuration parameters, output formats, performance verification, and visual regression tooling for another GUI/runtime host to call.

2026-06-19 scope update:

- The repo now also contains the active Desktop2Stereo host integration path in `src/main.py`, `src/capture/`, and `src/viewer/`.
- The long-term module boundary still remains: `stereo_runtime` owns depth inference, stereo synthesis, OpenXR per-eye render core, output tensors, timings, and provider/artifact contracts.
- The host layer owns capture, GUI/window/session lifecycle, OpenXR session/swapchain timing, runtime settings persistence, and final display/submit.
- Do not move GUI/OpenXR session policy back into `stereo_runtime`; only move reusable RGB->depth->output computation there.

In scope for this repository:

- Depth inference providers and benchmarks.
- Stereo synthesis and output packing.
- `StereoConfig` / OpenXR core config fields and defaults.
- Realtime temporal reset and depth post-processing primitives.
- Mode preset definitions and recommended parameter values.
- OpenXR roll-adaptive per-eye render core.
- Visual regression tools and quality documentation.
- Unit tests for arbitrary resolution, output formats, OpenXR roll, temporal reset, and config defaults.

Out of scope for this repository:

- New standalone product GUI implementation.
- Installer / packaging.
- Product logging UI and runtime configuration persistence UI.
- Product-level error recovery UI.

Now present as host integration code, but still outside `stereo_runtime` core:

- Desktop capture and player/window capture.
- Full OpenXR session / swapchain / frame timing runtime.

Those host pipeline pieces should remain separated from `stereo_runtime` by the API boundary documented in `docs/15-host-api-contract.md`.

Current completion estimate for this repository:

```text
Core algorithm and performance validation: 90-95%
External API/preset stability: 90-95%
Capture/runtime host integration: runtime-direct GPU paths integrated, real-device validation pending
```

## Current Status

### 2026-06-20 Realtime Stereo GUI Defaults + Hot Parameters

Latest pushed commit before the current uncommitted GUI/doc refresh:

```text
8ca5b07 feat: refine stereo runtime controls
```

Committed in `8ca5b07`:

- Removed the `Depth Safety` feature and its tests. It was a local heuristic path and is no longer part of the runtime/GUI contract.
- Removed the stereo scene `Auto` choice from the GUI. The persisted/default stereo preset is now `cinema`; legacy `auto` preset input is still normalized to `cinema` for compatibility.
- Added `Stereo Scale` next to IPD in the GUI, persisted as `Stereo Scale`, with English/Chinese labels and tooltips.
- Fixed left/right eye order so ordinary SBS viewing no longer depends on enabling `Cross Eyed` as a workaround.
- Added static tests for hot-reload stereo parameters and eye-order visual regression coverage.
- Improved runtime profiling/logging around stereo synthesis stages and removed noisy third-party console messages from the product logs.
- Kept temporal scene-change detection as a temporal-stability feature, not as stereo scene auto-switching. GPU hot-path sync points were reduced; depth-safety CPU paths were removed with the feature.

Current uncommitted changes after `8ca5b07`:

- `src/gui.py`:
  - `Temporal Strength` stereo dropdown now uses dense options `0.0` through `1.0` in `0.1` steps, default `0.7`.
  - Stereo quality localization now treats both `CN` and `ZH` as Chinese, so the quality dropdown shows `最低 / 中等 / 较高 / 最高` instead of `Lowest / Medium / High / Highest` when the GUI language is Chinese.
- `src/settings.yaml`:
  - `Stereo Scale: 0.5` is present.
  - `Temporal Strength` default is aligned to `0.7`.
- `docs/13-realtime-stereo-parameter-guide.md`:
  - Temporal Strength GUI range is documented as `0.0-1.0`, step `0.1`, default `0.7`.
- `tests/test_gui_config.py`:
  - Added assertions locking the dense Temporal Strength options and `CN`/`ZH` stereo quality language behavior.

Scene reset threshold note:

- The `Scene Reset Threshold` options `0.00, 0.12, 0.18, 0.22, 0.28, 0.35` are hand-tuned temporal reset sensitivity presets, not computed from a formula.
- Lower values reset temporal history more aggressively on scene changes; higher values are more conservative. `0.22` is the current balanced default.
- This threshold belongs to temporal stabilization reset behavior and is unrelated to the removed stereo-scene Auto switching UI.

Latest targeted verification for the current uncommitted changes:

```powershell
src\python3\python.exe -m py_compile src\gui.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_gui_config.py -q
```

Result:

```text
16 passed
```

Current worktree status at this handoff update:

```text
M docs/13-realtime-stereo-parameter-guide.md
M src/gui.py
M src/settings.yaml
M tests/test_gui_config.py
```

Before the next commit, review these four files together. The `src/settings.yaml` diff is currently expected because it keeps GUI defaults aligned with the new controls.

### 2026-06-19 GUI Runtime Parameter Layout + Streaming Asset Move

This session focused on making the existing Flet host GUI usable with the expanded realtime stereo/runtime parameter set, while keeping the runtime/core API boundary unchanged.

GUI updates in `src/gui.py`:

- Added simplified user-facing stereo controls:
  - `Depth Quick` fixed presets: Soft / Standard / Enhanced (`柔和` / `标准` / `增强`).
  - `Stereo Preset` default remains forced to `auto` on every GUI startup and save.
  - `Stereo Quality` is displayed in Chinese as `立体质量`.
- Added `Advanced Stereo` folding behavior:
  - advanced stereo controls are hidden by default,
  - `Convergence` and numeric `Depth Strength` are advanced-only,
  - `Convergence` + `Depth Strength` share one advanced row directly below `Stereo Mode` / `Stereo Quality`,
  - `Depth Resolution` + `Depth Quick` share one basic row,
  - `Temporal` and `Auto Scene Reset` standalone checkboxes were removed from the GUI; they are now inferred from numeric values:
    - `Temporal Strength == 0` disables temporal stabilization,
    - `Temporal Strength > 0` enables temporal stabilization,
    - `Scene Reset Threshold == 0` disables auto scene reset,
    - `Scene Reset Threshold > 0` enables auto scene reset.
- Added `Advanced Device Options` folding behavior:
  - Chinese label: `高级选项`,
  - controls visibility for `Capture FPS`, `Local VSync`, `Upscaler`, and `Upscaler Sharpness`,
  - `Capture FPS` now localizes `Auto` as `自动` in Chinese and still saves as `Target FPS: 0`.
- Moved acceleration toggles into advanced stereo controls:
  - `FP16` defaults to unchecked,
  - `FP16` is not persisted as a long-lived preference; after launch, saved settings are reset to default `False`,
  - `torch.compile`, `TensorRT`, and `Recompile TensorRT` share one acceleration row.
- Reworked layout density and automatic sizing:
  - footer/status spacing was reduced,
  - window width/height estimation was tuned several times to reduce extra blank space while keeping scroll fallback for expanded advanced sections,
  - bottom button and refresh button horizontal offsets were manually tuned per user visual checks.
- Updated room/environment model lookup to use only:

```text
src/xr_viewer/environments
```

There is no compatibility fallback to the old `src/environment` path.

Host/streaming asset update:

- The old `src/rtmp/` asset folder was moved to:

```text
src/streaming/rtmp/
```

- `src/main.py` now defines:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTMP_DIR = os.path.join(BASE_DIR, "streaming", "rtmp")
```

and RTMP/ffmpeg/mediamtx executable paths now resolve from `RTMP_DIR` instead of hard-coded `./rtmp/...` paths.

- Root `update_windows.bat` was updated to clean platform-specific RTMP folders under `src\streaming\rtmp\...`. Note: in the current worktree this root `update_windows.bat` is untracked because related batch files were previously moved/deleted outside this session.

Current GUI/host verification performed during this session:

```powershell
.\src\python3\python.exe -B -m py_compile src\gui.py
.\src\python3\python.exe -B -m py_compile src\main.py src\gui.py
.\src\python3\python.exe -B -m py_compile src\main.py
```

Additional checks used repeatedly:

```powershell
git diff --check -- src/gui.py
git diff --check -- src/main.py update_windows.bat src/gui.py
```

Known caveats:

- The GUI layout has been tuned visually through user screenshots, but it has not yet been validated with an automated Flet screenshot regression.
- The current worktree includes unrelated deleted/moved assets and scripts that were not created by this GUI cleanup. Do not revert them without explicit user approval.
- `src/settings.yaml` may be modified by local GUI runs and should be reviewed before committing.

### 2026-06-19 Capture Split + Runtime Output Handoff

Latest pushed commits:

```text
1ec01e9 refactor: split capture and route runtime output
bbbc72d feat: route openxr viewer runtime eyes
d5fcb32 feat: add runtime eye gpu interop paths
40f9451 refactor: split xrviewer runtime package
adb57c8 refactor: move xr_viewer to top-level package
9127a2d refactor: remove legacy xrviewer wrappers
```

Capture architecture was split out of the former monolithic `src/capture/__init__.py`.

New capture package layout:

- `src/capture/__init__.py`: thin public re-export layer.
- `src/capture/types.py`: `CaptureConfig`, frame/source/runner protocols.
- `src/capture/factory.py`: backend selection by `os_name` and `capture_tool`.
- `src/capture/runners.py`: polling/event runner loop behavior.
- `src/capture/preprocess.py`: `capture_frame_to_rgb()` and `prepare_rgb_for_depth_runtime()`.
- `src/capture/geometry.py`: monitor/window geometry helpers.
- `src/capture/backends/windows_dxcamera.py`
- `src/capture/backends/windows_desktop_duplication.py`
- `src/capture/backends/windows_capture_event.py`
- `src/capture/backends/macos_screencapturekit.py`
- `src/capture/backends/macos_coregraphics.py`
- `src/capture/backends/linux_mss.py`

Public compatibility imports remain valid:

```python
from capture import DesktopGrabber, capture_frame_to_rgb, prepare_rgb_for_depth_runtime
```

Recommended capture entry for `main.py` / host code:

```python
from capture import CaptureConfig, create_capture_runner
```

Detailed capture flow document:

```text
docs/19-capture-architecture-flow.md
```

`src/main.py` capture path now builds a `CaptureConfig`, calls `create_capture_runner(config)`, and receives frames through callbacks. Raw queue write/statistics still remain in `main.py`; the capture package does not depend on app state.

### 2026-06-19 StereoRuntime Second-Stage Integration

The main host pipeline has moved from depth-only runtime output toward full `stereo_runtime` output ownership.

Current ordinary Viewer / SBS path:

```text
capture runner
-> capture_frame_to_rgb()
-> prepare_rgb_for_depth_runtime()
-> StereoRuntime.process_rgb_frame()
-> StereoRuntimeResult.sbs
-> StereoWindow.update_runtime_frame()
```

The ordinary viewer no longer feeds RGB+depth back into the old viewer shader for SBS synthesis. It displays the final `StereoRuntimeResult.sbs` texture directly.

Legacy MJPEG streamer now uses `StereoRuntimeResult.sbs` directly instead of `streaming.legacy_sbs.make_sbs()`.

New runtime OpenXR API:

```python
result = runtime.process_openxr_frame(rgb_frame, openxr_config)
```

Returns:

```python
OpenXRRuntimeResult(
    depth=...,
    left_eye=...,
    right_eye=...,
    timing=...,
    debug_info=...,
    provider_info=...,
)
```

Current OpenXR host path:

```text
capture runner
-> prepare_rgb_for_depth_runtime()
-> StereoRuntime.process_openxr_frame()
-> OpenXRRuntimeResult.left_eye/right_eye
-> src/xr_viewer runtime-direct per-eye upload
   - CUDA/GL image interop direct write when available
   - GL PBO GPU fallback when image interop is unavailable
   - CPU/GL texture upload fallback
   - D3D11 native runtime-eye upload/render when D3D11 session path is active
-> _render_eye() or render_runtime_eye() direct texture display on virtual screen
-> OpenXR swapchain submit
```

Fallback preserved:

```text
(rgb, depth, frame_ts) queue item
-> old xrviewer RGB+depth upload
-> old DIBR shader path
```

OpenXR runtime parameter feedback:

- `xrviewer` reports current `ipd`, `depth_ratio`, `convergence`, and `screen_roll` back to `main.py`.
- `main.py` builds `OpenXRRenderConfig` from that state before calling `process_openxr_frame()`.
- This keeps controller/keyboard-driven OpenXR parameter changes aligned with runtime per-eye generation.

Current OpenXR GPU path status:

- The active OpenXR viewer package is now `src/xr_viewer/`, at the same level as `src/viewer/`.
- Legacy `src/viewer/xrviewer*.py` compatibility wrappers were removed; host code imports from `xr_viewer.base`, `xr_viewer.environment`, and `xr_viewer.implementation` directly.
- OpenGL runtime-eye fast path is implemented:
  - preferred: CUDA tensor -> `cudaGraphicsGLRegisterImage(GL texture)` -> CUDA array -> `cudaMemcpy2DToArray`,
  - fallback: CUDA/GL PBO upload,
  - final fallback: CPU/GL texture upload.
- D3D11 native runtime-eye path is implemented in `src/xr_viewer/d3d11_native_renderer.py`:
  - `D3D11NativeRenderer.update_runtime_eyes(left, right)`,
  - `render_runtime_eye(...)`,
  - CUDA-D3D11 interop copy to D3D11 textures when available,
  - automatic fallback to the OpenGL/PBO/CPU path if D3D11 native upload/render fails.
- Runtime direct can be disabled with `D2S_OPENXR_RUNTIME_DIRECT=0`; runtime-eye GPU upload can be disabled with `D2S_OPENXR_RUNTIME_EYE_GPU_UPLOAD=0`.

Latest verification after capture split and OpenXR runtime-eye integration:

```text
.\src\python3\python.exe -B -m pytest -q
143 passed, 1 warning
```

Targeted verification:

```text
tests/test_capture_preprocess.py
tests/test_capture_factory.py
tests/test_capture_public_api.py
tests/test_runtime_openxr.py
tests/test_openxr_render.py
tests/test_host_api_smoke.py
tests/test_d2s_depth_runtime_smoke.py
```

New/updated tests:

- `tests/test_capture_preprocess.py`
- `tests/test_capture_factory.py`
- `tests/test_capture_public_api.py`
- `tests/test_runtime_openxr.py`
- `tests/test_runtime_depth_safety.py`
- `tests/test_adapter_config.py`

### 2026-06-17 `stereo_runtime` API Migration

The real implementation package and host-facing calls now use `stereo_runtime`.

Recommended public import:

```python
from stereo_runtime import StereoRuntime, StereoRuntimeConfig
```

The old `stereo_lab` compatibility proxy has been removed. Host projects must import from `stereo_runtime`.

Actual scripts, smoke tools, benchmark tools, and tests import from `stereo_runtime`.

Current runtime handoff entry:

```python
runtime_config = StereoRuntimeConfig(
    model_id="lc700x/Distill-Any-Depth-Base-hf",
    model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
    mode="movie",
    stereo_quality="quality_4k",
    output_format="half_sbs",
    depth_backend="auto",
)

runtime = StereoRuntime(runtime_config)
runtime.load()
result = runtime.process_rgb_frame(rgb_frame)
```

Runtime contract field names now use `stereo_runtime_responsibility` and `not_stereo_runtime_responsibility`.

Triton disable environment variable:

- preferred: `STEREO_RUNTIME_DISABLE_TRITON=1`
- compatibility alias: `STEREO_LAB_DISABLE_TRITON=1`

Document movement cleanup:

- `4K 高质量立体生成算法实现计划书.md` moved from repo root to `docs/`.
- Active implementation paths inside current handoff docs now point to `src/stereo_runtime/...`.
- Historical benchmark logs may still mention `src/stereo_lab/...` because they describe past commits. Current implementation paths are under `src/stereo_runtime/...`.

Latest verification after migration:

```text
syntax ok 88 files
86 passed, 1 warning
host API smoke commands pass
```

### 2026-06-17 Repository Organization Update

The docs and scripts layout has been cleaned up for API handoff.

Latest pushed commit before this documentation refresh:

```text
Use `git log -1 --oneline` after pulling.
```

Current script layout:

- `scripts/benchmark/`: performance benchmarks and profiling entry points.
- `scripts/tools/`: model export, depth generation, comparison, consistency, and visual regression utilities.
- `scripts/smoke/`: host/API smoke checks and Auto runtime demos.
- `scripts/examples/`: small demos and OpenXR preview helpers.
- `scripts/windows/`: visible Windows launchers for manual testing.
- `scripts/dev/`: low-level TensorRT/Triton development probes.

Current docs layout:

- `docs/00-api-handoff-progress.md`: single current handoff entry point.
- `docs/00-goals.md`: current project goals and non-goals.
- `docs/14-host-api-preset-examples.md`: host/preset usage examples.
- `docs/15-host-api-contract.md`: API boundary and host responsibilities.
- `docs/benchmark/`: benchmark and optimization reports.
- `docs/archive/`: early design documents retained for history only.

Notes:

- `README.md`, `docs/00-goals.md`, `docs/14-host-api-preset-examples.md`, and `docs/15-host-api-contract.md` were rewritten as current concise entry documents for API handoff.
- Some consoles may display Chinese documents incorrectly if the active code page/decoder is not UTF-8. Prefer editor view or explicit UTF-8 reads when checking Chinese docs.
- Prefer the current entry documents above when handing off to another Agent.

### 2026-06-17 API / Preset / Auto Runtime Update

The external preset/API layer is now implemented and pushed.

Latest pushed commit:

```text
Use `git log -1 --oneline` after pulling.
```

Implemented host-facing API:

- `StereoModePreset`
- `PRESET_CHOICES`
- `AutoModeSignals`
- `AutoModeDecision`
- `AutoModeRuntime`
- `AutoModeRuntimeState`
- `auto_detection_required`
- `auto_mode_scores`
- `classify_auto_mode`
- `stereo_config_for_preset`
- `openxr_config_for_preset`
- `stereo_config_for_auto_mode`
- `openxr_config_for_auto_mode`
- `preset_summary`

Preset modes:

- `auto`
- `cinema`
- `game_low_latency`
- `still_image_hq`
- `debug_export`

Important Auto behavior:

- Scene detection is only required when the user selected `auto`.
- Manual presets such as `cinema`, `game_low_latency`, `still_image_hq`, and `debug_export` should not start scene detection.
- System metrics must be sampled asynchronously by the GUI/runtime host, not inside capture, depth inference, or stereo synthesis.
- `AutoModeRuntime` only consumes pre-aggregated `AutoModeSignals` snapshots.
- Behavior scoring prioritizes GPU 3D, Video Decode, input activity, idle time, audio, fullscreen/maximized state, and frame motion.
- Foreground process name is only a low-weight hint; the implementation does not rely on a large game/application whitelist.
- `AutoModeRuntime` handles consecutive-sample confirmation and hold time. Host code should still apply parameter blending using `blend_seconds`.

Host/API docs:

- `docs/14-host-api-preset-examples.md`
- `docs/15-host-api-contract.md`

Host/API scripts:

- `scripts/smoke/host_api_smoke.py`
- `scripts/smoke/auto_mode_runtime_demo.py`

Host/API tests:

- `tests/test_presets.py`
- `tests/test_host_api_smoke.py`
- `tests/test_auto_mode_runtime_demo.py`

Latest targeted verification:

```text
tests/test_auto_mode_runtime_demo.py + tests/test_presets.py + tests/test_host_api_smoke.py:
15 passed
```

### 2026-06-17 RTX 3090 Update

RTX 3090 formal fused-synthesis pass is complete. Current 4K `quality_4k` no longer has synthesis as the primary bottleneck for the Base model.

Latest detailed result doc:

```text
docs/benchmark/10-rtx3090-fused-synthesis-results.md
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

OpenXR note: the local environment has pyopenxr available as the `xr` module, but this lab does not yet include a full OpenXR session/swapchain runtime. The rotation-adaptive stereo core has been added in `src/stereo_runtime/openxr_render.py`; it accepts arbitrary `screen_roll` angles in radians and should be used by a future runtime integration instead of fixed SBS output. `scripts/examples/generate_openxr_stereo_preview.py` can generate roll-adaptive left/right preview images from RGB+depth inputs.

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
- Environment variable `STEREO_RUNTIME_DISABLE_TRITON=1` disables Triton fused paths globally. `STEREO_LAB_DISABLE_TRITON=1` remains supported as a compatibility alias.

Important:

- First Triton execution includes compile overhead. Do not use visual regression script smoke timing for performance claims.
- Use `bench_end_to_end_4k.py` for performance claims.
- `profile_synthesis_4k.py` reports `end_to_end_mean_ms` for real fused `synthesize_stereo` timing and `breakdown_mean_ms` for manual unfused component attribution.
- `generate_visual_regression_set.py` now accepts `--onnx` and `--trt-engine`; pass them explicitly for Large or other non-default engines.

Historical verification for this 2026-06-17 benchmark/API state:

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
docs/benchmark/07-depth-backend-benchmark.md
```

Important implementation files:

- `src/stereo_runtime/depth_provider.py`
- `src/stereo_runtime/depth_onnx_provider.py`
- `src/stereo_runtime/depth_trt_provider.py`
- `src/stereo_runtime/depth_trt_native_provider.py`

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

- `src/stereo_runtime/synthesis.py`
- `src/stereo_runtime/baseline_shift.py`
- `src/stereo_runtime/layers.py`
- `src/stereo_runtime/occlusion.py`
- `src/stereo_runtime/hole_fill.py`
- `src/stereo_runtime/output.py`

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
docs/benchmark/08-synthesis-optimization-log.md
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
scripts/tools/generate_visual_regression_set.py
```

Example:

```powershell
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\4k_native_base_quality
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
.\src\python3\python.exe -B -m py_compile src\main.py src\tools\preview_room_layout.py src\xr_viewer\implementation.py src\xr_viewer\base.py src\xr_viewer\environment.py src\xr_viewer\d3d11_native_renderer.py src\stereo_runtime\runtime.py src\capture\__init__.py
```

Tests:

```powershell
.\src\python3\python.exe -B -m pytest -q
```

Current latest result:

```text
143 passed, 1 warning
py_compile targeted syntax checks pass
```

Synthesis profile:

```powershell
.\src\python3\python.exe -B scripts\benchmark\profile_synthesis_4k.py --rgb 4K.jpg --out outputs\synthesis_profile_4k\<name>.json --backend quality_4k --layers 2 --output-format half_sbs --iters 5
```

End-to-end:

```powershell
.\src\python3\python.exe -B scripts\benchmark\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\end_to_end_4k\<name>.json --warmup 2 --iters 5 --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```

Visual regression:

```powershell
.\src\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\<name>
```

Host API smoke:

```powershell
.\src\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out outputs\host_api_smoke_cinema.json
.\src\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
.\src\python3\python.exe -B scripts\smoke\host_api_smoke.py --openxr --preset cinema --screen-roll 0.25 --out -
.\src\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset auto --out -
.\src\python3\python.exe -B scripts\smoke\host_api_smoke.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset cinema --output-format half_sbs --out outputs\host_api_smoke_4k_native.json
```

## Recommended Next Steps

1. Run a real OpenXR/headset validation pass for the current runtime-direct GPU/CPU fallback chain:
   - verify left/right eye order,
   - verify no Y flip or color-space regression,
   - verify `screen_roll`, `depth_ratio`, `ipd`, and `convergence` update correctly through the runtime config callback,
   - verify CUDA/GL image interop, GL PBO fallback, CPU/GL fallback, and D3D11 native runtime-eye fallback behavior on the target headset/runtime.
2. Improve OpenXR runtime-direct telemetry:
   - log current path: `runtime_direct_gpu_gl_texture`, `runtime_direct_gpu_gl_pbo`, `runtime_direct_cpu_gl`, `runtime_direct_d3d11_cuda`, or `rgb_depth_fallback`,
   - log left/right shape, dtype, device, upload timing, render mode, and fallback reason,
   - keep `D2S_OPENXR_RUNTIME_DIRECT=0` and `D2S_OPENXR_RUNTIME_EYE_GPU_UPLOAD=0` as forced fallback switches.
3. Keep `src/xr_viewer/` as the OpenXR viewer package boundary; do not reintroduce `src/viewer/xrviewer*.py` wrappers unless a legacy downstream dependency requires them.
4. Keep GUI/OpenXR host integration aligned with:
   - `docs/14-host-api-preset-examples.md`
   - `docs/15-host-api-contract.md`
   - `scripts/smoke/host_api_smoke.py`
   - `scripts/smoke/auto_mode_runtime_demo.py`
5. Start async scene detection only when `auto_detection_required(selected_preset)` is true.
6. Real system metric collection remains outside `stereo_runtime`:
   - GPU 3D / Video Decode sampling
   - keyboard/mouse activity sampling
   - audio activity sampling
   - foreground window/fullscreen detection
7. Re-run API and preset unit tests after host-facing changes:
   - `tests/test_host_api_smoke.py` locks the CLI JSON report contract for stereo and OpenXR host smoke paths.
   - `tests/test_auto_mode_runtime_demo.py` locks the simulated Auto host state-machine integration path.
8. Re-run formal benchmarks on RTX 3090 / RTX 5070 when available:
   - `bench_depth_backends.py`
   - `profile_synthesis_4k.py`
   - `bench_end_to_end_4k.py`
   - `generate_visual_regression_set.py`
9. Final locking step: generate visual regression sets for representative Cinema, Game / Low Latency, Still Image / HQ, and Debug / Export samples, then use those results to finalize preset default values.

## Current Bottleneck

For the integrated host path, the current bottleneck is no longer the absence of a GPU upload path; it is real-device validation and fallback telemetry:

- `StereoRuntime.process_openxr_frame()` produces `left_eye/right_eye`.
- `src/xr_viewer/` consumes `OpenXRRuntimeResult.left_eye/right_eye` directly.
- OpenGL CUDA image interop, GL PBO fallback, CPU/GL fallback, and D3D11 native runtime-eye upload/render paths are implemented.
- The next risk is correctness on real OpenXR runtimes: eye order, Y orientation, color format, swapchain runtime differences, sync behavior, and fallback selection.

For the algorithm path, fused Triton synthesis already covers the main 4K SBS/TAB/display output formats. Further algorithm performance work should be benchmark-driven.

The next major host-runtime jump likely requires:

- real headset validation of the runtime-direct path matrix,
- tighter telemetry and timing around upload/render fallback choice,
- then targeted fixes for the runtime/driver combination that fails first.

