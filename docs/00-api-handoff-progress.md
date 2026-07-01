# Handoff - 2026-06-27

## Project

Repo:

```text
E:\AI_2D_to_3D\4.LC700X_Desktop2Stereo\4k-stereo-synthesis-lab
```

GitHub:

```text
https://github.com/laiyangli001/4k-stereo-synthesis-lab
```

Current focus:

```text
Desktop2Stereo engineering-spec refactor tasks from prompts/codex-refactor-prompt.md
```

Latest pushed task commit:

```text
599981e feat: add GUI log panel controls and progress
```

Canonical specs for current work:

- `docs/28-Realtime-2d-to-3d-specification.md` - official final runtime process spec; `docs/25` is obsolete.
- `docs/26-desktop2stereo-engineering-design-specification.md` - engineering implementation, migration, compatibility cleanup, and compliance status.
- `prompts/codex-refactor-prompt.md`
- This file: `docs/00-api-handoff-progress.md`

## Current Boundaries

- Treat `docs/28-Realtime-2d-to-3d-specification.md` as canonical when Parallax Budget, render_size, OpenXR, or output contract details differ from the prompt or historical docs.
- Keep `stereo_runtime` responsible for depth inference, stereo synthesis, OpenXR render-core config, output tensors, timings, and provider/debug contracts.
- Keep capture/session/window lifecycle, GUI settings persistence, OpenXR session/swapchain timing, and final display/submit outside `stereo_runtime`.
- Keep compatibility paths where recent tasks introduced new contracts: `RuntimeSettingsSnapshot`, normalized parallax budgets, and `CapturedFrame` metadata.
- Do not commit or upload runtime artifacts: `models/`, `outputs/`, `python3/`, `python-cu13/`, `downloads/`, `.codegraph/`, or `4K.jpg`.

## Current Known Issues

- D3D11 native OpenXR direct shader still needs a follow-up to match the OpenGL RGB+depth direct shader's core DIBR semantics. Current D3D11 already applies `screen_roll` to parallax direction, but still lacks the OpenGL shader's 3-tap depth smoothing, non-linear depth shaping, edge falloff, soft disocclusion confidence, push-pull inpaint, alpha edge fade, feather controls, corner radius, and `u_resolution` / `u_viewport` semantics. Estimated AI time for the focused direct-shader parity pass is about 60-90 minutes excluding headset validation; full OpenGL `_render_eye()` experience parity is a larger 3-5 hour follow-up.
- Runtime scheduling/backpressure must stay latest-frame first. The recent high-refresh capture diagnosis showed direct `wc_cuda` capture can reach high cadence, but a full CUDA runtime frame without a GPU completion boundary can build an async GPU queue and backpressure WGC / CUDA interop down to roughly 50-60 FPS. This is a runtime scheduling issue, not an OpenXR-only presentation issue.
- The remaining "SBS only around 20 FPS" symptom is not explained by `StereoRuntime` compute time in the latest log. Runtime refresh shows `total_ms=3.4-3.7`, `depth_total_ms=1.7`, `synthesis_ms=1.7-2.0`, `stage_sbs=0.1`, and `pack_ms=0.0`, while `WindowsCaptureCUDA` reports about 72-82 FPS. Next investigation should check the outer loop after runtime compute: `RuntimePipelineLoop`, `runtime_q`, viewer/OpenXR submit, texture upload, present/vsync, or the FPS counter source.
- TensorRT ORT depth provider still has a serious GPU zero-copy violation: input is converted from CUDA tensor to CPU numpy before `OrtValue.ortvalue_from_numpy`, and output uses `OrtValue.numpy()` followed by `torch.from_numpy()`, putting the depth result back on CPU before later GPU work. This is now logged as a red CPU transfer/fallback, but the next optimization should remove the CPU round trip entirely.
- CUDA/GL image texture upload must remain image-texture-first. PBO is an acceptable GPU fallback, but it must be logged as fallback and must not hide the original image texture failure. Any CPU upload fallback in OpenXR, local viewer, stream/debug realtime display, or depth provider hot path must print a red console warning and record the reason.

## Future Work

Detailed engineering and migration rules live in `docs/26-desktop2stereo-engineering-design-specification.md`. This handoff file only tracks the current task queue and verification status.

Current task queue:

1. Complete GUI live hot-save direct emission of `RuntimeSettingsSnapshot`; keep settings.yaml polling as an explicit compatibility path.
2. Bring D3D11 native OpenXR RGB+depth direct shader to parity with the OpenGL direct shader's core DIBR quality semantics.
3. Validate CUDA/ROCm capture zero-copy on real hardware before any path reports `zero_copy=True`.
4. Keep runtime scheduling/backpressure covered: CUDA runtime should default to `D2S_RUNTIME_SYNC_AFTER_FRAME=auto`, latest-frame overwrite/drop is expected, and non-CUDA backends must not be accidentally forced into the CUDA sync policy.
5. Trace the remaining SBS/display FPS gap outside `StereoRuntime.process_*()`: add or inspect timing around raw dequeue, runtime call, runtime_q put/drop, viewer dequeue, texture upload, OpenXR/local submit, present/vsync, and FPS reporting.
6. Deepen CUDA/GL image texture diagnostics: image texture must be tried first, PBO must be marked as GPU fallback, CPU fallback must be red-warning visible, and the original image texture failure reason must not be swallowed.
7. Optimize TensorRT ORT depth provider for real GPU zero-copy: replace CPU numpy input binding with CUDA/DLPack or direct CUDA OrtValue binding, keep iobinding output on CUDA, and return a CUDA torch tensor without `OrtValue.numpy()` / `torch.from_numpy()` CPU staging.
8. Remove remaining compatibility redundancy after all consumers use the docs/28 contract: old snapshot/API aliases and debug-only fallback keys. Legacy parallax multiplier fields and historical render-scale numeric thresholds have been cleaned from the current runtime/config path and should now be guarded against regressions.
9. Continue network_stream encoder transport work, especially RTMP / low-latency paths, without redefining stereo synthesis semantics.
10. Keep `docs/26-desktop2stereo-engineering-design-specification.md` aligned to the `docs/28-Realtime-2d-to-3d-specification.md` eleven-step runtime flow.

## Current Status

### 2026-07-01 GUI Log Panel, Standard Logging, and Download Progress Pass

Implemented and pushed:

```text
599981e feat: add GUI log panel controls and progress
```

Summary:

- Added a right-side Flet log panel with persistent show/hide preference via `Show Log Panel`, status-bar link control, live GUI log queue display, log-level filtering, `反馈bug`, and `查看log文件` actions.
- Kept the Flet in-window log panel as the canonical GUI log surface. The abandoned pywebview / HTML viewer path was removed and must not be reintroduced unless explicitly requested.
- Kept the existing Python `logging` pipeline on standard `logging.StreamHandler`, `FileHandler`, and `GuiLogHandler`. Console/file/GUI handlers share the same log stream, while noisy Flet startup messages are downgraded or filtered so they do not flood INFO output.
- Replaced terminal-styled progress output with structured `[D2S_PROGRESS]` events. The GUI renders those events with Flet native `ProgressBar`, percent, downloaded / total size, speed, ETA, and blue / green status coloring. Plain status lines from `progress_write()` intentionally use raw stdout so long messages are not wrapped into multiple GUI log rows.
- Updated the Flet log layout to avoid wrapping long log strings; the log body uses horizontal scrolling while preserving simple selectable `ft.Text` log rows.
- Adjusted log-panel window sizing behavior: when the log panel is hidden it contributes zero width; when visible it starts at 500 px. The total window target width follows the current visible layout, while `page.window.min_width` is constrained to the left parameter pane width so hiding the log panel can shrink back to the main GUI.
- GUI window close now force-kills the child process tree immediately instead of relying on the graceful stop-file path, because model downloads may not poll `stop.request`; the Stop button still tries graceful stop before forced cleanup. `run_windows.bat` also force-kills existing `python.exe` and `pythonw.exe` processes before launching Desktop2Stereo, by design, to prevent orphaned model-download processes from accumulating.
- `max_width` is used only as a short-lived Flet/Windows resize trigger during log show/hide and is cleared afterward so the user can manually resize the window.

Current local follow-up not yet pushed:

- Ordinary `transformers.AutoModelForDepthEstimation.from_pretrained()` downloads and manual InfiniDepth `hf_hub_download(..., tqdm_class=...)` downloads now use the same scoped HuggingFace Hub tqdm patch, backed by the standard-library `DownloadProgress`.
- `DownloadProgress` respects HuggingFace's `initial` byte count so resumed downloads do not start from zero, and emits structured events instead of terminal control output.
- Model artifact preparation is now backend-aware and shared: ONNX/XPU/CPU paths only treat `.onnx` as executable, TensorRT/CUDA paths only treat `.trt` as executable, and MIGraphX/ROCm paths only treat `.mgx` as executable. A local artifact skips weight/download work only for its matching backend.
- The common model preparation order is now: selected model -> matching local executable artifact -> local weight file (`model.safetensors`, `model.pt`, `model.ckpt`) -> endpoint probe/download (`https://hf-mirror.com`, then `https://huggingface.co` unless `HF_ENDPOINT` is explicitly set). If no endpoint is reachable, runtime stops with a network/VPN retry message.
- MIGraphX artifact preparation runs only after MIGraphX availability is confirmed, so unsupported ROCm environments still keep the existing PyTorch ROCm fallback instead of failing early in the shared artifact layer.

Verification for the pushed GUI/log pass:

```powershell
src\python3\python.exe -m py_compile src\gui\builders.py src\gui\process.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_gui_config.py tests\test_progress.py tests\test_runtime.py -q
git diff --check
```

Result:

```text
py_compile passed
71 passed before the HuggingFace download-progress follow-up
72 passed after the local HuggingFace download-progress follow-up
44 passed for backend-aware model artifact / endpoint tests
git diff --check passed with CRLF warnings only
```

Handoff notes:

- Do not commit runtime preference churn from `src/settings.yaml` unless explicitly requested; the current uncommitted `Show Log Panel: true` entry is local runtime state.
- Do not re-add `pywebview`; the active user decision is to keep the built-in Flet log panel.
- For download progress visibility in the GUI, ensure model downloads go through `DownloadProgress` or the scoped HuggingFace tqdm patch. When `HF_ENDPOINT` is unset, HuggingFace downloads try `https://hf-mirror.com` first and fall back to `https://huggingface.co`; an explicit `HF_ENDPOINT` is respected as the only endpoint. Download preparation logs print the model URL before network access starts, then run a lightweight HEAD probe with a Range GET fallback so the GUI log captures HTTP status, content length/type, final redirected URL, and connection errors before the real download begins.
- Do not reintroduce backend-blind artifact checks. `.trt` is not valid evidence for ONNX/XPU/CPU or ROCm paths, and `.mgx` is not valid evidence for CUDA/TensorRT paths.
- Keep `flet`, `flet-desktop`, and `flet-cli` out of `requirements.txt`; Flet is supplied by the vendored/built-in GUI path.

### 2026-06-30 OpenXR Screen Presets, Controller Interaction, and OSD Pass

Implemented locally in the current worktree:

- Added the OpenXR **Headset Model** settings path. GUI/settings resolve the selected model through `src/utils/xr_headset_presets.py`; presets now store recommended viewing distance only, while screen width, height, and diagonal are derived from `XR_HEADSET_HORIZONTAL_FOV_DEG = 60.0` and 16:9 geometry.
- Updated the VR/AR focal-distance reference table to 60 degree horizontal FOV. Infinite-focus devices such as Pico 4 / 4 Ultra and HTC VIVE XR Elite use 20.0 m as the practical recommended distance, producing a 23.09 m wide virtual screen.
- OpenXR startup can replace the default Y-button screen preset with the selected headset recommendation, and the viewer width clamp was raised to 30.0 m so the 20 m / 60 degree preset is not clipped.
- Screen distance labels now use actual head-to-screen center distance via `_screen_view_distance()`, so the FPS/status panel, preset OSD, screen footprint logs, and reset behavior report the same physical distance.
- Right-grip screen movement keeps the existing sphere-orbit behavior, but after orbiting the screen center, yaw/pitch automatically face the headset. `D2S_OPENXR_RIGHT_GRIP_SCREEN_ROTATION` remains disabled by default and now only controls right-grip wrist-roll mapping to `screen_roll`.
- Preset OSD above the screen now holds for 5.0 s. Its trigger key ignores the live distance suffix so head jitter does not keep it alive forever, while `_apply_preset()` clears `_preset_osd_last_key` so pressing Y to restore the same default preset still shows the OSD again.
- Keyboard placement now keeps a gap equal to 15% of screen height below the display. Screen edge laser snap release is 6 degrees. Screen and keyboard laser hit circles share the same cursor-ring model and scale by eye-to-hit distance.

Verification run during this pass:

```powershell
src\python3\python.exe -m py_compile src\utils\xr_headset_presets.py src\viewer\settings.py src\xr_viewer\implementation.py
src\python3\python.exe -m pytest tests\test_xr_headset_presets.py tests\test_viewer_settings.py tests\test_keyboard_screen_preset.py -q
src\python3\python.exe -m py_compile src\xr_viewer\implementation.py
src\python3\python.exe -m py_compile src\xr_viewer\overlay.py
src\python3\python.exe -m py_compile src\xr_viewer\core_screen_control.py src\xr_viewer\overlay.py
src\python3\python.exe -m pytest tests\test_keyboard_screen_preset.py -q
git diff --check
```

Result:

```text
py_compile passed
33 passed for headset/viewer/keyboard preset tests before the final focused OSD fixes
26 passed for focused keyboard/screen preset tests after the final OSD fixes
git diff --check passed with CRLF warnings only
```

Handoff notes:

- Do not hand-maintain screen width/height for headset presets. The source of truth is recommended viewing distance plus `XR_HEADSET_HORIZONTAL_FOV_DEG`.
- Do not reintroduce live distance into the preset OSD trigger key; it should only affect displayed text, not reset the fade timer.
- Do not change right-grip screen movement to flat translation. Left grip translates; right grip orbits around the head and then faces the screen back toward the head.

### 2026-06-30 OpenXR GPU Upload and Provider Timing Pass

Implemented and pushed:

```text
94f22d6 perf: optimize gpu runtime upload paths
```

Summary:

- Added shared `src/viewer/gl_texture_uploader.py` with a common CUDA tensor -> OpenGL texture uploader used by OpenXR runtime-eye upload and local viewer runtime RGBA upload.
- Standardized upload policy: CUDA/GL image texture first, PBO as GPU fallback, CPU fallback only with red console warning and recorded reason.
- OpenXR full synthesis path now expects runtime-produced uint8/RGBA CUDA eye tensors where possible so the viewer only performs texture upload and present, avoiding repeated float clamp/multiply/to-uint8 work at the viewer boundary.
- TensorRT native now records real CUDA event timing for depth preprocess/model/normalize/upsample/postprocess so CPU enqueue timing is not mistaken for GPU model time.
- MIGraphX ROCm provider imported the ROCm7 build rule from the legacy v2.5 engine: try FP8 autocast first, fall back to FP16, and skip quantization for force-FP32 models.
- Docs updated so `docs/28` is the normative rule set, `docs/26` records engineering implementation/status, and this handoff tracks remaining work.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\depth_provider.py src\stereo_runtime\pipeline.py src\stereo_runtime\providers\amd\migraphx.py src\stereo_runtime\providers\nvidia\tensorrt_native.py src\stereo_runtime\runtime.py src\viewer\viewer.py src\viewer\gl_texture_uploader.py src\xr_viewer\core_runtime_eye.py src\xr_viewer\implementation.py tests\test_openxr_runtime.py tests\test_pytorch_rocm_provider.py
src\python3\python.exe -m pytest tests\test_openxr_runtime.py tests\test_viewer_runtime.py tests\test_runtime_pipeline.py tests\test_runtime_openxr.py tests\test_pytorch_rocm_provider.py -q
```

Result:

```text
py_compile passed
72 passed
```

Handoff notes:

- `runtime_direct_opengl_pbo active (CUDA)` means image texture path was not active and PBO was used as GPU fallback; it does not by itself mean CPU fallback. The next bug hunt should preserve the root error for image texture registration/copy failure instead of silently disabling it.
- OpenXR `xr_wait` / `xr_poll` / `xr_submit` / present timing must be analyzed as display runtime scheduling, not as StereoRuntime depth/synthesis/SBS compute time.
- TensorRT ORT remains the next concrete depth-provider zero-copy task because it still performs CPU numpy input/output staging.

### 2026-06-30 Realtime CPU Transfer Audit

A realtime-path scan was run after adding red CPU fallback / CPU transfer console warnings:

```powershell
rg -n "\.cpu\(\)\.numpy|\.cpu\(\).*\.numpy|OrtValue\.numpy|\.numpy\(\)|torch\.from_numpy|np\.asarray\(" src\viewer src\xr_viewer src\streaming src\stereo_runtime
```

Classification:

- Already red-warning covered realtime fallback paths: local `StereoWindow` RGB/depth texture CPU upload, OpenXR RGB+depth CPU upload, OpenXR runtime-eye CPU GL upload, OpenXR D3D11 CPU `UpdateSubresource`, Metal viewer RGB/depth upload, `runtime_output_to_numpy()`, legacy SBS streaming output, ONNX depth input `tensor.detach().cpu().numpy()`, and TensorRT ORT input/output CPU staging.
- Offline/debug/report paths that are not realtime display or inference bottlenecks: `src/stereo_runtime/io.py` image save helpers and `src/stereo_runtime/report.py` report canvas export.
- GPU zero-copy optimization targets that still need implementation, not just warnings: TensorRT ORT input binding currently converts CUDA tensor to CPU numpy before `OrtValue.ortvalue_from_numpy`; TensorRT ORT output currently calls `OrtValue.numpy()` then `torch.from_numpy()`, putting depth back on CPU; ONNX depth provider should prefer DLPack / CUDA OrtValue path whenever available; legacy streaming/output numpy conversion remains inherently CPU until streamer transport accepts GPU frames.
- Non-problem CPU conversions: `np.asarray()` on static matrices, UI/model metadata, glTF/environment data, or already-CPU inputs are not treated as realtime GPU fallback unless adjacent warning code marks an actual GPU-to-CPU transfer.

Follow-up rule:

- Red warnings are diagnostic visibility only. Any warning in the depth provider or display upload hot path should be treated as a bug for the 100% GPU zero-copy target unless the mode explicitly requires CPU transport, such as legacy MJPEG.

### 2026-06-29 Remaining SBS FPS Gap Investigation

Latest user log after CUDA runtime sync and CPU-side inference cleanup:

```text
StereoRuntime frame refresh: total_ms=3.4-3.7 depth_total_ms=1.7 synthesis_ms=1.7-2.0 pack_ms=0.0 stage_sbs=0.1
WindowsCaptureCUDA capture_fps=72.4-82.3 copy_ms=0.04-0.07 enqueue_ms=0.04-0.05 handler_ms=0.09-0.13
```

Interpretation:

- `StereoRuntime` compute is not the current 20 FPS limiter. A 3.4-3.7ms runtime frame implies the depth+synthesis+SBS compute path can run far above 20 FPS.
- SBS packing is not the bottleneck: `sbs_backend=triton_half_sbs`, `stage_sbs=0.1`, and `pack_ms=0.0`.
- Capture is also above the reported 20 FPS symptom in this run, though still below the 120Hz monitor cadence: `WindowsCaptureCUDA` reports roughly 72-82 FPS.
- The likely bottleneck is after runtime compute: queue handoff, viewer/OpenXR/local display submit, texture upload, present/vsync, or a misleading FPS counter that is measuring presentation rather than runtime production.

Next trace points:

```text
RuntimePipelineLoop raw dequeue
StereoRuntime.process_* call boundary
runtime_q put / overwrite / blocking time
viewer or OpenXR runtime_q dequeue
texture upload time
swapchain/local present/submit time
reported SBS FPS counter source
```

Do not spend more time optimizing depth, synthesis, or SBS pack until the outer-loop timings prove they are the limiter.

### 2026-06-29 CUDA Runtime Backpressure Specification

Documented the high-refresh capture finding across the canonical and engineering handoff docs.

Finding:

- Direct `wc_cuda` capture can reach high-refresh cadence in isolation.
- Main pipeline stages before full runtime consumption can hold the expected capture cadence.
- Full CUDA runtime processing can still reduce capture event rate when depth/synthesis/pack GPU work is submitted asynchronously without a per-frame completion boundary.
- `runtime_sync` improving capture cadence while producing producer-side raw overwrite/drop confirms the fix belongs to runtime scheduling/backpressure, not to OpenXR viewer submit or capture callback cost.
- `overwrite/drop` is the expected latest-frame loss point; `drain_drop=0` only means stale frames were overwritten before the consumer drain stage.

Spec updates:

- `docs/28-Realtime-2d-to-3d-specification.md`: added global realtime scheduling and backpressure rules.
- `docs/26-desktop2stereo-engineering-design-specification.md`: added engineering rules for CUDA runtime sync, latest-frame queues, and log interpretation.
- `docs/00-api-handoff-progress.md`: added this handoff entry plus future-work guardrails.

Verification:

```powershell
git diff --check -- docs\28-Realtime-2d-to-3d-specification.md docs\26-desktop2stereo-engineering-design-specification.md docs\00-api-handoff-progress.md
```

Expected result:

```text
passed; CRLF warnings only
```

### 2026-06-29 OpenXR Realtime Hot Path and Capture Diagnostics

This pass removed the per-frame CPU synchronization from scene-cut temporal reset and then traced the remaining OpenXR SBS refresh bottleneck with live runtime logs.

Implemented in this follow-up:

- Reworked auto scene reset into GPU-side temporal alpha gating so the realtime path no longer needs a Python `if changed` branch, `.item()`, or CUDA event query to decide whether to reset temporal history.
- Removed `reset_cooldown_frames` from the realtime path instead of keeping a CPU-owned cooldown that cannot know GPU-only scene-cut state without synchronization.
- Kept temporal blending active when GPU-side scene reset is unavailable; do not fall back to CPU scene reset in the realtime path.
- Fixed GUI fallout from the cooldown removal, including the stale `reset_cooldown_label` reference.
- Moved the GUI Scene Reset Threshold control beside Temporal Strength and moved Advanced Stereo beside Hole Fill Mode.
- Enabled existing `FPSBreakdown` output from `D2S_OPENXR_DEBUG=1` and from the existing source-health tick, so OpenXR diagnosis prints capture, raw queue, runtime, and timing breakdowns without adding a second counter stack.
- Added a source-level `WindowsCaptureCUDA` console line inside the capture callback:

```text
[WindowsCaptureCUDA] capture_fps=<fps> frames=<count> monitor=<index> mode=<mode>
```

- Fixed the 4K capture preprocess fast-path miss where `torch.device("cuda") != torch.device("cuda:0")` caused CUDA frames to skip `triton_bgr_resize_norm` and fall back to the much slower PyTorch BGR/RGB/float conversion path.
- Generalized the device comparison helper for `cpu`, `cuda`, `mps`, `xpu`, and `hip` so bare device names and `:0` device names are treated consistently.

Live-run findings:

- Before the device comparison fix, `FPSBreakdown` showed `pre=torch_bgr_norm` and `rt_cap2rgb` around `36-39ms`, while `rt_call` was usually around `10-13ms`; the main bottleneck was capture preprocess, not depth/stereo synthesis.
- After the fix, `pre=triton_bgr_resize_norm` was restored and `rt_cap2rgb` dropped to roughly `0.3-0.4ms`.
- The later run showed `WindowsCaptureCUDA` / source capture rate around `21-24fps` in that test window, so source event rate is now a separate variable to validate against a known 60Hz changing source.
- `stage_scene=0.0` in refresh logs confirms the previous scene-cut CPU sync point is no longer the observed hot-path cost.

Verification:

```powershell
src\python3\python.exe -m py_compile src\capture\preprocess.py src\app_runtime\runtime_callbacks.py src\app_runtime\runtime_context.py src\capture\backends\windows_capture_event.py tests\test_capture_preprocess.py tests\test_windows_capture_event.py
src\python3\python.exe -m pytest tests\test_capture_preprocess.py tests\test_capture_preprocess_triton.py
src\python3\python.exe -m pytest tests\test_windows_capture_event.py
git diff --check
```

Result:

```text
py_compile passed
17 passed for capture preprocess / Triton preprocess
7 passed for Windows capture event tests
git diff --check passed with CRLF warnings only
```

Handoff notes:

- The next performance question should be tested with the new `[WindowsCaptureCUDA] capture_fps=...` console line while capturing a known 60Hz source; do not infer source FPS from `StereoRuntime frame refresh`, which is a periodic timing refresh log, not a per-frame FPS line.
- If source capture is confirmed below 60fps with a known 60Hz changing input, inspect the third-party `wc_cuda` / Windows Graphics Capture event cadence before changing stereo synthesis settings.

### 2026-06-29 Depth Strength Runtime Range Cleanup

- Unified `Depth Strength` to use the same `0.00` to `0.50` value across GUI, `settings.yaml`, hot-save, hot-reload, and runtime consumers.
- Changed the GUI dropdown to `0.05` steps and kept the backend shift formulas unchanged.
- Updated GUI stereo presets and `Depth Quick` mappings to `Soft=0.20`, `Standard=0.25`, and `Enhanced=0.30`; current `src/settings.yaml` now uses `Depth Strength: 0.25`.
- Verified with `py_compile`, `tests/test_gui_config.py`, `tests/test_hot_reload.py`, and `git diff --check`.

### 2026-06-29 Cinema Hole-Fill Preset Lightening

- Kept `Traditional / Fastest` as the high-FPS no-extra-hole-fill stereo mode.
- Changed the `Cinema / Balance` preset and reset defaults to a lighter hole-fill strategy: `hole_fill_radius=1`, `hole_fill_strength=0.60`, `mask_feather_radius=1`, `edge_dilation=1`, and `temporal_strength=0.25`.
- Added `Hole Fill Radius` and `Hole Fill Strength` emission to GUI collect/hot-save so preset values reach runtime instead of silently falling back to the old `3 / 1.0` defaults.
- Mapped `Image / High Quality` to the `quality` content-aware hole-fill mode while leaving it outside the realtime default path.

### 2026-06-29 GUI FP16 Default Enablement

- Enabled GUI `FP16` by default and changed config load to honor `settings.yaml` instead of forcing the checkbox back to the default each time.
- Kept the existing MPS save guard so FP16 is still written as off for MPS devices.
- Updated current `src/settings.yaml` to `FP16: true` for CUDA/TensorRT use.

### 2026-06-29 GUI FP16 ONNX Dtype Wiring

- Wired GUI `FP16` into `StereoRuntimeConfig.onnx_dtype`: checked maps to `fp16`, unchecked maps to `fp32`.
- Updated runtime artifact report paths so `onnx_path` and `trt_engine_path` follow the selected dtype instead of always pointing at FP16 artifacts.
- Added `DepthProviderConfig.onnx_dtype` and passed it through ONNX CUDA, TensorRT ORT, and native TensorRT provider artifact preparation.
- With `FP16` unchecked, missing `model_fp32_<HxW>.onnx` now triggers the existing FP32 export/probe path instead of reusing an existing FP16 ONNX artifact via `auto`.

### 2026-06-29 TensorRT Startup Console Logging

- Restored TensorRT startup console visibility for native TensorRT and ONNXRuntime TensorRT EP first-load paths.
- Native TensorRT now prints engine build start/ready messages plus first-load details: engine path, ONNX path, dtype, input size, CUDA graph/profile flags, and TensorRT DLL dirs.
- ONNXRuntime TensorRT EP now prints first-load details: ONNX path, dtype, TensorRT cache path, active/available providers, and TensorRT DLL dirs.

### 2026-06-29 FP16 Launch Persistence Fix

- Removed the GUI subprocess-start cleanup that reset `FP16` back to `DEFAULTS["FP16"]` after launch.
- Kept one-shot recompile flags resetting after launch, but preserved the user-selected FP16 checkbox state in `settings.yaml`.
- Updated current `src/settings.yaml` to `FP16: false` to match the user's cancellation test state and avoid an initial FP16 TensorRT load before the FP32 runtime update.
- Deferred ONNX / TensorRT provider `load()` when artifact input size has not been selected yet, preventing generic runtime preload from loading the default FP16 artifact before first-frame `_ensure_artifacts_for_input()` selects FP32.
- Expanded slow-frame runtime logs with `depth_pre_ms`, `depth_post_ms`, and `depth_gap_ms` so depth bottlenecks can be separated into preprocess, model inference, postprocess, and unaccounted load/sync overhead.
- Added steady-state runtime timing refresh logs: frames above `D2S_SLOW_RUNTIME_LOG_MS` still print as `slow frame`, while latest frames below the slow threshold print as `frame refresh` every `D2S_RUNTIME_FRAME_LOG_REFRESH_S` seconds by default.

### 2026-06-29 Legacy IPD / Parallax Multiplier Cleanup

Continued compliance against `docs/26-desktop2stereo-engineering-design-specification.md`, with runtime semantics governed by `docs/28-Realtime-2d-to-3d-specification.md`. This pass removed the remaining runtime/config/test protection for the old physical-IPD-style parallax chain.

Implemented in this follow-up:

- Removed `ipd` from `AppModeSettings`, OpenXR mode config, runtime context creation, `OpenXRRuntimeConfig`, and `OpenXRViewerCore` construction.
- Removed `IPD` lazy export from `utils`, removed `ipd` from `RuntimeExports` and `ViewerSettings`, and deleted `IPD` from `src/settings.yaml`.
- Changed the D3D11 RGB+depth fallback shader path so `D3D11NativeRenderer.render_eye()` consumes a resolved per-eye `eye_offset`; the shader uniform is now `parallaxOffset`, not `eyeOffset`, and no D3D11 fallback code computes `eye_sign * ipd * 0.5`.
- Renamed local/Metal viewer residuals from `ipd_uv` / `eyeOffset` to `parallax_budget_uv` / `parallaxOffset` while preserving the current viewer fallback displacement behavior.
- Renamed legacy MJPEG SBS helper parameter `ipd_uv` to `parallax_budget_uv` without changing its output math.
- Removed `IPD`, `Stereo Scale`, and `Max Shift Ratio` from active `src/settings.yaml` and removed old-field compatibility inputs from adapter/hot-reload tests.
- Updated `docs/26-desktop2stereo-engineering-design-specification.md` so it no longer says IPD / Stereo Scale / Max Shift Ratio continue to be read as compatibility inputs; the engineering spec now records that those compatibility entries are cleaned.

Compliance scan result:

```text
rg -n "\bipd\b|IPD|ipd_uv|eyeOffset" src tests
```

The scan now returns only test negative assertions. Production `src` has no positive `ipd`, `IPD`, `ipd_uv`, or `eyeOffset` runtime/config reference.

Verification:

```powershell
src\python3\python.exe -m py_compile src\app_runtime\app_runner.py src\app_runtime\mode_configs.py src\app_runtime\runtime_context.py src\main.py src\streaming\legacy_sbs.py src\utils\__init__.py src\utils\runtime_exports.py src\viewer\metal_viewer.py src\viewer\settings.py src\xr_viewer\d3d11_native_renderer.py src\xr_viewer\implementation.py src\xr_viewer\openxr_runtime.py src\xr_viewer\overlay.py
src\python3\python.exe -m pytest tests\test_app_runner.py tests\test_mode_configs.py tests\test_openxr_runtime.py tests\test_viewer_settings.py tests\test_viewer_runtime.py tests\test_openxr_state.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_legacy_sbs.py
src\python3\python.exe -m pytest
```

Result:

```text
py_compile passed
81 passed
514 passed, 1 warning
```

Handoff notes:

- `Depth Strength` remains intentionally user-facing and still participates as depth response strength; it is not treated as part of the removed physical IPD multiplier chain.
- D3D11 native OpenXR direct shader parity with the richer OpenGL DIBR shader remains a separate follow-up. This pass only removed stale IPD semantics from the D3D11 fallback and constructor flow.
- Remaining compatibility cleanup should focus on old snapshot/API aliases and debug-only fallback keys, not on IPD / Stereo Scale / Max Shift Ratio runtime inputs.

### 2026-06-28 GUI Render Size Policy User-Layer Convergence

Continued compliance against `docs/28-Realtime-2d-to-3d-specification.md` using `docs/26-desktop2stereo-engineering-design-specification.md` as the engineering checklist. This pass focused on the Render Size / 4K Tier contract: users should see only fixed 4K render-size tiers, not `native` / `fixed` / `dynamic` policy choices.

Implemented in this follow-up:

- Collapsed GUI render-size policy display/parsing helpers so the user layer always resolves to `scaled`.
- Changed GUI config collection to save `"Render Size Policy": "scaled"` directly instead of round-tripping the hidden dropdown value.
- Kept low-level `RenderSizePolicy` parsing intact as a legacy configuration compatibility path.
- Updated GUI regression tests to assert that the visible render-size surface exposes only fixed tier labels and does not write old policy choices back to settings.

Verification:

```powershell
src\python3\python.exe -m py_compile src\gui\handlers.py src\gui\config_mgr.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_gui_config.py tests\test_render_size.py -q
```

Result:

```text
56 passed
```

### 2026-06-28 Docs 26 Engineering Flow Alignment

Reduced `docs/00-api-handoff-progress.md` Future Work to a task queue and moved detailed compatibility/migration rules into `docs/26-desktop2stereo-engineering-design-specification.md`.

Implemented in this follow-up:

- Added a docs/26 document-position section: docs/28 is the final runtime process spec, docs/26 is the engineering implementation and migration spec, docs/00 is only handoff/progress.
- Added a docs/26 mapping table from each docs/28 runtime step to current implementation modules and remaining migration boundaries.
- Added a docs/26 compatibility cleanup section covering snapshot aliases, debug fallback keys, legacy parallax multipliers, historical render-scale paths, and D3D11 direct shader parity.
- Replaced the detailed docs/00 Future Work explanation with a concise current task queue that points to docs/26 for engineering detail.

Verification:

```powershell
git diff --check -- docs\26-desktop2stereo-engineering-design-specification.md docs\00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

### 2026-06-28 Docs 28 Canonical Runtime Spec Promotion

Promoted `docs/28-Realtime-2d-to-3d-specification.md` to the official final runtime process specification and made `docs/25-2d-to-3d-runtime-specification.md` obsolete.

Implemented in this follow-up:

- Updated `docs/28-Realtime-2d-to-3d-specification.md` so it carries the final runtime-flow authority while preserving its structured eleven-step format.
- Ensured the former `docs/25` normative content is represented in docs/28: render_size / 4K tier semantics, normalized-depth parallax budget, depth provider contract, RuntimeSettingsSnapshot, hot-reload classification, OpenXR RGB+depth direct/full synthesis, output packing contracts, and runtime debug/result fields.
- Marked `docs/25-2d-to-3d-runtime-specification.md` as obsolete at the top of the file instead of deleting it.
- Updated `docs/26-desktop2stereo-engineering-design-specification.md` and `docs/README.md` so current spec references point to docs/28.

Verification:

```powershell
git diff --check -- docs\28-Realtime-2d-to-3d-specification.md docs\25-2d-to-3d-runtime-specification.md docs\26-desktop2stereo-engineering-design-specification.md docs\README.md docs\00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

### 2026-06-28 Docs 25 Final Runtime Flow Cleanup

Accepted the split between the canonical runtime process specification and the engineering/migration specification.

Implemented in this follow-up:

- Cleaned `docs/25-2d-to-3d-runtime-specification.md` so it describes the final target runtime flow, not intermediate compatibility or migration steps.
- Removed legacy/current-code wording from the `docs/25` runtime flow, OpenXR RGB+depth direct path, RuntimeSettingsSnapshot field list, hot-reload debug fields, IPD handling, render-scale semantics, and parallax implementation section.
- Renamed the mode matrix row from `openxr traditional` to `openxr_rgb_depth_direct` and clarified that the direct shader path must consume a spec-derived shader uniform snapshot.
- Removed `render_size_policy` from the `docs/25` final RuntimeSettingsSnapshot/debug field contract; remaining compatibility handling stays in `docs/26` and Future Work.
- Left `docs/26-desktop2stereo-engineering-design-specification.md` as the place for current implementation state, legacy adapters, compatibility fallbacks, and cleanup tracking.

Verification:

```powershell
git diff --check -- docs\25-2d-to-3d-runtime-specification.md docs\00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

### 2026-06-28 Docs 25/26 Compliance Alignment Follow-up

Checked `docs/25-2d-to-3d-runtime-specification.md` and `docs/26-desktop2stereo-engineering-design-specification.md` together after the render-size, parallax, capture metadata, RuntimeSettingsSnapshot, OpenXR adapter, and stream transport passes.

Outcome:

- Excluding the separately recorded compatibility-cleanup work, the main `docs/25` runtime logic is now considered basically satisfied by the current implementation and tests.
- `docs/26` had stale "current gaps / priorities" wording that still described already-completed work as future implementation; updated it to a compliance-status table.
- `docs/26` now explicitly treats `docs/25` as canonical for parallax budget, render_size, and OpenXR output semantics when the two documents differ.
- Remaining non-cleanup gaps are now narrowed to GUI live hot-save directly emitting `RuntimeSettingsSnapshot`, D3D11 native direct shader parity with OpenGL DIBR quality semantics, hardware validation for CUDA/ROCm zero-copy, and future RTMP/low-latency stream transport work.
- A broader keyword sweep found no new `docs/25` vs `docs/26` semantic conflict in production code. It did find stale `docs/26` wording for `CapturedFrame`, structured output size consumption, and parallax migration status; those sections were updated.
- The sweep also found non-canonical older docs that still explain legacy `Depth Strength` / `IPD` / `Stereo Scale` / `Max Shift Ratio` semantics. These implementation-experience records have been moved to `docs/archive/implementation-experience/` and are not current normative specs. A later documentation-cleanup pass can decide whether to delete or rewrite them.
- `src/settings.yaml` currently contains `Render Scale: 1.0` in the local working tree. Runtime/GUI normalization maps unknown old values back to the default `4K / 3840x2160`, but the file value itself is not spec-clean and should be normalized when user-local settings are safe to touch.

Verification:

```powershell
git diff --check -- docs\26-desktop2stereo-engineering-design-specification.md docs\00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

### 2026-06-28 Runtime Settings Spec Alias Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` RuntimeSettingsSnapshot compliance pass by accepting the spec-facing field names that differ from older internal runtime config names.

Implemented in this follow-up:

- Added `RuntimeSettingsSnapshot.parallax_budget_preset` as the spec-facing alias for internal `parallax_preset`.
- Added `RuntimeSettingsSnapshot.temporal_enabled` as the spec-facing alias for internal `temporal`.
- Kept existing internal fields supported; aliases map to runtime config only when the internal field is not explicitly set.
- Included the alias fields in hot-reload classification, temporal-reset handling, active settings merge, and active settings debug output.
- Added regression coverage proving alias-only snapshots update runtime config, reset temporal state, and surface the alias names through structured hot-reload result metadata and debug compatibility fields.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\settings_snapshot.py src\stereo_runtime\runtime.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_runtime_pipeline.py -q
git diff --check -- src\stereo_runtime\settings_snapshot.py src\stereo_runtime\runtime.py tests\test_settings_snapshot.py
```

Result:

```text
46 passed; diff-check passed with CRLF warnings only
```

### 2026-06-28 Runtime Settings Result Fields Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` compliance pass by promoting hot-reload/runtime-settings result metadata out of debug-only storage.

Implemented in this follow-up:

- Added structured `active_settings_version`, `hot_reload_class`, and `hot_reload_changed_fields` fields to `StereoRuntimeResult` and `OpenXRRuntimeResult`.
- Kept the legacy `debug_info["active_settings_version"]`, `debug_info["hot_reload_class"]`, and `debug_info["hot_reload_changed_fields"]` keys for compatibility.
- Updated RGB runtime and OpenXR runtime paths to populate the structured fields from the runtime's applied settings state.
- Updated `openxr_result_from_stereo_result()` so OpenXR full-synthesis conversion preserves the structured settings metadata.
- Added regression coverage for RGB results, OpenXR RGB+depth results, and full-synthesis conversion preserving the structured fields.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_runtime_openxr.py tests\test_settings_snapshot.py tests\test_runtime_pipeline.py -q
git diff --check -- src\stereo_runtime\runtime.py tests\test_runtime_openxr.py docs\00-api-handoff-progress.md
```

Result:

```text
44 passed; diff-check passed with CRLF warnings only
```

### 2026-06-28 Render Size Aspect Protection Specification Follow-up

Clarified the canonical `docs/25-2d-to-3d-runtime-specification.md` aspect-ratio rules after the fixed-tier `render_size` semantics were finalized.

Implemented in this follow-up:

- Kept ultrawide aspect protection as a valid Parallax Budget rule, but moved its input explicitly to resolved `render_size` rather than raw `capture_size`.
- Clarified that 4K-tier input classification and aspect-factor budget protection are separate stages: `capture_size` decides 4K-tier eligibility, while `render_size` decides short-side budget and `aspect_factor`.
- Documented that 4K ultrawide inputs such as `3840x1600` may map to fixed 16:9 tiers such as `2560x1440`, which should not trigger ultrawide budget reduction after mapping.
- Documented that non-4K ultrawide inputs such as `3440x1440` keep `render_size = capture_size` and still trigger aspect protection when the final render aspect exceeds 2:1.
- Updated window-capture budget recalculation thresholds to refer to `render_size` short-side and final `render_size` aspect, avoiding raw capture aspect jitter in fixed-tier cases.

Verification:

```powershell
git diff --check -- docs/25-2d-to-3d-runtime-specification.md docs/00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

### 2026-06-28 Render Scale Fixed Tier Runtime Follow-up

Continued the render-size compliance pass by removing the remaining continuous-scale and threshold behavior from the runtime and GUI render-scale path.

Implemented in this follow-up:

- Changed `RenderSizeConfig.scale_factor` semantics from a float scale to a canonical fixed tier label (`1K / 1920x1080`, `2K / 2560x1440`, `3K / 3200x1800`, `4K / 3840x2160`).
- Removed runtime threshold mapping for `0.58` / `0.75` / `0.92`; unknown or numeric legacy values now fall back to the default 4K tier instead of selecting a lower tier.
- Implemented direction-independent 4K-tier input detection, including portrait 4K, DCI 4K, 16:10 4K, and 3840x1600 ultrawide, while excluding 1440p ultrawide and narrow tall windows.
- Preserved portrait orientation when resolving fixed tiers, so 2160x3840 can resolve to 1080x1920 / 1440x2560 / 1800x3200 / 2160x3840.
- Updated GUI render-scale load/save to store canonical tier labels instead of floats.
- Updated pipeline debug `stereo_render_scale` to report the tier label, matching the structured fixed-tier contract used by local, stream, and OpenXR render-size resolution.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\render_size.py src\gui\handlers.py src\gui\config.py src\stereo_runtime\pipeline.py src\stereo_runtime\settings_snapshot.py tests\test_render_size.py tests\test_runtime_pipeline.py tests\test_viewer_settings.py tests\test_settings_snapshot.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_render_size.py tests\test_runtime_pipeline.py tests\test_viewer_settings.py tests\test_settings_snapshot.py tests\test_gui_config.py tests\test_runtime_context.py -q
```

Result:

```text
95 passed
```

Notes / next improvements:

- OpenXR downsampling uses the same `RuntimePipelineContext.render_size_config -> resolve_render_size() -> capture_frame_to_rgb(target_resolution=render_size)` path, so this fixed-tier behavior applies there without a separate OpenXR-only scale branch.

### 2026-06-28 Render Size 4K Tier Specification Follow-up

Updated the canonical render-size specification after confirming `native` / `fixed` / `dynamic` are no longer user-facing policy requirements. `Render Scale` is now documented as a stable 4K-tier selection signal, not a continuous scale for arbitrary input sizes.

Implemented in this follow-up:

- Replaced the user-facing `native` / `fixed` / `dynamic` Render Size Policy language in `docs/25-2d-to-3d-runtime-specification.md` with the current Render Size / 4K Tier contract.
- Clarified that non-4K inputs keep `capture_size`, while 4K-tier inputs map GUI `Render Scale` labels to stable 3840x2160 / 3200x1800 / 2560x1440 / 1920x1080 tiers.
- Documented that GUI Render Scale uses fixed resolution labels (`1K / 1920x1080`, `2K / 2560x1440`, `3K / 3200x1800`, `4K / 3840x2160`); numeric thresholds such as `0.58` / `0.75` / `0.92` are not valid tier-selection semantics.
- Added the portrait 4K equivalent tiers: 2160x3840 / 1800x3200 / 1440x2560 / 1080x1920.
- Replaced the old landscape-only 4K detection pseudocode with a 4K-tier input rule based on full/ultrawide 4K dimensions or near-4K pixel area: `long_side >= 3840 and short_side >= 1600`, or `pixels >= 3840*2160*0.85 and long_side >= 3200 and short_side >= 1600`.
- Clarified that 3840x2160, 2160x3840, 4096x2160, 3840x2400, and 3840x1600 enter 4K tier, while 2560x1440, 3440x1440, 1080x1920, and 1000x3000 stay at capture size.
- Updated `docs/26-desktop2stereo-engineering-design-specification.md` so the implementation gap and follow-up priority no longer point to user-selectable `native/scaled/fixed/dynamic` policy modes.

Verification:

```powershell
git diff --check -- docs/25-2d-to-3d-runtime-specification.md docs/26-desktop2stereo-engineering-design-specification.md docs/00-api-handoff-progress.md
```

Result:

```text
passed; CRLF warnings only
```

Notes / next improvements:

- This is a documentation/specification correction only. Runtime code already follows the 4K-tier scaled behavior for current tests; a follow-up can add explicit portrait 4K tests if we want to lock that contract in code.

### 2026-06-28 Render Size Policy GUI Persistence Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` render-size policy compliance pass by fixing GUI load/save persistence for the independent `Render Size Policy` layer.

Implemented in this follow-up:

- Updated GUI config load so `render_policy_dd` reflects `cfg["Render Size Policy"]` instead of always displaying `scaled`.
- Updated GUI config save so `Render Size Policy` is derived from the selected render policy dropdown value instead of being hard-coded to `scaled`.
- Added GUI config regression coverage proving the policy dropdown uses the configured value for load/save and guarding against reintroducing the hard-coded `scaled` path.

Verification:

```powershell
src\python3\python.exe -m py_compile src\gui\config_mgr.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_gui_config.py tests\test_render_size.py tests\test_runtime_context.py -q
```

Result:

```text
54 passed
```

Notes / next improvements:

- This closes the config load/save overwrite bug for non-`scaled` compatibility values. User-selectable `native` / `fixed` / `dynamic` policy modes are no longer a product requirement; current spec direction is `Render Scale` as a 4K-tier selection signal, with non-4K inputs preserving `capture_size`.

### 2026-06-28 Output Transport Debug Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` transport-layer compliance pass by making runtime pipeline debug report the resolved output transport instead of deriving `transport` only from `run_mode`.

Implemented in this follow-up:

- Updated `_attach_pipeline_debug()` so `debug_info["transport"]` prefers the explicit `RuntimePipelineContext.output_transport` value.
- Preserved fallback labels for older contexts without explicit transport: `openxr_swapchain` for OpenXR and `local_window` otherwise.
- Added regression coverage proving `encoded_stream` remains the primary `transport` debug value for network-stream style contexts, while OpenXR/local fallback behavior remains unchanged.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\pipeline.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime_pipeline.py tests\test_runtime_context.py tests\test_mode_configs.py -q
src\python3\python.exe -m pytest tests\test_capture_runners.py tests\test_capture_metadata.py tests\test_capture_session.py tests\test_windows_capture_event.py tests\test_capture_preprocess.py tests\test_runtime_pipeline.py tests\test_mjpeg_streamer.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py tests\test_mode_configs.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_openxr_runtime.py tests\test_runtime_callbacks.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
23 passed
224 passed
```

Notes / next improvements:

- `application_runtime_target` and `output_transport` are still also emitted as explicit debug fields for readability, but the canonical `transport` field now reflects the resolved transport contract when the context provides one.

### 2026-06-28 Polling Capture Copy Metadata Follow-up

Continued the `docs/26-desktop2stereo-engineering-design-specification.md` capture-copy compliance pass by making polling capture backends explicitly report their non-zero-copy status, matching the event capture metadata contract.

Implemented in this follow-up:

- Updated `PollingCaptureRunner` to attach `metadata["zero_copy"] = False` when wrapping polling backend frames into `CapturedFrame`.
- Kept `FrameCopyMode.COPY` for polling backends, making DesktopDuplication / DXCamera-style paths explicit copy paths until hardware-backed zero-copy validation proves otherwise.
- Added polling runner regression coverage for `CapturedFrame` metadata: backend name, copy mode, zero-copy flag, capture tool, raw device, dtype, and capture size.
- Added pipeline regression coverage proving polling capture metadata reaches runtime debug fields as `capture_copy_mode="copy"` and `capture_zero_copy=False`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\capture\runners.py tests\test_capture_runners.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_capture_runners.py tests\test_capture_metadata.py tests\test_capture_session.py tests\test_windows_capture_event.py tests\test_runtime_pipeline.py -q
src\python3\python.exe -m pytest tests\test_capture_runners.py tests\test_capture_metadata.py tests\test_capture_session.py tests\test_windows_capture_event.py tests\test_capture_preprocess.py tests\test_runtime_pipeline.py tests\test_mjpeg_streamer.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py tests\test_mode_configs.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_openxr_runtime.py tests\test_runtime_callbacks.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
20 passed
222 passed
```

Notes / next improvements:

- Hardware-backed CUDA/ROCm validation is still required before any capture path can be promoted from explicit non-zero-copy metadata to true zero-copy.

### 2026-06-28 Network Stream Encoder Metadata Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` network-stream compliance pass by aligning MJPEG transport metadata with the encoded stream profile instead of the pre-encoder packed frame size.

Implemented in this follow-up:

- Kept network streaming consuming packed SBS/runtime output frames outside `stereo_runtime`; no stereo algorithm semantics moved into the transport layer.
- Updated `MJPEGStreamer.set_frame()` so cached stream HTML metadata (`WIDTH` / `HEIGHT`) reports the encoder transport size when `EncoderProfile.resize_width` / `resize_height` are set.
- Kept resize and RGB/BGR/BGRA conversion inside `MJPEGStreamer._prepare_frame_for_jpeg()`, preserving `EncoderProfile` as the transport-side contract.
- Hardened `MJPEGStreamer.stop()` so tests and callers can close a streamer that was constructed but not started without hanging in WSGI shutdown.
- Added MJPEG streamer regression coverage for resized transport metadata and no-resize packed-frame metadata.

Verification:

```powershell
src\python3\python.exe -m py_compile src\streaming\mjpeg_streamer.py tests\test_mjpeg_streamer.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py
src\python3\python.exe -m pytest tests\test_mjpeg_streamer.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py -q
src\python3\python.exe -m pytest tests\test_mjpeg_streamer.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py tests\test_mode_configs.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_runtime_callbacks.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
9 passed
203 passed
```

Notes / next improvements:

- MJPEG/legacy stream now has an explicit transport metadata contract for encoder resize. RTMP remains a separate window-capture based legacy stream path and can be audited independently if it becomes part of the runtime-result transport contract.

### 2026-06-28 Settings YAML Hot Reload Snapshot Queue Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` and `docs/26-desktop2stereo-engineering-design-specification.md` hot-reload compliance pass by moving the main-process `settings.yaml` compatibility hot-reload path onto `RuntimeSettingsSnapshot + settings_update_q` instead of directly mutating `StereoRuntime`.

Implemented in this follow-up:

- Split `StereoHotReloader` into `poll_settings_snapshot_if_needed()` plus `log_settings_snapshot()`, while keeping `apply_if_needed()` as the direct-apply compatibility API for older callers/tests.
- Updated `RuntimeCallbacks.apply_stereo_hot_reload_if_needed()` to enqueue the polled `RuntimeSettingsSnapshot` with `send_settings_snapshot()` and update OpenXR state from the same snapshot, rather than directly applying runtime config changes.
- Updated `RuntimePipelineLoop` to drain and apply settings snapshots again immediately after the hot-reload poll and before the runtime frame call, so YAML compatibility changes still take effect on a frame boundary before synthesis/rendering.
- Centralized snapshot active-preset resolution in the pipeline so fixed `stereo_preset` snapshots override stale context `stereo_active_preset`, while `auto` preserves the current active preset.
- Added regression coverage for poll-only hot reload, callback enqueue behavior, and pipeline application of queued hot-reload snapshots before runtime processing.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\hot_reload.py src\app_runtime\runtime_callbacks.py src\stereo_runtime\pipeline.py tests\test_hot_reload.py tests\test_runtime_pipeline.py tests\test_runtime_callbacks.py
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_runtime_pipeline.py tests\test_runtime_callbacks.py -q
src\python3\python.exe -m pytest tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_runtime_callbacks.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
26 passed
197 passed
```

Notes / next improvements:

- The GUI process still persists hot controls to `settings.yaml`; because GUI and runtime are separate processes, this remains the compatibility transport. The main runtime process now converts that compatibility signal into `RuntimeSettingsSnapshot` and applies it through `settings_update_q` at frame boundaries.

### 2026-06-28 Runtime Output Size Structured Consumption Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` output-size compliance pass by auditing host/viewer consumers of runtime output dimensions and locking the structured-field precedence with an OpenXR viewer regression test.

Findings in this follow-up:

- `viewer.viewer_runtime.frame_size_from_runtime_result()` already prefers `runtime_result.output_display_size` before falling back to legacy `debug_info["runtime_output_display_size"]`.
- `xr_viewer.openxr_runtime.frame_size_from_runtime_result()` already prefers `runtime_result.output_display_size` before the same legacy fallback.
- `StereoRuntimeLogger` already prefers structured `output_format`, `output_dtype`, `output_eye_size`, and `output_pack_backend` fields before legacy debug fields.
- Added OpenXR viewer coverage proving `_log_runtime_eye_stats_once()` logs structured `output_*` values even when stale `runtime_output_*` debug fields are present.

Verification:

```powershell
src\python3\python.exe -m py_compile tests\test_openxr_runtime.py
src\python3\python.exe -m pytest tests\test_openxr_runtime.py tests\test_viewer_runtime.py tests\test_session_helpers.py -q
src\python3\python.exe -m pytest tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
25 passed
192 passed
```

Notes / next improvements:

- `runtime_output_eye_size` and `runtime_output_display_size` remain published in `debug_info` only for legacy diagnostics and fallback compatibility. The host/viewer contract is now the structured `output_eye_size` / `output_display_size` fields.

### 2026-06-28 Parallax Legacy Default Exit Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` parallax compliance pass by making the normalized-depth `standard` budget model the default at the low-level synthesis and OpenXR render boundaries, not only at the host-facing runtime config layer.

Implemented in this follow-up:

- Changed `ShiftParams`, `StereoConfig`, and `OpenXRRenderConfig` defaults from `parallax_preset="legacy"` to `"standard"`.
- Changed `openxr_render_config_from_snapshot()` and `OpenXRStateController` fallback presets to `standard` when no explicit snapshot/runtime preset is available.
- Changed `resolve_parallax_budget(preset=None)` to resolve `standard`; only explicit `preset="legacy"` now enters the old `IPD * stereo_scale * depth_strength * max_shift_ratio` multiplier path.
- Changed hot-reload fallback defaults for runtimes without `parallax_preset` to `standard`.
- Updated tests so legacy IPD multiplier behavior is covered only through explicit `parallax_preset="legacy"`, while default synthesis/OpenXR debug now reports `standard`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\baseline_shift.py src\stereo_runtime\parallax.py src\stereo_runtime\synthesis.py src\stereo_runtime\openxr_render.py src\stereo_runtime\adapter.py src\stereo_runtime\openxr_state.py src\stereo_runtime\hot_reload.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_render.py tests\test_openxr_state.py tests\test_runtime_openxr.py tests\test_adapter_config.py tests\test_hot_reload.py
src\python3\python.exe -m pytest tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_render.py tests\test_openxr_state.py tests\test_runtime_openxr.py tests\test_adapter_config.py tests\test_hot_reload.py -q
```

Result:

```text
130 passed
```

Notes / next improvements:

- The legacy multiplier implementation remains available behind explicit `parallax_preset="legacy"` for compatibility and regression coverage, but is no longer selected by default in normalized-depth runtime paths.

### 2026-06-28 OpenXR Legacy Shader Uniform Structured Field Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` OpenXR RGB+depth direct-path compliance pass by promoting the legacy shader uniform bundle out of debug-only metadata while keeping old debug keys as compatibility fallbacks.

Implemented in this follow-up:

- Added `OpenXRRuntimeResult.legacy_shader_uniforms` as the structured carrier for direct-path viewer shader uniforms derived from `OpenXRRenderConfig`.
- Kept `debug_info["openxr_legacy_shader_uniforms"]` and flat `openxr_*` debug keys for compatibility, but made runtime tests assert the structured result field as the primary contract.
- Updated `CoreRuntimeEyeMixin` to prefer `runtime_result.legacy_shader_uniforms` before falling back to the debug bundle or flat legacy debug keys.
- Updated `StereoRuntimeLogger` to prefer the structured uniform field for OpenXR output logs, so logging no longer directly depends on flat `debug_info["openxr_*"]` values.
- Added regression coverage for structured viewer consumption, debug fallback retention, runtime result propagation, and logger structured-field precedence.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\stereo_runtime\session_helpers.py src\xr_viewer\core_runtime_eye.py tests\test_runtime_openxr.py tests\test_openxr_runtime.py tests\test_session_helpers.py
src\python3\python.exe -m pytest tests\test_runtime_openxr.py tests\test_openxr_runtime.py tests\test_session_helpers.py -q
```

Result:

```text
33 passed
```

Notes / next improvements:

- Flat OpenXR debug fields (`openxr_ipd`, `openxr_depth_strength`, `openxr_stereo_scale`, `openxr_max_shift_ratio`, etc.) remain published only as compatibility fields and can be removed after downstream/debug consumers stop requiring them.

### 2026-06-28 Hot Reload Runtime Quality, Preset, Provider, And Depth Size Snapshot Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` hot-reload compliance pass by routing runtime quality mode, preset selection, provider-affecting profile settings, and depth-provider size settings through the same `RuntimeSettingsSnapshot` path as the other user-adjustable stereo controls.

Implemented in this follow-up:

- Added hot-reload parsing for `Stereo Quality` / `Synthetic View` into normalized `stereo_quality`.
- Added structured `RuntimeSettingsSnapshot.runtime_quality_mode` application by mapping explicit `Runtime Quality Mode` / `Stereo Runtime Mode` snapshots to `StereoRuntimeConfig.mode`; `Run Mode` remains application-target owned and is not treated as a hot-reload quality mode.
- Added structured `RuntimeSettingsSnapshot.stereo_preset` and hot-reload parsing for `Stereo Preset` / `Stereo Mode Preset`.
- Preserved the fast-quality temporal/postprocess disable behavior while also updating the runtime backend field itself.
- Made fixed preset changes override the stale context `active_preset`, while `auto` keeps the current runtime-selected active preset.
- Made explicit `Temporal` and `Auto Scene Reset` / `Auto Reset Temporal` hot-reload toggles override positive strength/threshold values, matching initialization semantics.
- Added conditional hot-reload rebuild fields for `Depth Model` / `model_id` and depth backend controls (`Depth Backend`, `MIGraphX`, `TensorRT`, `ONNX`) so provider rebuilds are requested only when the selected model/backend differs from the current runtime config.
- Added structured `RuntimeSettingsSnapshot.profile_sync` and hot-reload parsing for `Depth Profile Sync` / `Profile Sync`; this is classified as a runtime-handled provider rebuild field because it feeds `DepthProviderConfig.profile_sync`.
- Wired `Depth Resolution` into runtime initialization by mapping it to `StereoRuntimeConfig.export_width` and deriving `export_height` from the existing 294:518 artifact ratio.
- Passed `StereoRuntimeConfig.export_width` into `DepthProviderConfig.depth_resolution`, so PyTorch/ROCm/MPS/XPU provider paths no longer stay fixed at the default 518 when the GUI selects a different depth resolution.
- Added hot-reload parsing for `Depth Resolution` plus explicit `Export Height` / `Export Width`, emitting `export_height` / `export_width` only when the resolved provider size changes.
- Hardened the legacy fallback path for runtimes without `apply_settings_snapshot()` by filtering snapshot-only fields such as `debug_flags` before `dataclasses.replace()` updates `StereoRuntimeConfig`.
- Added regression coverage for raw value parsing, snapshot construction, OpenXR propagation, runtime quality mode application, fixed preset application, auto preset preservation, explicit temporal toggles, conditional depth-provider rebuild fields, profile-sync provider rebuild, depth-resolution provider sizing, and fallback runtime updates.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\adapter.py src\stereo_runtime\hot_reload.py src\stereo_runtime\settings_snapshot.py src\stereo_runtime\runtime.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_settings_snapshot.py -q
src\python3\python.exe -m pytest tests\test_adapter_config.py tests\test_hot_reload.py tests\test_settings_snapshot.py -q
src\python3\python.exe -m pytest tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py -q
```

Result:

```text
26 passed
47 passed
181 passed
```

### 2026-06-28 Hot Reload Debug And Mask Snapshot Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` hot-reload compliance pass by aligning the remaining initial-settings fields with `RuntimeSettingsSnapshot` hot reload.

Implemented in this follow-up:

- Added structured `RuntimeSettingsSnapshot.debug_output` so `Debug Stereo Output` can update runtime config instead of remaining an initialization-only setting.
- Added `debug_flags={"debug_output": ...}` to the hot-reload snapshot for spec-level debug flag tracing.
- Added hot-reload parsing for `Screen Edge Mask Suppression` into `screen_edge_mask_suppression`.
- Added `debug_output` to `runtime_stereo_overrides()` so the fallback path without `apply_settings_snapshot()` still preserves the setting.
- Updated hot-reload tests covering raw values, snapshot construction, fallback overrides, and OpenXR stereo-control propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\hot_reload.py src\stereo_runtime\settings_snapshot.py tests\test_hot_reload.py
src\python3\python.exe -m pytest tests\test_hot_reload.py -q
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_openxr_state.py -q
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py -q
```

Result:

```text
6 passed
41 passed
150 passed
```

### 2026-06-28 Hot Reload Parallax And Packing Snapshot Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` hot-reload compliance pass by routing normalized parallax budget controls and output packing format from `settings.yaml` hot reload into `RuntimeSettingsSnapshot`.

Implemented in this follow-up:

- Added hot-reload parsing for `Max Disparity Px` / `Max Disparity PX` into `max_disparity_px`.
- Added hot-reload parsing for `Parallax Preset` / `Parallax Budget Preset` into `parallax_preset`.
- Added hot-reload parsing for `Display Mode` into normalized `output_format`, keeping Output Packing Format independent from runtime mode.
- Preserved legacy `Max Shift Ratio` handling while allowing normalized parallax budget and packing fields to reach runtime and OpenXR snapshot consumers.
- Updated hot-reload tests covering raw values, snapshot construction, and OpenXR stereo-control propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\hot_reload.py tests\test_hot_reload.py
src\python3\python.exe -m pytest tests\test_hot_reload.py -q
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_openxr_state.py -q
src\python3\python.exe -m pytest tests\test_hot_reload.py tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_state.py -q
```

Result:

```text
150 passed
```

### 2026-06-28 Pipeline Rebuild Snapshot Guard Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` hot-reload compliance pass by preventing pipeline-owned rebuild fields from being silently merged into `StereoRuntime` active settings when the current pipeline/context has not been rebuilt.

Implemented in this follow-up:

- Added `RuntimeSettingsPipelineRebuildRequired` for settings that require pipeline/context rebuild rather than in-place runtime apply.
- Updated `StereoRuntime.apply_settings_snapshot()` to reject non-depth-provider `PIPELINE_REBUILD` fields such as `render_size_policy`, `stereo_render_scale`, `stereo_synthesis_mode`, and `output_transport`.
- Kept depth-provider rebuild fields handled inside `StereoRuntime`.
- Updated `RuntimePipelineLoop` so pipeline rebuild requests propagate to the lifecycle layer instead of being logged and swallowed as ordinary runtime errors.
- Added regression tests for rejecting stale runtime merges and propagating pipeline rebuild exceptions.
- Updated the engineering design spec with the runtime-vs-pipeline rebuild ownership rule.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\settings_snapshot.py src\stereo_runtime\runtime.py src\stereo_runtime\pipeline.py src\stereo_runtime\__init__.py tests\test_settings_snapshot.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_settings_snapshot.py tests\test_runtime_pipeline.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
136 passed
```

### 2026-06-28 Runtime Settings Temporal Reset Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` hot-reload compliance pass by making temporal-affecting runtime settings clear temporal history instead of only updating runtime config values.

Implemented in this follow-up:

- Added a runtime hot-reload temporal-reset field set for settings that change temporal enablement or the active parallax/depth-response distribution.
- Reset `StereoRuntime.temporal_state` and the OpenXR RGB+depth direct `_openxr_depth_temporal` cache when those fields change.
- Added pending `debug_info["temporal_reset_reason"] = "settings_changed"` on the next runtime result, then consume it so later frames do not repeat the reason.
- Added regression coverage for temporal setting changes resetting both stereo and OpenXR depth temporal history.
- Updated the engineering debug contract to include `settings_changed` as a valid `temporal_reset_reason`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests\test_settings_snapshot.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
134 passed
```

### 2026-06-28 Scene Reset Reason Debug Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` temporal-state compliance pass by aligning scene-cut temporal resets with the same structured reset-reason debug field used by pipeline-level render-size and source-target resets.

Implemented in this follow-up:

- Added `debug_info["temporal_reset_reason"] = "scene_reset"` when synthesis auto scene-cut detection resets stereo temporal history.
- Kept existing debug-only `temporal_reset`, `scene_delta`, and `temporal_reset_count` fields for detailed diagnostics.
- Updated the scene-cut regression test to verify the structured reset reason.
- Updated the engineering debug contract to include `scene_reset` as a valid `temporal_reset_reason`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\synthesis.py tests\test_synthesis.py
src\python3\python.exe -m pytest tests\test_synthesis.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
133 passed
```

### 2026-06-28 Source Target Temporal Reset Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` temporal-state compliance pass by making capture source/target changes explicitly reset stereo temporal history even when the resolved render size stays the same.

Implemented in this follow-up:

- Added source-target tracking to `RuntimePipelineLoop` using `CapturedFrame` source metadata (`capture_mode`, `monitor_index`, `window_title`, and stable source metadata keys).
- Reset `stereo_runtime.temporal_state.reset_stereo()` when the source target changes between frames.
- Added `debug_info["temporal_reset_reason"] = "source_target_changed"` on the affected runtime result; if multiple reset causes happen on the same frame, the field records comma-separated reasons.
- Added a two-frame pipeline regression test covering same-size monitor target switching.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\pipeline.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime_pipeline.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
133 passed
```

### 2026-06-28 Render Size Temporal Reset Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` temporal-state compliance pass by making render-size changes explicitly reset stereo temporal history at the pipeline boundary.

Implemented in this follow-up:

- Added previous-render-size tracking to `RuntimePipelineLoop`.
- Reset `stereo_runtime.temporal_state.reset_stereo()` when the resolved `render_size` changes between frames.
- Added `debug_info["temporal_reset_reason"] = "render_size_changed"` on the affected runtime result.
- Added a two-frame pipeline regression test covering the reset and debug metadata.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\pipeline.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime_pipeline.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
132 passed
```

### 2026-06-27 Depth Size Contract Debug Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` depth/render-size compliance pass by making provider-internal depth size and actual render-aligned depth size visible on runtime results.

Implemented in this follow-up:

- Added `depth_render_size` to RGB and OpenXR runtime `debug_info` from the actual depth tensor shape.
- Added `depth_provider_size` to runtime `debug_info` from provider metadata, preferring explicit provider size fields and falling back to `depth_resolution` when that is the only stable provider contract.
- Reused one `provider_report()` per frame for both result `provider_info` and depth-size debug metadata.
- Updated runtime and OpenXR tests to verify provider size and render-aligned depth size.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py tests\test_runtime.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_openxr.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
131 passed
```

### 2026-06-27 Host Target And Transport Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` runtime-result debug compliance pass by moving application target and transport labeling to the host/pipeline boundary, where the resolved GUI run mode is known.

Implemented in this follow-up:

- Added `application_runtime_target` and `output_transport` to `AppRuntimeContext` and `RuntimePipelineContext`.
- Mapped GUI run modes to spec contracts: OpenXR Link -> `openxr/openxr_swapchain`, MJPEG/RTMP streamer -> `network_stream/encoded_stream`, 3D Monitor -> `local_display/local_fullscreen`, Local Viewer -> `local_display/local_window`.
- Propagated these host-resolved fields into each runtime result `debug_info` after pipeline processing.
- Updated runtime context and pipeline tests to cover the mapping and per-frame debug metadata.

Verification:

```powershell
src\python3\python.exe -m py_compile src\app_runtime\runtime_context.py src\stereo_runtime\pipeline.py tests\test_runtime_context.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime_context.py tests\test_runtime_pipeline.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_context.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
131 passed
```

### 2026-06-27 Runtime And Pipeline Contract Defaults Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` runtime-result debug compliance pass by filling contract fields that were previously present only when an active `RuntimeSettingsSnapshot` supplied them.

Implemented in this follow-up:

- Added runtime-config default debug fields for `runtime_quality_mode`, `output_format`, `max_disparity_px`, and `parallax_preset` in both RGB and OpenXR runtime result paths. Active snapshot values still take precedence.
- Added pipeline-owned `render_size_policy` and `stereo_render_scale` debug fields when `RuntimePipelineLoop` has a `RenderSizeConfig`.
- Updated runtime and pipeline tests to cover these defaults.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\stereo_runtime\pipeline.py tests\test_runtime.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_pipeline.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
125 passed
```

### 2026-06-27 Runtime Contract Scalar Debug Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` runtime/debug compliance pass by making the hot-reload and synthesis-control scalar fields stable in normal runtime debug metadata, not only in active-setting snapshots or quality-branch debug output.

Implemented in this follow-up:

- Added common `synthesize_stereo()` debug scalars for `convergence`, `temporal_enabled`, `temporal_strength`, `hole_fill_mode`, `hole_fill_radius`, `hole_fill_strength`, `edge_threshold`, `edge_dilation`, and `mask_feather_radius`.
- Kept these fields scalar so they survive the existing non-`debug_output` pruning path.
- Added a regression test proving the fast backend still exposes the spec contract scalars when tensor-heavy debug output is disabled.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\synthesis.py tests\test_synthesis.py
src\python3\python.exe -m pytest tests\test_synthesis.py tests\test_runtime_openxr.py -q
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py tests\test_synthesis.py -q
```

Result:

```text
125 passed
```

### 2026-06-27 Packing And Transport Debug Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` runtime-result debug compliance pass by exposing packing and transport as separate debug concepts. This keeps the spec-layer distinction explicit: packing is owned by stereo runtime output, while transport is owned by the host/pipeline path.

Implemented in this follow-up:

- Added `packing_format` to RGB and OpenXR runtime debug metadata.
- Added pipeline-level `transport` debug metadata: `openxr_swapchain` for OpenXR runtime mode and `local_window` for the current viewer pipeline path.
- Added tests covering local render-size pipeline transport, OpenXR pipeline transport, and runtime packing metadata.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\stereo_runtime\pipeline.py tests\test_runtime_openxr.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py -q
```

Result:

```text
67 passed
```

### 2026-06-27 Depth Response Debug Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` parallax/debug compliance pass by making the active depth-response curve explicit in parallax debug metadata. Runtime results already exposed `resolved_max_disparity_px` and `parallax_budget_preset`; this follow-up adds the missing `depth_response` contract label.

Implemented in this follow-up:

- Added `depth_response_name="linear_clamp_convergence_v1"` to `ParallaxBudget`.
- Added `depth_response` to `parallax_debug_info()`, which flows into synthesis and OpenXR render/debug paths.
- Added tests covering the debug contract directly and through OpenXR runtime result debug info.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\parallax.py tests\test_parallax.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py tests\test_parallax.py -q
```

Result:

```text
67 passed
```

### 2026-06-27 OpenXR Direct Shader Uniform Bundle Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` OpenXR direct-path compliance pass by bundling adapter/runtime-generated legacy shader uniforms before the viewer consumes them. This keeps existing legacy scalar debug fields for compatibility, but gives the viewer a single structured source for RGB+depth direct shader parameters.

Implemented in this follow-up:

- Added `debug_info["openxr_legacy_shader_uniforms"]` from `OpenXRRenderConfig` in `StereoRuntime.process_openxr_frame()`.
- Updated `CoreRuntimeEyeMixin._apply_runtime_rgb_depth_config()` to prefer the bundled shader uniform dict, with existing scattered `openxr_*` fields retained as fallback.
- Added tests that verify runtime emits the bundle and the viewer prefers it even when legacy scattered fields contain stale values.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\xr_viewer\core_runtime_eye.py tests\test_runtime_openxr.py tests\test_openxr_runtime.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py -q
```

Result:

```text
61 passed
```

### 2026-06-27 OpenXR Result Conversion Struct Field Follow-up

Continued the runtime-output metadata cleanup by removing a remaining production dependency on `debug_info["runtime_output_format"]` inside `openxr_result_from_stereo_result()`. The conversion now uses `StereoRuntimeResult.output_format` first when deciding whether a fast-plus half-SBS result should be split into full OpenXR eye textures.

Implemented in this follow-up:

- Updated `openxr_result_from_stereo_result()` to prefer structured `output_format`, with legacy debug metadata only as a fallback.
- Updated `tests/test_runtime_openxr.py` so the split-half-SBS conversion still works when legacy `debug_info["runtime_output_format"]` is stale or wrong but `output_format` is correct.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py -q
```

Result:

```text
60 passed
```

### 2026-06-27 Pipeline Render Size Debug Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` compliance pass by making the pipeline-resolved coordinate system visible on each runtime result. `RuntimePipelineLoop` already resolves the actual render size before preprocessing; this follow-up records that boundary in result debug metadata so capture size and synthesis/render size are no longer inferred indirectly from tensor shapes.

Implemented in this follow-up:

- Added pipeline-level `capture_size` and `render_size` debug fields after runtime processing and before queueing the result.
- Kept the logic outside `StereoRuntime` because capture/session sizing remains a host/pipeline responsibility; `StereoRuntime` still receives the already prepared render RGB.
- Updated `tests/test_runtime_pipeline.py` to verify a 4K capture with scaled render policy records `capture_size=3840x2160` and `render_size=1920x1080`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\pipeline.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py -q
```

Result:

```text
60 passed
```

### 2026-06-27 Active Settings Debug Contract Follow-up

Continued the `docs/25-2d-to-3d-runtime-specification.md` compliance pass by expanding the runtime result debug surface for active `RuntimeSettingsSnapshot` fields. This keeps the snapshot application path unchanged, but makes it easier to distinguish whether GUI/API settings reached runtime and which normalized-depth controls are active.

Implemented in this follow-up:

- Expanded active settings debug propagation to include `presentation_flags`, `debug_flags`, `output_format`, `max_disparity_px`, `parallax_preset`, `convergence`, and `hole_fill_mode`.
- Preserved the existing session-restart boundary for `application_runtime_target` and capture source/target fields; this follow-up does not pretend those fields are hot-applied.
- Updated `tests/test_settings_snapshot.py` to verify merged active snapshot fields survive a later hot reload and appear in `OpenXRRuntimeResult.debug_info`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_settings_snapshot.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py -q
```

Result:

```text
60 passed
```

### 2026-06-27 Runtime Output Metadata Struct Fields Follow-up

Continued the render-size/runtime-output compliance pass by promoting host-consumed runtime output metadata from debug-only keys into structured runtime result fields. Local Viewer and OpenXR host startup now consume `output_display_size` directly, with legacy `debug_info["runtime_output_display_size"]` retained only as a compatibility fallback.

Implemented in this follow-up:

- Added structured output contract fields to `StereoRuntimeResult` and `OpenXRRuntimeResult`: `output_eye_size`, `output_display_size`, `output_format`, `output_dtype`, and `output_pack_backend`.
- Kept legacy debug keys such as `runtime_output_eye_size` and `runtime_output_display_size` populated for existing logs/tests/compatibility.
- Updated Local Viewer and OpenXR startup helpers to prefer `runtime_result.output_display_size`, then fall back to legacy debug metadata, then fall back to tensor shape inference.
- Updated FPS breakdown runtime stats to prefer structured output fields before legacy debug metadata.
- Added `frame_size_from_runtime_result()` in `src/viewer/viewer_runtime.py`.
- Preserved the existing local viewer display policy that constrains non-stream local windows to 1280 px width while retaining the runtime output aspect ratio.
- Added tests covering structured runtime result fields, structured viewer/OpenXR consumption, and legacy debug fallback behavior.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\runtime.py src\stereo_runtime\session_helpers.py src\viewer\viewer.py src\viewer\viewer_runtime.py src\xr_viewer\core_runtime_eye.py src\xr_viewer\openxr_runtime.py src\utils\breakdown.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_session_helpers.py tests\test_breakdown.py
src\python3\python.exe -m pytest tests\test_runtime.py tests\test_runtime_openxr.py tests\test_viewer_runtime.py tests\test_openxr_runtime.py tests\test_runtime_pipeline.py tests\test_session_helpers.py tests\test_breakdown.py -q
```

Result:

```text
50 passed
```

Notes / next improvements:

- `debug_info` still carries the old runtime output keys for compatibility, but new host code should consume the structured fields.
- The network-only legacy stream path still consumes the packed SBS tensor directly, which matches its transport contract and does not own a display window size.

### 2026-06-27 GUI Hot Reload Snapshot Compliance Follow-up

Continued the 2D-to-3D runtime specification compliance pass by closing the gap where GUI hot-save changes were persisted to `settings.yaml` and then applied through the legacy loose hot-reload path instead of the unified runtime settings snapshot contract.

Implemented in this follow-up:

- Added `auto_reset_temporal` and `scene_reset_threshold` to `RuntimeSettingsSnapshot` hot-reload classification and runtime config update mapping.
- Added `hot_reload_runtime_settings_snapshot()` so YAML-driven GUI hot-save updates are converted into a `RuntimeSettingsSnapshot` with source metadata.
- Updated `StereoHotReloader.apply_if_needed()` to apply settings through `StereoRuntime.apply_settings_snapshot()` when available, while preserving the legacy direct config replacement fallback for compatibility tests and simple callers.
- Updated OpenXR hot-reload propagation to send the same snapshot through `update_openxr_runtime_config(snapshot=...)` instead of separate loose stereo parameter arguments.
- Added tests covering hot-reload snapshot construction and the snapshot-based OpenXR/runtime propagation path.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\settings_snapshot.py src\stereo_runtime\hot_reload.py tests\test_hot_reload.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests/test_hot_reload.py tests/test_settings_snapshot.py tests/test_runtime_pipeline.py -q
```

Result:

```text
23 passed
```

Notes / next improvements:

- GUI is still a separate parent process from `src/main.py`; this follow-up routes the current persisted hot-save handoff through the snapshot contract inside the runtime process, rather than adding a new inter-process settings channel.
- A future IPC-backed GUI live settings channel can call `RuntimeCallbacks.send_settings_snapshot()` directly if the host/runtime process model is changed.

### 2026-06-27 2D-to-3D Runtime Specification Compliance Pass

A local compliance pass was completed against `docs/25-2d-to-3d-runtime-specification.md`. The focus was to align current code, tests, and the runtime handoff/debug surface with the spec without removing explicit legacy compatibility paths.

Implemented in this pass:

- Updated `docs/25-2d-to-3d-runtime-specification.md` so `scaled` render-size policy matches the current implementation: only 4K-class inputs are mapped to stable downsample tiers, while sub-4K inputs keep `capture_size`.
- Updated `tests/test_runtime_pipeline.py` so pipeline render-size coverage validates the 4K tier path instead of the previous continuous-scale assumption for 1080p input.
- Added runtime debug fields for `hot_reload_class` and `hot_reload_changed_fields` in both RGB and OpenXR runtime result paths.
- Expanded `RuntimeSettingsSnapshot` to carry spec-layer fields such as `application_runtime_target`, `runtime_quality_mode`, `stereo_synthesis_mode`, `render_size_policy`, `stereo_render_scale`, `output_transport`, capture fields, presentation flags, and debug flags.
- Split snapshot field classification from fields that are allowed to update `StereoRuntimeConfig`, preventing spec-layer fields from being passed into `replace(self.config, **updates)`.
- Added active settings snapshot merge/retention in `StereoRuntime`, and surfaced active high-level settings into runtime debug info when present.
- Narrowed depth-provider rebuilds so only depth-provider-relevant fields (`depth_backend`, `model_id`, `export_height`, `export_width`) recreate the provider; render-size and synthesis-mode pipeline rebuild fields no longer replace the depth provider.
- Changed host-facing `StereoRuntimeConfig.parallax_preset` default from `legacy` to `standard`, so the default normalized-depth runtime path now uses the explicit parallax budget model.
- Preserved explicit `parallax_preset="legacy"` compatibility and added settings mapping for `Parallax Preset` / `Parallax Budget Preset` plus `Max Disparity Px` / `Max Disparity PX`.
- Added tests covering spec-layer snapshot classification, non-config spec fields, active settings debug visibility, default standard parallax behavior, explicit legacy compatibility, and parallax settings mapping.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\adapter.py src\stereo_runtime\runtime.py src\stereo_runtime\settings_snapshot.py tests\test_adapter_config.py tests\test_settings_snapshot.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests/test_adapter_config.py tests/test_presets.py tests/test_synthesis.py tests/test_runtime.py tests/test_parallax.py tests/test_render_size.py tests/test_settings_snapshot.py tests/test_runtime_pipeline.py tests/test_runtime_openxr.py tests/test_openxr_state.py
```

Result:

```text
146 passed
```

Notes / next improvements:

- Low-level `StereoConfig`, `ShiftParams`, and `OpenXRRenderConfig` defaults now select `standard`; explicit `parallax_preset="legacy"` remains available only as a compatibility path.
- OpenXR direct path now exposes `OpenXRRuntimeResult.legacy_shader_uniforms` as the primary direct-path shader-uniform contract; flat `openxr_ipd`, `openxr_depth_strength`, `openxr_stereo_scale`, and `openxr_max_shift_ratio` debug keys remain as compatibility fields alongside normalized parallax debug fields.
- GUI hot-save still needs a follow-up to emit full `RuntimeSettingsSnapshot` objects for live runtime settings instead of relying only on persisted settings and legacy callbacks.
- `src/settings.yaml` already had unrelated local edits in the working tree and was not intentionally modified as part of this compliance pass.

### 2026-06-27 Streaming Encoder Profile Runtime Wiring

The MJPEG streamer dry-run has been formally applied and connected through legacy stream mode and viewer MJPEG runtime mode. This step only changes profile wiring plus encoding-side resize/pixel-format handling; it does not move transport policy into `stereo_runtime`.

Implemented:

- Added `EncoderProfile` as the transport profile contract for streaming codecs, quality, target FPS, optional resize, bitrate, and pixel format.
- Updated `MJPEGStreamer` to accept `EncoderProfile` while preserving legacy `fps` / `quality` constructor arguments.
- Moved MJPEG resize and RGB/BGR/BGRA conversion into the streamer immediately before JPEG encoding.
- Updated legacy stream runtime to pass an encoder profile and keep `runtime_result.sbs` as the packed SBS source.
- Updated viewer MJPEG runtime to pass the same encoder profile contract to `MJPEGStreamer`.
- Updated mode config builders so viewer and legacy runtime configs derive encoder profiles from existing stream FPS/quality settings.
- Added tests for mode config profile generation plus legacy/viewer profile propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\streaming\encoder_profile.py src\streaming\mjpeg_streamer.py src\streaming\legacy_runtime.py src\viewer\viewer_runtime.py src\app_runtime\mode_configs.py tests\test_legacy_runtime.py tests\test_viewer_runtime.py tests\test_mode_configs.py
src\python3\python.exe -m pytest tests\test_legacy_runtime.py tests\test_viewer_runtime.py tests\test_mode_configs.py -q
```

Result:

```text
7 passed
```

Notes:

- `docs/27-vr-headset-focal-distance-reference.md` has unrelated local edits and was intentionally left out of this task's code/doc changes.

Commit title:

```text
refactor: add streaming encoder profile runtime wiring
```

### 2026-06-27 CUDA/ROCm Capture Copy Metadata - Phase 1

Task 7 has started by making Windows event capture copy semantics explicit and propagating capture/preprocess device metadata into runtime debug output. This phase labels the current implementation; it does not claim true zero-copy for the Windows event path.

Implemented in this phase:

- Added explicit `frame_raw_device` override support to `capture_frame_from_raw()`.
- Updated Windows event capture so `WindowsCaptureCUDA` and `WindowsCaptureROCm` prefer `clone()` and label `FrameCopyMode.CLONE`; CPU WindowsCapture labels `FrameCopyMode.COPY`.
- Added backend metadata `zero_copy: False` for Windows event captures to make the current copy/clone behavior explicit.
- Extended `capture_frame_to_rgb()` tensor path to accept capture metadata overrides and attach `_d2s_capture_copy_mode` / `_d2s_capture_zero_copy` to preprocessed tensors.
- Preserved capture metadata through `prepare_rgb_for_stereo_runtime()` metadata copying.
- Propagated `capture_copy_mode`, `capture_zero_copy`, `capture_frame_raw_device`, and preprocess device fields into runtime result `debug_info`.
- Added targeted tests for CUDA/ROCm event-capture labeling, preprocess metadata overrides, and runtime pipeline debug propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\capture\types.py src\capture\backends\windows_capture_event.py src\capture\preprocess.py src\stereo_runtime\pipeline.py tests\test_windows_capture_event.py tests\test_capture_preprocess.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_windows_capture_event.py tests\test_capture_preprocess.py tests\test_runtime_pipeline.py tests\test_capture_metadata.py -q
```

Result:

```text
23 passed
```

Notes / next improvements:

- Real CUDA/ROCm hardware validation is still needed to confirm whether backend buffers are GPU tensors and whether any hidden CPU staging occurs inside third-party capture packages.
- Additional capture backends such as Desktop Duplication / DXCamera can be labeled in a follow-up if they enter the active runtime path.

### 2026-06-26 Render Size Runtime Policy - Phase 3

Task 6 now exposes render-size policy through GUI/settings and persists the user-facing policy fields into the runtime context path.

Implemented in this phase:

- Added GUI defaults and `settings.yaml` defaults for `Render Size Policy`, `Render Scale`, fixed render size, dynamic pixel cap, minimum dimension, and alignment.
- Added Advanced Device Options controls for render policy, render scale, fixed size, pixel cap, minimum side, and output alignment.
- Wired GUI config load/save so the controls persist canonical `Render ...` settings keys.
- Added EN/CN labels and tooltips for the render-size policy controls.
- Wired viewer/runtime settings resolution so GUI settings produce `RenderSizeConfig` and reach `create_runtime_context()`.
- Added targeted tests for render-size settings parsing, viewer settings propagation, and GUI persistence/static wiring.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\render_size.py src\viewer\settings.py src\utils\runtime_exports.py src\utils\__init__.py src\main.py src\app_runtime\runtime_context.py src\gui\config.py src\gui\builders.py src\gui\handlers.py src\gui\config_mgr.py src\gui\localization.py tests\test_render_size.py tests\test_viewer_settings.py tests\test_gui_config.py
src\python3\python.exe -m pytest tests\test_render_size.py tests\test_viewer_settings.py tests\test_runtime_context.py tests\test_runtime_pipeline.py tests\test_gui_config.py -q
```

Result:

```text
61 passed
```

Notes / next improvements:

- Host/window ownership still needs migration to consume runtime output size debug fields consistently.
- Render-size policy controls are currently under Advanced Device Options; live hot-update through `RuntimeSettingsSnapshot` can be added in a later GUI hot-save pass.

### 2026-06-26 Render Size Runtime Policy - Phase 2

Task 6 now routes pipeline preprocess sizing through the runtime render-size policy while keeping the policy opt-in for compatibility.

Implemented in this phase:

- Added optional `render_size_config` to `RuntimePipelineContext`.
- Added `render_size_config` to `AppRuntimeContext` with a default native `RenderSizeConfig()`.
- Wired `build_runtime_pipeline_context()` to pass the app render-size policy into the runtime pipeline.
- Routed tuple/list capture sizes through `resolve_render_size()` before calling `capture_frame_to_rgb()`.
- Preserved legacy scalar `target_height` and unset-config paths by passing them through unchanged.
- Added pipeline and app-runtime bridge coverage for render-size policy propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\pipeline.py src\stereo_runtime\render_size.py tests\test_runtime_pipeline.py
src\python3\python.exe -m py_compile src\app_runtime\runtime_context.py tests\test_runtime_context.py
src\python3\python.exe -m pytest tests\test_runtime_context.py tests\test_render_size.py tests\test_runtime_pipeline.py tests\test_runtime_openxr.py -q
```

Result:

```text
32 passed
```

Notes / next improvements:

- GUI settings still need to expose and persist user-facing render-size policy values; the app context currently uses the default native policy.
- Host/window ownership still needs migration to consume runtime output size debug fields consistently.

### 2026-06-26 Render Size Runtime Policy - Phase 1

Task 6 from `prompts/codex-refactor-prompt.md` has started with a standalone runtime policy module and normalized runtime output size debug fields. This phase does not change capture/window sizing behavior yet.

Implemented in this phase:

- Added `src/stereo_runtime/render_size.py` with `RenderSizePolicy`, `RenderSizeConfig`, `resolve_render_size()`, and `runtime_output_size_text()`.
- Implemented `native`, `scaled`, `fixed`, and `dynamic` render-size policy resolution with alignment handling.
- Exported render-size policy helpers through `stereo_runtime` lazy public API.
- Updated `StereoRuntime.process_rgb_frame()` and `process_openxr_frame()` debug info to consistently report `runtime_output_eye_size` and `runtime_output_display_size`.
- Added tests for render-size policy resolution and runtime output size debug fields.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\render_size.py src\stereo_runtime\runtime.py src\stereo_runtime\__init__.py tests\test_render_size.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_render_size.py tests\test_runtime_openxr.py tests\test_openxr_state.py tests\test_openxr_render.py -q
```

Result:

```text
36 passed
```

Notes / next improvements:

- `RuntimePipelineContext` now has an opt-in `RenderSizeConfig` path in Phase 2; App/GUI settings still need to wire user-facing policy values into it.
- OpenXR/viewer consumers can now prefer `runtime_output_display_size` and `runtime_output_eye_size`, but host code has not yet been fully migrated to render-size policy ownership.

### 2026-06-26 OpenXR Adapter Snapshot Uniforms - Phase 1

Task 5 from `prompts/codex-refactor-prompt.md` has started with a compatible adapter-level conversion path. Legacy OpenXR viewer callbacks remain accepted, but render config creation now goes through a snapshot-style adapter.

Implemented in this phase:

- Added `openxr_render_config_from_snapshot()` in `src/stereo_runtime/adapter.py` to convert `RuntimeSettingsSnapshot` normalized fields into `OpenXRRenderConfig`.
- Updated `OpenXRStateController` to store runtime settings as a `RuntimeSettingsSnapshot` plus separate legacy `ipd` and `screen_roll` overrides.
- Preserved legacy viewer callback behavior where `ipd` updates the OpenXR uniform without rewriting runtime `ipd_mm`.
- Added snapshot update support to `RuntimeCallbacks.update_openxr_runtime_config()`.
- Added OpenXR runtime debug propagation for `resolved_max_disparity_px`, `parallax_budget_preset`, `parallax_resolver_version`, and adapter-origin `openxr_max_disparity_px`.
- Added tests for adapter conversion, snapshot-driven OpenXR state updates, legacy fallback behavior, and OpenXR resolved disparity debug fields.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\adapter.py src\stereo_runtime\openxr_state.py src\app_runtime\runtime_callbacks.py src\stereo_runtime\runtime.py tests\test_openxr_state.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_openxr_state.py tests\test_runtime_openxr.py tests\test_openxr_render.py tests\test_settings_snapshot.py -q
```

Result:

```text
35 passed
```

Notes / next improvements:

- `process_openxr_frame()` now returns `legacy_shader_uniforms` on `OpenXRRuntimeResult`; legacy `openxr_ipd/openxr_depth_strength/openxr_stereo_scale/openxr_max_shift_ratio` debug fields remain published only for compatibility alongside normalized resolved disparity debug fields.
- `StereoHotReloader` still calls the legacy OpenXR config callback with loose params; this remains compatible but can be routed through snapshots in a follow-up.

### 2026-06-26 Runtime Preprocess Device Dispatch - Phase 1

Task 4 from `prompts/codex-refactor-prompt.md` has started. The immediate compatibility break in `tests/test_capture_preprocess.py` is fixed by restoring `capture_frame_to_rgb(..., target_height=...)` support while keeping the newer positional `target_resolution` call path.

Implemented in this phase:

- Added explicit target-resolution normalization for `target_resolution`, `target_height`, and `size` arguments.
- Added tensor preprocess validation for HWC BGR/BGRA inputs.
- Standardized tensor preprocess and runtime preparation outputs to BCHW RGB float tensors while preserving CHW input compatibility in `prepare_rgb_for_stereo_runtime()`.
- Added explicit numpy/torch input kind tracking, origin/output device tracking, and `preprocess_device_transfer` metadata on tensor outputs.
- Propagated preprocess metadata into `StereoRuntimeResult.debug_info` and `OpenXRRuntimeResult.debug_info`.
- Kept numpy output behavior unchanged for legacy callers.
- Added tests for CPU numpy -> tensor preprocessing, CPU torch tensor preprocessing, invalid target argument combinations, and runtime debug metadata propagation.

Verification:

```powershell
src\python3\python.exe -m py_compile src\capture\preprocess.py src\stereo_runtime\runtime.py tests\test_capture_preprocess.py tests\test_runtime_openxr.py
src\python3\python.exe -m pytest tests\test_capture_preprocess.py tests\test_runtime_openxr.py tests\test_runtime_pipeline.py tests\test_capture_metadata.py -q
```

Result:

```text
30 passed
```

Notes / next improvements:

- GPU CUDA/ROCm paths are now structurally explicit through device-origin/device-output metadata, but still need hardware-backed validation.
- `CapturedFrame` metadata such as `frame_raw_device` and `copy_mode` is not yet passed directly into preprocess; this should be addressed with task 7 or a later phase of task 4.

### 2026-06-26 CaptureFrame Metadata Contract

Task 3 from `prompts/codex-refactor-prompt.md` is implemented as a compatible upgrade from raw queue triples to `CapturedFrame` metadata objects.

Implemented:

- Added `FrameCopyMode` and expanded `CapturedFrame` with capture source, size, raw type/device/dtype, copy mode, original format, and free-form metadata fields.
- Added `capture_frame_from_raw()` and `ensure_captured_frame()` helpers so producers can create metadata frames while legacy `(frame_raw, size, timestamp)` tuples remain accepted.
- Updated `PollingCaptureRunner` and `WindowsCaptureEventRunner` to emit `CapturedFrame` objects through `on_frame`.
- Added Windows event backend copy-mode tracking for `copy()` vs `clone()` buffers.
- Updated `CaptureSessionLoop` to enqueue `CapturedFrame` while still accepting legacy three-argument frame callbacks.
- Updated `RuntimePipelineLoop` to unpack either `CapturedFrame` or legacy tuples from `raw_q`.
- Exported `FrameCopyMode` from `capture` public API.

Verification:

```powershell
src\python3\python.exe -m py_compile src\capture\types.py src\capture\runners.py src\capture\session.py src\capture\backends\windows_capture_event.py src\capture\__init__.py src\stereo_runtime\pipeline.py tests\test_capture_metadata.py tests\test_capture_session.py tests\test_windows_capture_event.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_capture_metadata.py tests\test_capture_session.py tests\test_windows_capture_event.py tests\test_runtime_pipeline.py -q
```

Result:

```text
12 passed
```

Related regression note:

- `tests/test_capture_public_api.py tests/test_capture_factory.py tests/test_monitor_mapping.py` passed when run with adjacent capture tests.
- `tests/test_capture_preprocess.py` currently fails three pre-existing assertions because the tests call `capture_frame_to_rgb(..., target_height=...)` while the implementation does not accept that keyword. This was not introduced by the metadata contract change and should be handled as a separate compatibility cleanup.

Commit title:

```text
refactor: add captured frame metadata contract
```

### 2026-06-26 Parallax Budget Resolver

Task 2 from `prompts/codex-refactor-prompt.md` is implemented with `docs/25-2d-to-3d-runtime-specification.md` taking precedence where it is stricter than the prompt.

Implemented:

- Added `src/stereo_runtime/parallax.py` with `resolve_parallax_budget()`, `ParallaxBudget`, `PARALLAX_BUDGET_TABLE`, and resolver debug metadata.
- Implemented the `docs/25` short-side budget table for `comfort / standard / strong / extreme`, with interpolation and `aspect > 2.0` ultrawide protection.
- Preserved legacy `IPD * stereo_scale * depth_strength * max_shift_ratio` behavior behind the `legacy` preset for compatibility.
- Updated `compute_shift_px()` so normalized parallax budgets are treated as total left/right disparity and each eye receives half of that budget.
- Added `max_disparity_px` and `parallax_preset` to `StereoConfig`, `OpenXRRenderConfig`, `StereoRuntimeConfig`, and `RuntimeSettingsSnapshot`.
- Added `resolved_max_disparity_px`, `parallax_budget_preset`, and `parallax_resolver_version` to synthesis/OpenXR debug info.
- Exposed parallax budget presets through `src/stereo_runtime/presets.py` and public lazy exports.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\parallax.py src\stereo_runtime\baseline_shift.py src\stereo_runtime\synthesis.py src\stereo_runtime\openxr_render.py src\stereo_runtime\adapter.py src\stereo_runtime\presets.py src\stereo_runtime\settings_snapshot.py tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_render.py tests\test_presets.py tests\test_settings_snapshot.py
src\python3\python.exe -m pytest tests\test_parallax.py tests\test_synthesis.py tests\test_openxr_render.py tests\test_presets.py tests\test_settings_snapshot.py -q
src\python3\python.exe -m pytest tests\test_runtime_openxr.py tests\test_adapter_config.py tests\test_hot_reload.py tests\test_runtime_pipeline.py -q
```

Result:

```text
88 passed
40 passed
```

Process improvement applied:

- `docs/25-2d-to-3d-runtime-specification.md` is the canonical Parallax Budget spec. `codex-refactor-prompt.md` still contains an older width-percentage resolver shape, so future prompt-driven work should explicitly prefer `docs/25` when the two differ.

Commit title:

```text
refactor: add parallax budget resolver
```

### 2026-06-26 RuntimeSettingsSnapshot Queue - Phase 1

Task 1 from `prompts/codex-refactor-prompt.md` has started against `docs/26-desktop2stereo-engineering-design-specification.md`.

Implemented in this phase:

- Added `src/stereo_runtime/settings_snapshot.py` with `RuntimeSettingsSnapshot`, `SnapshotChangeClass`, and `RuntimeSettingsRestartRequired`.
- Added `settings_update_q` next to `raw_q` and `runtime_q` in `AppRuntimeContext`.
- Added `RuntimeCallbacks.send_settings_snapshot()` for future GUI/host producers.
- Added `RuntimePipelineLoop` handling for latest-only settings snapshots before processing each frame.
- Added `StereoRuntime.apply_settings_snapshot()` for hot reload and depth-provider rebuild changes; session-restart snapshots raise `RuntimeSettingsRestartRequired` for the outer host layer.
- Added `active_settings_version` to stereo and OpenXR runtime `debug_info`.
- Added targeted tests in `tests/test_settings_snapshot.py` and `tests/test_runtime_pipeline.py`.

Verification:

```powershell
src\python3\python.exe -m py_compile src\stereo_runtime\settings_snapshot.py src\stereo_runtime\runtime.py src\stereo_runtime\pipeline.py src\app_runtime\runtime_context.py src\app_runtime\runtime_callbacks.py tests\test_settings_snapshot.py tests\test_runtime_pipeline.py
src\python3\python.exe -m pytest tests\test_settings_snapshot.py tests\test_runtime_pipeline.py -q
```

Result:

```text
10 passed
```

Notes / next improvements:

- GUI still writes `settings.yaml`; this phase only adds the queue-backed runtime path and callback entry point. A follow-up should convert GUI hot-save values into `RuntimeSettingsSnapshot` objects and send them through a live host channel instead of relying only on YAML mtime polling.
- OpenXR state updates still use the existing legacy callback path. Task 5 should move OpenXR uniform conversion into adapter-level snapshot handling.
- Pipeline rebuild currently recreates the depth provider from the updated runtime config. If a real provider rebuild is expensive on target hardware, add structured telemetry around rebuild duration and provider fallback reason before enabling frequent depth-backend updates.

Commit title:

```text
refactor: add runtime settings snapshot queue
```
