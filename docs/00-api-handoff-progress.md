# Handoff - 2026-06-26

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
refactor: add captured frame metadata contract
```

Canonical specs for current work:

- `docs/26-desktop2stereo-engineering-design-specification.md`
- `docs/25-2d-to-3d-runtime-specification.md`
- `prompts/codex-refactor-prompt.md`
- This file: `docs/00-api-handoff-progress.md`

## Current Boundaries

- Treat `docs/25-2d-to-3d-runtime-specification.md` as canonical when Parallax Budget details differ from the prompt.
- Keep `stereo_runtime` responsible for depth inference, stereo synthesis, OpenXR render-core config, output tensors, timings, and provider/debug contracts.
- Keep capture/session/window lifecycle, GUI settings persistence, OpenXR session/swapchain timing, and final display/submit outside `stereo_runtime`.
- Keep compatibility paths where recent tasks introduced new contracts: `RuntimeSettingsSnapshot`, normalized parallax budgets, and `CapturedFrame` metadata.
- Do not commit or upload runtime artifacts: `models/`, `outputs/`, `python3/`, `python-cu13/`, `downloads/`, `.codegraph/`, or `4K.jpg`.

## Current Known Issues

- None currently recorded for this handoff.

## Current Status

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

- `process_openxr_frame()` still reports legacy `openxr_ipd/openxr_depth_strength/openxr_stereo_scale/openxr_max_shift_ratio` debug fields for compatibility, alongside normalized resolved disparity debug fields.
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

