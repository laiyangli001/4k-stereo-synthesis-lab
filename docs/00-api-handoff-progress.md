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
refactor: enforce runtime temporal contracts
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

- Added `auto_reset_temporal`, `scene_reset_threshold`, and `reset_cooldown_frames` to `RuntimeSettingsSnapshot` hot-reload classification and runtime config update mapping.
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

- `StereoConfig` and `OpenXRRenderConfig` still keep `legacy` defaults for low-level direct construction and compatibility; the host-facing `StereoRuntimeConfig` default now selects `standard`.
- OpenXR direct path still exposes legacy debug/uniform names such as `openxr_ipd`, `openxr_depth_strength`, `openxr_stereo_scale`, and `openxr_max_shift_ratio` alongside normalized parallax debug fields.
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
