# OpenXR Stereo Runtime Plan

## Current Scope

The local Python environment has pyopenxr available as the `xr` module.

The lab repository now has OpenXR stereo rendering core helpers in:

```text
src/stereo_lab/openxr_render.py
```

Preview script:

```text
scripts/generate_openxr_stereo_preview.py
```

This module implements the part that prevents rotation-related stereo distortion:

- per-eye RGB+depth reprojection;
- roll-adaptive parallax direction;
- OpenXR-style screen model matrix;
- pose-to-view matrix;
- asymmetric FOV projection matrix.

It does not yet own a full OpenXR session loop, swapchain creation, D3D11/GL texture upload, or `xr.CompositionLayerProjection` submission.

## Why Fixed SBS Is Not Enough

Fixed SBS/TAB output bakes the stereo disparity into image space. If a VR virtual screen is later rolled or otherwise rotated, the baked disparity does not automatically rotate with the screen local axes.

For OpenXR mode, the correct path is:

1. keep `rgb` and `depth` available as runtime textures/tensors;
2. locate OpenXR views every frame;
3. render each eye independently;
4. bind disparity to the virtual screen local direction, especially `screen_roll`.

## Desktop2Stereo Reference

Desktop2Stereo's OpenXR path uses the same architectural pattern:

- `xrviewer_core.py` computes per-eye view/projection from `xr.locate_views()`;
- `_build_model_mat4()` applies virtual screen yaw/pitch/roll;
- `_render_eye()` sets `u_eye_offset`, depth strength, convergence, and `u_roll`;
- the GLSL shader uses `cos(u_roll), sin(u_roll)` to rotate the parallax direction.

## Implemented In Lab

`render_openxr_stereo(rgb, depth, OpenXRRenderConfig(screen_roll=...))` returns left/right eye tensors where parallax follows `screen_roll`.

`screen_roll` is an arbitrary float angle in radians, not a 90-degree-only switch. The preview script accepts any angle in degrees via `--screen-roll-deg`.

At `screen_roll=0`, the result matches the existing horizontal warp path. At non-zero roll, the sampling direction rotates in UV space:

```text
parallax_dir = (cos(screen_roll), sin(screen_roll))
```

The module also provides:

- `is_pyopenxr_available()` for checking the local `xr` module;
- `OpenXREyeView` / `OpenXRFov` data structures;
- `build_openxr_eye_mvp()` for per-eye projection-view-screen matrix composition.

## Remaining Runtime Work

To become a complete OpenXR output mode, add:

1. OpenXR session lifecycle using `xr`.
2. Per-eye swapchain creation.
3. GPU texture upload path for `rgb` and `depth`.
4. Per-eye render into swapchain images.
5. `xr.CompositionLayerProjectionView` submission.
6. Optional D3D11/Triton shader path for performance parity with Desktop2Stereo.

The new core module is intended to be reused by that runtime integration instead of producing fixed SBS for VR.
