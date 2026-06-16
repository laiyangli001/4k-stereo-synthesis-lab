# 实时立体参数说明

本文定义实时桌面、播放器、VR 场景的立体参数面、参考标准、默认值，以及 GUI/API 传参映射。

## 范围

实时路径面向低延迟连续画面。方案结合 iw3 的参数化思路、Desktop2Stereo 的实时运行边界，以及 VR 舒适度约束。

最终实时视频/VR 主路径：

```text
RGB frame -> depth inference -> depth postprocess -> online scene reset -> light temporal -> per-eye stereo/OpenXR render
```

固定显示输出时，最后一步把左右眼打包为 `half_sbs`、`full_sbs`、`half_tab`、`full_tab`、`anaglyph`、`interleaved`、`leia`、`mono` 或 `depth_map`。OpenXR 输出时，最后一步必须使用运行时每眼 pose/FOV/roll，不能用固定 SBS 冒充 VR 输出。

## 参考标准

OpenXR 是 VR runtime 输出的硬标准。它负责：

- 每眼 pose/FOV。
- swapchain 归属。
- projection layer。
- tracking space。
- frame timing。

OpenXR 输出不应该被表示成固定 SBS，再假装成 VR 输出。

DIBR / 2D-to-3D 视频处理没有统一工业标准，但有通用工程原则：

- 视差必须有上限，避免眼疲劳。
- 零视差 / 收敛面必须可控。
- 遮挡边界必须处理。
- temporal 状态不能跨镜头污染。
- depth 边缘平滑要保守。

VR 舒适度按产品要求处理：

- 低延迟优先。
- 不积压旧帧。
- 避免明显 temporal lag。
- 避免过大视差和突然的视差跳变。
- VR 中宁愿 3D 强度弱一点，也不要错深度或可见拖影。

以下离线视频处理项明确不进入实时主路径：

- Offline video lookahead。
- TransNetV2 scene detection。
- HDR/video codec pipeline。

## P0 参数

P0 必须暴露给 GUI/API 调用方。所有参数都有默认值，调用方可以只传用户实际修改的值。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `depth_strength` | `2.0` | 立体视差强度。值越高，立体感越强，但边缘伪影和眼疲劳风险也越高。 |
| `convergence` | `0.0` | 深度收敛偏移，用于移动立体零视差平面。 |
| `ipd` | `0.064` | 瞳距，单位米。 |
| `max_shift_ratio` | `0.05` | 最大水平位移，占画面宽度的比例。 |
| `temporal_strength` | `0.85` | `temporal=True` 时的在线 temporal 平滑强度。 |
| `auto_scene_reset` / `auto_reset_temporal` | `False` | 启用在线场景变化重置。GUI 的实时播放器/VR 模式在启用 temporal 时建议同时启用。内部字段是 `auto_reset_temporal`。 |
| `scene_reset_threshold` | `0.22` | 场景重置的平均帧差阈值。 |
| `reset_cooldown_frames` | `3` | 检测到切镜后，抑制重复触发 reset 的冷却帧数。 |
| `edge_dilation` | `2` | 遮挡 mask 的膨胀半径。 |
| `edge_threshold` | `0.04` | depth 边缘检测阈值，用于遮挡区域判断。 |
| OpenXR per-eye pose/fov/roll | runtime 提供 | VR runtime 应传入每眼 pose/FOV 和任意角度 screen roll。 |

建议实时默认策略：

```text
temporal=True
temporal_strength=0.65-0.80 for VR, up to 0.85 for desktop/player
auto_reset_temporal=True
scene_reset_threshold=0.18-0.25
reset_cooldown_frames=2-5
```

代码默认值保持保守和向后兼容。脚本需要显式传 `--temporal` 和 `--auto-reset-temporal`，这样旧 benchmark 仍然可比。

## P1 参数

P1 建议暴露给质量/HQ 调节，但默认保持兼容行为。

| 参数 | 默认值 | 状态 |
|---|---:|---|
| `foreground_scale` | `0.0` | 已实现。可选的前景/depth 对比度重映射。 |
| `depth_antialias_strength` | `0.0` | 已实现。可选 depth 抗锯齿，用于改善边缘稳定性。 |
| `synthetic_view` | `backend` selection | 只作为模式选择暴露。映射到 `backend=fast/quality_4k/hq_4k` 或 OpenXR render mode。 |
| `cross_eyed` | `False` | 已实现。在最终输出打包前交换左右眼。 |
| `anaglyph_method` | `red_cyan` | 已实现。`red_cyan` 可走 Triton；其他方法走 PyTorch fallback。 |

支持的 anaglyph 方法：

- `red_cyan`
- `green_magenta`
- `amber_blue`
- `gray`

## GUI 传参约定

GUI 调用方应把配置当作部分覆盖，只传用户修改过的值：

```python
StereoConfig(
    backend="quality_4k",
    output_format="half_sbs",
    depth_strength=user_depth_strength,
    temporal=True,
    temporal_strength=0.75,
    auto_reset_temporal=True,
)
```

未传入的值由 `StereoConfig` 默认值补齐。OpenXR 集成时，GUI/runtime 应把 runtime pose/FOV/roll 传给 OpenXR 层，不要先转换成固定 SBS。

建议 GUI 命名：

| GUI 名称 | 内部字段 |
|---|---|
| `depth_strength` | `StereoConfig.depth_strength` / `OpenXRRenderConfig.depth_strength` |
| `convergence` | `StereoConfig.convergence` / `OpenXRRenderConfig.convergence` |
| `ipd` | `StereoConfig.ipd` / `OpenXRRenderConfig.ipd` |
| `max_shift` | `StereoConfig.max_shift_ratio` / `OpenXRRenderConfig.max_shift_ratio` |
| `temporal_strength` | `StereoConfig.temporal_strength` |
| `auto_scene_reset` | `StereoConfig.auto_reset_temporal` |
| `scene_reset_threshold` | `StereoConfig.scene_reset_threshold` |
| `edge_dilation` | `StereoConfig.edge_dilation` |
| `foreground_scale` | `StereoConfig.foreground_scale` |
| `depth_antialias_strength` | `StereoConfig.depth_antialias_strength` |
| `synthetic_view` | `StereoConfig.backend` 或 OpenXR mode |
| `cross_eyed` | `StereoConfig.cross_eyed` |
| `anaglyph_method` | `StereoConfig.anaglyph_method` |
| OpenXR `pose/fov/roll` | `OpenXREyeView`, `OpenXRFov`, `OpenXRScreenPose`, `OpenXRRenderConfig.screen_roll` |

## 验证

修改 P0/P1 默认值或映射后，运行：

```powershell
.\python3\python.exe -B -m pytest -q
.\python3\python.exe -B -m compileall -q src scripts tests
```

视觉质量验收时，生成视觉回归集并重点检查：边缘、遮挡处、Half-SBS 中线、撕裂、空洞、重复纹理。
