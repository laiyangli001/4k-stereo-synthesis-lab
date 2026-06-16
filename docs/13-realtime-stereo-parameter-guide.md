# 实时立体参数说明

本文定义实时桌面、播放器、VR 场景的立体参数面、参考标准、默认值、GUI/API 传参映射，以及用视觉回归测试寻找最优参数的方法。

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
temporal_strength=0.75-0.85 for Cinema, 0.55-0.70 for Game / Low Latency
auto_reset_temporal=True
scene_reset_threshold=0.18-0.25
reset_cooldown_frames=2-5
```

代码默认值保持保守和向后兼容。脚本需要显式传 `--temporal` 和 `--auto-reset-temporal`，这样旧 benchmark 仍然可比。

## P1 参数

P1 建议暴露给静态图片/HQ 调节，但默认保持兼容行为。

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

## Auto Mode 自动模式

GUI 默认应使用 `Auto Mode`，手动选择模式作为高级选项保留。目标是让普通用户不需要理解 Cinema、Game、Still Image/HQ 的差异，也能在多数场景下获得合适参数。

Auto Mode 不应该试图“完全猜中用户意图”，而应该做低风险的场景分类：

```text
Auto Mode
-> Cinema
-> Game / Low Latency
-> Still Image / HQ
-> Debug / Export
```

可检测信号：

- 画面运动强度：连续帧差很小，倾向 Still Image / HQ；中等变化且节奏稳定，倾向 Cinema；高频快速变化，倾向 Game / Low Latency。
- 前台进程 / 窗口类型：播放器进程倾向 Cinema；游戏、全屏 DirectX/Vulkan/OpenGL 窗口倾向 Game / Low Latency。
- 帧率和延迟压力：24/30/60fps 稳定视频倾向 Cinema；高 FPS、强交互、低延迟要求倾向 Game / Low Latency。
- 静止时长：画面静止超过 `1.0-2.0` 秒，倾向 Still Image / HQ；恢复运动后回到 Cinema 或 Game / Low Latency。
- OpenXR 状态：只要处于 HMD/OpenXR 输出，参数应整体偏保守，避免过强视差、temporal lag 和错深度。
- 用户动作：进入截图、导出、视觉回归时，倾向 Still Image / HQ 或 Debug / Export。

推荐默认分类规则：

```text
if user_export_action:
    mode = Debug / Export
elif still_duration > 1.0-2.0 seconds:
    mode = Still Image / HQ
elif game_or_fast_interaction_detected:
    mode = Game / Low Latency
elif player_or_stable_video_detected:
    mode = Cinema
else:
    mode = Cinema
```

模式切换必须防抖，不能频繁跳变：

- 连续 N 帧满足条件后才切换。
- 切换后至少保持 `2-5` 秒。
- 参数使用渐变，不要瞬间跳变。
- 检测到 scene reset 或剧烈变化时，可以快速降级到 Game / Low Latency。
- 从 Game / Low Latency 回到 Cinema 或 Still Image / HQ 时应更慢，避免来回振荡。

建议 GUI 暴露：

```text
模式:
- Auto 推荐
- Cinema
- Game / Low Latency
- Still Image / HQ
- Debug / Export
```

Auto Mode 内部可以维护一个轻量状态机：

```text
AutoSceneClassifier:
    frame_motion_score
    scene_cut_score
    still_duration
    foreground_process
    fullscreen_state
    openxr_active
    user_export_action
```

状态机输出：

```text
StereoModePreset
StereoConfig
OpenXRRenderConfig
```

原则：默认 Auto，手动模式给高级用户；自动模式负责减少普通用户误选模式导致的画质、延迟或舒适度问题。

## 视觉回归调参方法

最优参数可以通过视觉回归测试判断，但不能只看一张图。单张 4K 图只能判断当前画面是否更好，不能代表所有视频、播放器和 VR 场景的全局最优。

推荐建立一组代表性样本：

- 人像近景：检查脸部、头发、肩膀边缘和前景平面感。
- 快速运动：检查拖影、temporal lag 和边缘抖动。
- 高对比边缘：检查撕裂、空洞、重复纹理和遮挡边界。
- 暗场：检查 depth 噪声、闪烁和错误凸起。
- 字幕 / GUI 边缘：检查文字变形、中线异常和高频边缘错位。
- 镜头切换：检查 `auto_reset_temporal` 是否及时清掉旧状态。
- VR/OpenXR 旋转画面：检查任意 roll 下立体方向是否跟随屏幕姿态。

建议参数 sweep：

```text
depth_strength: 1.5 / 2.0 / 2.5 / 3.0
temporal_strength: 0.65 / 0.75 / 0.85
scene_reset_threshold: 0.18 / 0.22 / 0.25
edge_dilation: 1 / 2 / 3
foreground_scale: 0.0 / 0.2 / 0.4
depth_antialias_strength: 0.0 / 0.5 / 1.0
```

视觉验收重点：

- 边缘是否撕裂。
- 遮挡处是否有空洞。
- Half-SBS 中线是否异常。
- 是否出现重复纹理。
- temporal 后是否有拖影。
- depth 是否过平、过锐或边缘闪烁。
- VR 中是否出现错深度、突变视差或明显不适。

推荐按内容运动特征和延迟要求输出四类配置，而不是按“普通显示器 / VR”粗分：

- Cinema 模式：默认观影模式，适合电影、剧集、动画等相对平缓画面。目标是平衡立体强度、画面稳定性和舒适度。
- Game / Low Latency 模式：适合游戏、桌面操作、快速鼠标移动、UI 高频变化和剧烈画面切换。目标是低延迟、快速响应，降低 temporal lag。
- Still Image / HQ 模式：静态图片、暂停帧、截图、单张 2D -> 3D 生成等质量优先模式。可以追求更清晰、更强立体，也可以适度启用 `foreground_scale` 和 `depth_antialias_strength`，但必须通过视觉回归保护。
- Debug / Export 模式：用于 SBS/显示器预览、截图、文件导出和算法调试，不代表 HMD/OpenXR 主观舒适默认值。

建议初始参数范围：

| 模式 | `depth_strength` | `temporal_strength` | `scene_reset_threshold` | `reset_cooldown_frames` | `edge_dilation` | `depth_antialias_strength` | `foreground_scale` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cinema | `1.8-2.3` | `0.75-0.85` | `0.18-0.22` | `2-4` | `2` | `0.3-0.7` | `0.0-0.2` |
| Game / Low Latency | `1.4-1.9` | `0.55-0.70` | `0.16-0.20` | `1-2` | `1-2` | `0.0-0.4` | `0.0` |
| Still Image / HQ | `2.0-2.8` | `0.0` | `n/a` | `n/a` | `2-3` | `0.5-1.0` | `0.2-0.5` |
| Debug / Export | `2.0-3.0` | `0.75-0.85` | `0.20-0.25` | `2-5` | `2-3` | `0.0-1.0` | `0.0-0.4` |

Still Image / HQ 中的 `foreground_scale` 和 `depth_antialias_strength` 不能理解成“必开增强”。

- `foreground_scale` 会改变 depth 分布，让前景或主体更突出。它可能改善人物和主体层次，也可能把脸、墙面、天空、字幕等本应平缓的区域拉出错误深度。
- `depth_antialias_strength` 会平滑 depth 边缘。它可能减少边缘锯齿、闪烁和遮挡毛刺，也可能让前景边界变软，造成软边、漏深度或细节损失。

因此 Still Image / HQ 的后处理必须用固定样本做视觉回归对比：

```text
不开后处理
foreground_scale=0.2
depth_antialias_strength=0.5
foreground_scale=0.2 + depth_antialias_strength=0.5
```

重点看人物边缘是否软化、字幕是否变形、遮挡边界是否更干净、墙面/天空是否被错误拉出深度。对于暂停帧和单张图片，`temporal` 和 `auto_reset_temporal` 应关闭，因为没有跨帧状态需要平滑或重置。

推荐判定方式：

```text
1. 先固定 backend/output_format/depth model。
2. 每次只改变一组参数，生成视觉回归输出。
3. 对比 contact_sheet_labeled.png、左右眼、depth_map、occlusion_mask、absdiff。
4. 记录每组参数的可见问题和性能影响。
5. 选出 Cinema、Game / Low Latency、Still Image / HQ、Debug / Export 四套默认值。
```

视觉回归是调参的主要依据之一，但最终默认值必须同时满足画质、延迟和 VR 舒适度。

## 验证

修改 P0/P1 默认值或映射后，运行：

```powershell
.\python3\python.exe -B -m pytest -q
.\python3\python.exe -B -m compileall -q src scripts tests
```

视觉质量验收时，生成视觉回归集并重点检查：边缘、遮挡处、Half-SBS 中线、撕裂、空洞、重复纹理。
