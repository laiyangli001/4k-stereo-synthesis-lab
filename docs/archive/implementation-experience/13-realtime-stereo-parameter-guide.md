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
| `temporal_strength` | `0.7` | `temporal=True` 时的在线 temporal 平滑强度。 |
| `auto_scene_reset` / `auto_reset_temporal` | `True` | 启用在线场景变化重置。GUI 的实时播放器/VR 模式在启用 temporal 时建议同时启用。内部字段是 `auto_reset_temporal`。 |
| `scene_reset_threshold` | `0.22` | 场景重置的平均帧差阈值。 |
| `reset_cooldown_frames` | `3` | 检测到切镜后，抑制重复触发 reset 的冷却帧数。 |
| `edge_dilation` | `2` | 遮挡 mask 的膨胀半径。 |
| `edge_threshold` | `0.04` | depth 边缘检测阈值，用于遮挡区域判断。 |
| `mask_feather_radius` | `3` | 对遮挡补洞 mask 做边缘羽化。默认 `3` 是当前视觉回归里边缘虚影和边缘硬切之间的折中值。 |
| `hole_fill_mode` | `soft_low_ghost` | GUI 暴露的补洞模式。`soft_low_ghost` 用更软的补洞策略降低边缘重复轮廓；`balanced` 仍保留为速度和锐度更均衡的选项。 |
| `hole_fill_radius` | `3` | 由 `hole_fill_mode` 派生的补洞采样半径。只有旧配置没有 `Hole Fill Mode` 时才直接读取。 |
| `hole_fill_strength` | `1.0` | 由 `hole_fill_mode` 派生的补洞混合强度。只有旧配置没有 `Hole Fill Mode` 时才直接读取。 |
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
| `foreground_scale` | `0.5` | 已实现。可选的前景/depth 对比度重映射。 |
| `depth_antialias_strength` | `2.0` | 已实现。GUI 中对应 `Anti-aliasing=2`，用于改善边缘稳定性。 |
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
| `mask_feather_radius` | `StereoConfig.mask_feather_radius` |
| `hole_fill_mode` | `StereoConfig.hole_fill_mode` |
| `hole_fill_radius` | `StereoConfig.hole_fill_radius`，由 `hole_fill_mode` 派生，旧配置兼容 |
| `hole_fill_strength` | `StereoConfig.hole_fill_strength`，由 `hole_fill_mode` 派生，旧配置兼容 |
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

如果你要开发一个后台服务来实时切换性能模式/灯光，建议组合以下信号：

- 特征	桌面办公	看视频	游戏
- GPU 3D 使用率	<5%	<15%	>50% 且持续
- GPU Video Decode	<5%	>20%（硬解）	<10%
- 前台窗口全屏	通常否	是（全屏视频）或最大化	是（真全屏）
- 前台进程类别	Office、浏览器（非播放）	播放器、浏览器（视频页）	游戏进程（可白名单）
- 音频输出	无或偶尔系统音	持续音频流	持续音效/音乐
- 键鼠输入频率	间歇、短点击	几乎无（长时间空闲）	高频、连续（WASD/鼠标）

落地算法：

- 定时采集 GPU 引擎负载（取 2~3 秒均值，防抖动）。

- 若 VideoDecode > 25% 且 3D < 20% → 直接判为 视频场景。

- 若 3D > 60% 或（3D > 30% 且前台进程为游戏）→ 判为 游戏场景。

- 若以上均不满足，且 CPU/GPU 整体空闲，键鼠空闲时常 >30 秒 → 静态办公。

- 其他状态保留上一次判定或判为“混合”。



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

## 三类最终内容模式

面向最终使用场景，模式应收敛为三类：

```text
电影
游戏
图片
```

当前 API 仍保留 `cinema`、`game_low_latency`、`still_image_hq`、`debug_export` 等 preset 名称，用于兼容已有配置和调参；但宿主产品层可以把它们组织成更简单的三类内容模式：

| 产品模式 | 当前 preset / 路径 | 说明 |
|---|---|---|
| 电影 | `cinema` | 全屏视频、播放器、稳定视频内容 |
| 游戏 | `game_low_latency` | 全屏游戏、高 GPU 3D、高输入频率、低延迟优先 |
| 图片 | `still_image_hq` / flat depth / thumbnail refine | 静态图片、普通桌面、浏览器、图片搜索页、截图 |

普通桌面办公不再作为独立模式。很少有人长期戴着 VR/头显做常规办公，因此桌面、网页、文件管理器、图片搜索页等都归入“图片 / 静态画面”处理。

## Depth Safety Gate

Depth Safety Gate 已移除。实时、本地、OpenXR 和图片路径都不再执行低纹理/缩略图安全门控，也不再提供 Depth Safety 参数。


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
mask_feather_radius: 0 / 1 / 2 / 3 / 4 / 5
hole_fill_mode: balanced / soft_low_ghost / sharp_test
foreground_scale: 0.0 / 0.2 / 0.4
depth_antialias_strength: 0.0 / 1.0 / 2.0
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

## 电影 / 游戏 / 图片模式自动测试

三类模式的自动测试应使用固定 manifest，而不是临时挑一张 4K 图。manifest 的作用是把“这张图代表什么场景”“期望走哪个 preset”“肉眼重点检查什么”写死，保证每次调参都能复现同一组比较。

推荐输出目录：

```text
outputs/visual_preset_matrix/<timestamp>/
  summary.json
  summary.md
  cinema/<sample_id>/
  game_low_latency/<sample_id>/
  still_image_hq/<sample_id>/
```

自动测试入口：

```powershell
.\python3\python.exe -B scripts\tools\generate_preset_visual_matrix.py `
  --manifest samples\visual_preset_manifest.json `
  --depth-backend tensorrt_native `
  --out-dir outputs\visual_preset_matrix
```

如果只验证 manifest 和输出结构，不跑模型：

```powershell
.\python3\python.exe -B scripts\tools\generate_preset_visual_matrix.py `
  --manifest samples\visual_preset_manifest.json `
  --depth-backend luma `
  --out-dir outputs\visual_preset_matrix `
  --dry-run
```

manifest 示例：

```json
{
  "samples": [
    {
      "id": "cinema_face_closeup",
      "path": "cinema/face_closeup.png",
      "category": "cinema",
      "expected_preset": "cinema",
      "checks": ["face_edges", "hair_occlusion", "subtitle_safe"]
    },
    {
      "id": "game_hud_motion",
      "path": "game/hud_motion.png",
      "category": "game",
      "expected_preset": "game_low_latency",
      "checks": ["hud_edges", "no_temporal_lag", "no_softening"]
    },
    {
      "id": "image_portrait_single",
      "path": "image/portrait_single.png",
      "category": "image_natural",
      "expected_preset": "still_image_hq",
      "checks": ["subject_edges", "natural_depth", "background_not_overpulled"]
    },
    {
      "id": "image_thumbnail_grid",
      "path": "image/thumbnail_grid.png",
      "category": "image_thumbnail_grid",
      "expected_preset": "still_image_hq",
      "depth_policy": "flat_or_thumbnail_refine",
      "checks": ["no_curved_screen", "thumbnail_boundaries_flat", "text_stable"]
    }
  ]
}
```

### 测试图片选择标准

最小样本集必须覆盖四类：`cinema`、`game`、`image_natural`、`image_unsafe_ui`。推荐再补 `image_thumbnail_grid` 和 `image_low_texture`。图片文件不建议直接提交到仓库，优先放在本地私有样本目录或由宿主产品测试集提供。

电影模式样本应覆盖：

- 人像近景：脸、头发、肩膀、衣领边缘清楚，用于看遮挡和人物边缘。
- 暗场：低亮度、高噪声、局部高光，用于看 depth 闪烁和错误凸起。
- 字幕场景：底部字幕或 UI 条，用于看文字是否被拉出深度或中线异常。
- 前景 / 背景高对比：人物、栏杆、树枝、门框等，用于看空洞、撕裂和重复纹理。
- 动画 / 二次元画面：大色块和强轮廓，用于看 depth 是否过锐或边缘毛刺。

游戏模式样本应覆盖：

- HUD / 小地图 / 血条 / 准星：文字和 UI 必须稳定，不能漂浮或软化。
- 快速运动截图：运动模糊、爆炸、粒子、快速转向，用于看低延迟参数是否保守。
- 暗色游戏场景：用来检查暗部错误 depth 和边缘空洞。
- UI overlay + 3D 背景：菜单、背包、弹窗叠在 3D 场景上，用于看 UI 平面是否被错误立体化。
- 高帧率交互场景：参数默认应减少 temporal lag，不能为了稳定牺牲响应。

图片模式样本必须同时包含“可做深度”的正样本和“应该平面化”的反样本：

- 单张大图 / 人像 / 风景：这是 `still_image_hq` 正样本，应允许更清晰、更强立体。
- 天空、墙面、海面、雪地等低纹理自然照片：不能只因低纹理就强制全图平面化。
- 桌面 / 文件管理器 / 设置页：应触发 flat depth 或强 flatten，避免整块屏幕弯曲。
- 浏览器普通页面 / 文档页：文字必须稳定，不能漂浮、凹陷或产生双边。
- 图片搜索 / 缩略图网格：先全局平面化，再可选后台 `thumbnail_hq_refine`，不能把整个网页当成一张自然照片。
- 纯色 / 深色背景：用于检查中心近、四角远的径向错误 depth 是否被安全门控压住。

图片选择的关键不是“漂亮”，而是覆盖失败模式。至少要有一半图片模式样本是负例：UI、缩略图、文本页、纯色背景。否则 `still_image_hq` 很容易在自然照片上看起来很好，但在真实桌面/浏览器里产生内凹屏幕、漂浮文字和错误深度。

### 自动测试判定

每个样本至少跑 `cinema`、`game_low_latency`、`still_image_hq` 三个 preset，生成同一套 `contact_sheet_labeled.png`、左右眼、SBS、depth_map、occlusion_mask 和 absdiff。最终判定不只看 expected preset，也要看其他 preset 是否暴露风险。

重点判定：

- Cinema：立体感稳定，人物/字幕/暗场没有明显错深度，temporal 不拖影。
- Game：HUD 和文字稳定，边缘不软化，快速画面不出现明显 temporal lag。
- Still Image / HQ 正样本：主体层次更清楚，边缘更干净，背景没有被过度拉出。
- Still Image / HQ 负样本：UI、文本、缩略图、纯色背景不应弯曲、漂浮或产生局部凹凸。
- Half-SBS / Full-SBS：中线不能异常，左右眼不能撕裂、空洞或重复纹理。

用于最终默认参数选择时，建议按样本记录人工判定：

```text
pass: 没有明显肉眼问题，可以作为默认候选
warn: 可接受但有轻微问题，只适合特定内容或高级选项
fail: 出现撕裂、空洞、重复纹理、文字漂浮、屏幕弯曲或明显不适
```

## 验证

修改 P0/P1 默认值或映射后，运行：

```powershell
.\python3\python.exe -B -m pytest -q
.\python3\python.exe -B -m compileall -q src scripts tests
```

视觉质量验收时，生成视觉回归集并重点检查：边缘、遮挡处、Half-SBS 中线、撕裂、空洞、重复纹理。
## 当前 GUI 调参速查

本节对应当前 GUI 和 `stereo_runtime` 的真实参数。多数高级立体参数会在 GUI 修改后自动保存到 `src/settings.yaml`，主程序热更新后立即生效；推理加速相关选项不属于本节。

### 核心视差参数

| 参数 | 当前范围 / 默认 | 原理 | 观察现象 | 调参建议 |
|---|---|---|---|---|
| `IPD (mm)` / `IPD` | GUI `58-70`；默认 `64` mm | 参与有效基线 `ipd_mm / 1000 * stereo_scale`，基线越大，两眼横向差越大。 | 增大后 3D 更强，但更容易头晕、边缘拉裂、近景过度凸出；减小后更舒服但立体弱。 | 先用 `64`；难融合降到 `60-62`；太平升到 `66-68`。 |
| `Depth Strength` | GUI `0.0-10.0`；默认 `2.0` | 直接放大深度转视差的强度。 | 越大前后层次越明显，也越容易出现背景跑前面、边缘撕裂、空洞。 | 游戏 `1.4-2.0`，电影 `2.0-2.8`，静态图 `2.5-3.5`。 |
| `Depth Quick` | `Soft / Standard / Enhanced` | 快捷设置 `Depth Strength`：`Soft=1.4`，`Standard=2.0`，`Enhanced=2.6`。 | Soft 稳定舒适；Enhanced 更强但更容易错位。 | 先用 Standard，太冲改 Soft。 |
| `Convergence` | GUI `-0.5` 到 `1.0`，步进 `0.25`；默认 `0.0` | 把某个深度值作为屏幕平面，接近该深度的区域视差趋近 0。 | 改变“谁在屏幕上、谁凸出、谁退后”。背景像在前面时，先确认眼序，再调它。 | 电影/游戏常用 `0.25-0.5`；近景太顶眼则提高；整体太往后则降低。 |
| `Max Shift Ratio` / 位移比例 | GUI `0.00-0.10`，步进 `0.01`；默认 `0.05` | 限制最大像素位移，避免视差无限变大。 | 越大 3D 上限越强，空洞和边缘问题越明显；越小更稳但立体被压平。 | 实时游戏 `0.03-0.05`；电影 `0.05`；静态图 `0.05-0.08`。 |
| `Stereo Scale` | 内部默认 `0.5`，preset 约 `0.42-0.65` | 缩放物理 IPD，和 IPD 一起决定有效基线。 | 类似全局立体强度旋钮。 | 不建议和 IPD 同时乱调；当前主要由 preset 控制。 |

### 深度预处理参数

| 参数 | 当前范围 / 默认 | 原理 | 观察现象 | 调参建议 |
|---|---|---|---|---|
| `Foreground Scale` | GUI `-0.9` 到 `5.0`；默认 `0.5` | 对深度曲线做非线性重映射，改变前景/背景深度分布；算法要求值必须大于 `-1.0`。 | 正值通常强调前景层次；负值压缩/柔化前景差异；过大容易近景过冲，过低会变平。 | 近景太炸或背景关系怪时试 `-0.5, 0, 0.5`，不要直接用极端值。 |
| `Anti-aliasing` / `Depth Antialias Strength` | GUI `0-10`；默认 `2` / `2.0` | 平滑深度图，减少硬边和噪声。 | 增大后边缘更稳、闪烁少，但深度边界会糊，前景轮廓可能不准。 | 最低质量 `fast` 会强制归零；电影默认 `2`；边缘发糊就降到 `0-1`。 |

### 合成质量与速度参数

| 参数 | 当前范围 / 默认 | 原理 | 观察现象 | 调参建议 |
|---|---|---|---|---|
| `Stereo Preset` | `cinema / game_low_latency / still_image_hq / debug_export`；默认 `cinema` | 一组参数预设。GUI 显示为 Cinema / Balance、Game / Low Latency、Image / High Quality、Debug / Export。 | 控制台会打印当前 preset。固定 preset 更容易调参。 | 调参时先固定 `cinema` 或 `game_low_latency`。 |
| `Synthetic View` / `Stereo Quality` | `fast / fast_plus / quality_4k / hq_4k`；默认 `quality_4k` | `fast` 基础视差；`fast_plus` 加轻量边缘 mask/hole fill；`quality_4k` 分层合成和更完整遮挡处理；`hq_4k` 至少 3 层。 | fast 最快但边缘洞明显；fast_plus 平衡；quality_4k 边缘更好但慢；hq_4k 更稳更慢。 | 实时先 `fast_plus`；画质优先 `quality_4k`；静态图可用 `hq_4k`。 |
| `Edge Threshold` | GUI `0.00-0.10`，步进 `0.01`；默认 `0.04` | 深度边缘检测阈值；低值会检测更多边缘。 | 低值更积极补边/遮挡，但可能误处理纹理；高值更保守，洞可能更多。 | 边缘裂缝多用 `0.02-0.04`；误修多用 `0.06-0.10`。 |
| `Edge Dilation` | GUI `0-4`；默认 `2` | 把检测到的边缘 mask 扩张。 | 越大遮挡边缘处理范围越宽，洞少但边界可能糊或变胖。 | 实时 `1-2`；静态图 `2-3`；边缘发糊就降。 |
| `Mask Feather` / `Mask Feather Radius` | GUI `0-5`；默认 `3` | 对遮挡补洞 mask 的边缘做羽化，减少硬切和双边虚影。 | 增大后边缘更柔和、重影更少，但细边可能变软；减小后更锐，但边缘虚影更容易出现。 | 当前视觉回归建议默认 `3`；需要更锐可试 `1-2`，边缘虚影明显可试 `4`。 |
| `Hole Fill Mode` | GUI `Balanced / Soft / Low Ghost / Sharp Test`；默认 `Soft / Low Ghost` | 用有限预设控制补洞半径和混合强度，避免直接暴露自由 radius/strength。 | Balanced 画质和速度更均衡；Soft / Low Ghost 重影更少但边缘更软；Sharp Test 用于对比锐度，不建议日常默认。 | 默认使用 Soft / Low Ghost 优先压低边缘重复轮廓；需要更锐利边缘时切回 Balanced。 |
| `hole_fill` | 内部：`none / fast / edge_aware` | 对遮挡/空洞区域做填补。`fast` 较轻，`edge_aware` 更重。 | 开启后空洞减少，但边界可能被涂抹。 | 实时低延迟用 `fast`；画质优先用 `edge_aware`。 |
| `layers` | 内部默认 `2`，`hq_4k` 至少 `3` | 分层合成时把深度分成多层再合成。 | 层数更多，遮挡关系通常更稳，但速度更慢。 | 实时保持 `2`；静态高质量可 `3`。 |

### 时间稳定参数

| 参数 | 当前范围 / 默认 | 原理 | 观察现象 | 调参建议 |
|---|---|---|---|---|
| `Temporal Strength` | GUI `0.0-1.0`，步进 `0.1`；默认 `0.7`，`0` 表示关闭 | 跨帧混合左右眼结果，稳定闪烁。 | 高值减少抖动，但快速运动会拖影、延迟感更重。 | 游戏 `0-0.4`；电影 `0.6-0.85`；静态图可关闭。 |
| `Scene Reset Threshold` | GUI `0.00-0.35`；默认 `0.22`，`0` 表示关闭自动重置 | 检测画面变化过大时重置 temporal，避免场景切换残影。 | 太低会频繁重置导致不稳定；太高会切场景后拖影。 | 电影 `0.18-0.28`；游戏可 `0` 或 `0.22`。 |
| `Reset Cooldown Frames` | GUI `1,2,3,4,6`；默认 `3` | 场景重置后的冷却帧数，防止连续触发。 | 低值响应快但可能抖；高值更稳但恢复慢。 | 实时 `1-3`；电影 `3-4`。 |

### 输出与观看参数

| 参数 | 当前范围 / 默认 | 原理 | 观察现象 | 调参建议 |
|---|---|---|---|---|
| `Display Mode` / `Output Format` | `Half-SBS, Full-SBS, Half-TAB, Full-TAB, Depth Map, Anaglyph, Interleaved, Mono, Leia`；默认 `Half-SBS` | 只决定打包方式，不应改变左右眼语义。 | Half-SBS 宽度压半，兼容性好；Full-SBS 保留单眼完整宽度但输出更大。 | 普通 3D 显示器/播放器按设备要求选；调试眼序用 Full-SBS 最直观。 |
| `Cross Eyed` | `true/false`；默认 `false` | 只交换左右眼，给交叉眼观看用。 | 普通显示模式下不应依赖它；如果开了才正常，说明眼序反了。 | 普通设备 false；交叉眼裸眼看 SBS 才 true。 |
| `Anaglyph Method` | `red_cyan / green_magenta / amber_blue`；默认 `red_cyan` | 红蓝/补色模式通道混合方式。 | 不同眼镜颜色匹配不同，颜色会严重失真。 | 只有 Display Mode = Anaglyph 时才关心。 |

### 推荐观察流程

1. 固定 `Stereo Preset=cinema` 或 `game_low_latency`，不要先用 auto。
2. 固定 `Cross Eyed=false`，确认普通 SBS 左眼在左、右眼在右。
3. 用同一张有前景、中景、背景的画面，只改一个参数观察。
4. 先调 `Convergence`，再调 `Depth Strength`，再调 `Max Shift Ratio`。
5. 之后再细调 `Foreground Scale`、`Edge Threshold`、`Edge Dilation`。
6. 观察标准：近景应比背景更凸出；背景不应跑到前景前面；垂直边缘不应大面积撕裂；快速移动时不应明显拖影；眼睛应能在 5-10 秒内自然融合。

### 常见现象对照

| 现象 | 优先检查 / 调整 |
|---|---|
| 普通显示必须开 Cross Eyed 才正常 | 眼序错误。普通模式应为 `Cross Eyed=false`。 |
| 近景太顶眼、难融合 | 降低 `Depth Strength` 或 `IPD`，提高 `Convergence`，降低 `Max Shift Ratio`。 |
| 整体太平 | 提高 `Depth Strength`，略增 `IPD` 或 `Max Shift Ratio`。 |
| 背景像跑到前面 | 先确认眼序；再调 `Convergence` 和 `Foreground Scale`。 |
| 边缘裂缝、空洞明显 | 降低 `Max Shift Ratio`，降低 `Edge Threshold`，提高 `Edge Dilation`，或从 `fast` 切到 `fast_plus/quality_4k`。 |
| 边界发糊、前景轮廓变胖 | 降低 `Edge Dilation`、`Anti-aliasing` 或 hole fill 强度。 |
| 快速运动拖影 | 降低 `Temporal Strength`，降低或关闭自动 scene reset 的误触发，游戏模式可接近 `0`。 |
