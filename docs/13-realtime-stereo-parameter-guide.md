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

## 图片模式的 Depth Safety Gate

并不是所有画面都适合做单目深度推理。桌面 UI、浏览器普通页面、百度图片缩略图网格、文件管理器、设置页、纯色或深色背景，本质上都是屏幕上的二维信息面。

如果对这些画面强行做单目深度估计，深度模型很容易生成错误深度，表现为：

- 整个画面像一张复杂纹理平面，被预测成轻微内凹或外凸的曲面。
- 缩略图之间的边界被误认为物体轮廓，产生局部凹凸。
- 大面积纯色区域缺少纹理线索，被模型先验填成“远处”或径向深度。
- 在 SBS / VR / 裸眼 3D 中，屏幕中心可能前凸、四边后凹，形成明显“内凹感”。
- UI 文字、卡片、缩略图边缘产生不稳定漂浮和深度跳变。

核心判断：

```text
不是所有画面都应该启用 depth inference。
如果场景本身是 2D UI 信息面，安全策略应该优先输出平面 depth。
```

### 为什么缩略图 / UI 容易出错

打开图片搜索页时，模型输入是完整屏幕截图：

```text
screen capture -> depth model -> full-frame depth map
```

模型通常不会理解“这里有 30 张小图”，也没有实例分割能力。对它来说，这是一张纹理密集的大图。

结果：

- 缩略图网格被当成一整张复杂纹理平面。
- 小图边界和卡片边框被误认为真实物体边缘。
- 每张小图内部的真实照片内容和网页 UI 平面语义混在一起。

纯色 / 深色背景也会触发类似问题。单目深度模型依赖纹理、边缘、透视、物体形状等线索；大面积纯白、纯黑、灰色或极简深色背景会让模型进入不确定状态。

常见退化：

- 输出平均深度。
- 输出中心近、四角远的径向深度。
- 把背景当成远景，把屏幕边缘推远。

这些错误在立体显示中会被放大为内凹、外凸或屏幕弯曲。

这里必须掌握好度，不能因为局部低纹理就影响正常图片处理。正常照片中也常见天空、墙面、浅景深背景、雾、雪地、海面等低纹理区域，它们不应该被简单压平成屏幕平面。

纯色 / 低纹理安全门控只应该在“明显不可信”的情况下触发，例如：

- 大面积低纹理区域占比很高。
- RGB 边缘很少，但 depth 出现明显中心近、四角远的径向变化。
- depth 边缘和 RGB 边缘明显不对齐。
- 画面同时符合 UI / 网页 / 缩略图 / 大色块背景特征。
- Auto 场景信号不支持电影或游戏判断。

不应该单独因为 `rgb_texture_score` 低就强制平面 depth。建议至少两个以上安全指标同时触发，才执行 `force_flat_depth=True`。否则只降低 `depth_strength_scale`，或只对局部低置信区域做轻度 flatten。

根因：

- 主流单目深度模型主要训练于自然图像和真实场景，缺少桌面截图、浏览器页面、文件管理器、缩略图网格、深色 UI 等数据。
- 模型只做像素到深度的映射，不理解“这是浏览器窗口”“这是图片卡片”。
- 实时链路通常直接全图推理，缺少区域语义和前置场景判断。

### 图片模式安全策略

建议在图片模式中引入轻量 `Depth Safety Gate`：

```text
RGB frame
  -> async scene detector
  -> image/static suitability check
  -> if UI / low-texture / thumbnail grid: flat depth or thumbnail_hq_refine
  -> if single large image: depth provider
  -> depth confidence / quality gate
  -> stereo synthesis
```

推荐场景到 depth 策略映射：

| 场景 | depth 策略 |
|---|---|
| 桌面 / 文件管理器 / 普通浏览器页面 | 图片模式，默认平面 depth |
| 图片缩略图网格 | 图片模式，先平面 depth，再尽快异步 `thumbnail_hq_refine` |
| 纯色 / 低纹理背景 | 图片模式，输出平面 depth |
| 单张大图 / 静态照片 | 图片模式，可启用 depth inference |
| 全屏视频 | 电影模式，启用 depth，但限制最大视差 |
| 全屏游戏 | 游戏模式，启用 depth，低延迟优先 |

推荐轻量指标：

| 指标 | 用途 |
|---|---|
| `rgb_texture_score` | 判断是否大面积低纹理 / 纯色 |
| `large_flat_area_ratio` | 判断是否 UI / 背景平面占比过高 |
| `thumbnail_grid_score` | 判断是否存在大量规则缩略图网格 |
| `depth_variance` | 判断 depth 是否退化为弱变化或无意义曲面 |
| `center_bias_score` | 判断是否出现中心近、四角远的模型先验 |
| `depth_edge_rgb_edge_alignment` | 判断 depth 边缘是否和 RGB 边缘对齐 |

这些指标可以用 OpenCV / PyTorch 梯度、边缘、连通区域、矩形网格统计实现，不需要额外大模型。

建议的可信度分级：

| 判断 | 处理 |
|---|---|
| 单一低纹理信号，但像正常照片 | 保留 depth，仅轻微降低 `depth_strength_scale` |
| 低纹理 + depth/RGB 边缘不对齐 | 局部 flatten，降低强度 |
| 低纹理 + 大面积平面 + UI/网页特征 | 强制平面 depth |
| 低纹理 + 中心径向深度偏置明显 | 强制平面 depth 或大幅 flatten |
| 单张正常大图，有主体/透视/边缘线索 | 正常启用 depth |

默认策略应该偏向“不误伤正常图片”：

```text
force_flat_depth 只用于高置信 UI / 纯色背景 / 缩略图网格。
正常图片低纹理区域优先使用 depth_strength_scale 或局部 flatten，而不是全图平面化。
```

当画面不适合 depth，但不想完全关闭立体时，可以使用保守约束：

```text
depth = lerp(depth, flat_depth, flatten_strength)
depth_strength *= depth_strength_scale
edge_dilation 降低
foreground_scale 关闭
hole_fill 保守
temporal 开启但低强度
```

### 缩略图 HQ refine

逐张缩略图单独推理不推荐作为电影 / 游戏实时默认方案，但可以作为“图片模式”的后台增强能力。

适用条件：

- 当前不是电影模式。
- 当前不是游戏模式。
- 画面基本静态。
- GPU 有空闲。
- 检测到规则缩略图网格或多张图片卡片。

这种情况下可以尽快异步尝试，不必等待很久。推荐先输出平面 depth，后台立即开始缩略图检测和局部 refine；一旦局部 depth patch 可用，再渐进合成回全屏 depth。

关键限制：

- 不阻塞捕获、显示、depth 主链路。
- 缩略图卡片边框仍保持屏幕平面。
- 只在卡片内部允许低强度 3D。
- 每张缩略图内部 depth 独立归一化，但强度必须压低。
- patch 边缘需要 feather 到平面，避免卡片漂浮或边界撕裂。
- 用户滚动、切页、输入活跃时立刻取消或丢弃后台任务。

建议名称：

```text
thumbnail_hq_refine
```

推荐流程：

```text
检测静态缩略图网格
-> 先输出全局平面 depth
-> 后台检测每个 thumbnail rect
-> 对 rect 内图片裁剪后批量/队列推理
-> 生成局部 depth patch
-> patch depth 压低强度并 feather 到平面
-> 合成回全屏 depth
-> stereo 重新合成或渐进更新
```

建议后续实现 API：

```python
@dataclass(frozen=True)
class DepthSafetyDecision:
    use_depth: bool
    reason: str
    force_flat_depth: bool
    flatten_strength: float
    depth_strength_scale: float
    confidence: float
```

默认决策建议：

| 条件 | 默认决策 |
|---|---|
| `auto` + 全屏视频 | 电影：启用 depth，降低强度 |
| `auto` + 游戏 | 游戏：启用 depth |
| `auto` + 桌面 / 普通浏览器 / 文件管理器 | 图片：平面 depth |
| `auto` + 缩略图网格 | 图片：先平面 depth，再尽快异步 `thumbnail_hq_refine` |
| `auto` + 纯色 / 低纹理背景 | 图片：平面 depth |
| 手动 preset | 尊重用户选择，但可提供安全警告或可选 safety 开关 |

关键原则：

- 不通过降低 depth 推理分辨率解决这个问题。
- 不把错误 depth 继续立体化。
- 不依赖大白名单。
- 不在推理热路径同步采集系统指标。
- UI / 缩略图 / 纯色背景优先平面化。
- 用户进入单张大图、全屏视频、全屏游戏时再启用 depth。

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
