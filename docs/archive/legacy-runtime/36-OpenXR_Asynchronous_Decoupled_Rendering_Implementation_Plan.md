# OpenXR 异步解耦渲染重构实施计划书

## 1. 目标与边界

本文以 `docs/35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md` 为技术目标，面向 Desktop2Stereo 当前工程制定 OpenXR 异步重构实施计划。

核心目标不是在现有 OpenXR 路径上做局部补丁，而是把 OpenXR 输出重构为“Projection 主屏硬实时 + Quad 覆盖层软实时”的解耦架构：

- 虚拟显示器必须稳定跟随目标刷新率，优先消费最新 runtime 画面，并作为 Projection Layer 内 3D 屏幕平面绘制。
- Projection Layer 是常驻主场景层，负责房间/背景 fallback、controller、laser、虚拟屏幕主画面、屏幕边框和墙面反射。
- Quad Layer 不再承载虚拟显示器主画面，专注 Glow、文字信息面板、虚拟键盘等 2D 覆盖元素。
- Quad Layer 与 Projection Layer 不是 fallback 关系；正常运行时 Projection 每帧提交，Quad 覆盖层有 safe 纹理时同帧提交，由 OpenXR compositor 最终合成。
- 房间背景、Glow、墙面反射不得阻塞显示器更新和 `xrEndFrame` 提交。
- 所有高成本效果只消费已经完成的旧结果，不等待当前帧结果。
- OpenXR submit/present、viewer upload、runtime inference、capture 必须分段统计，不混淆性能归因。
- 现有实现只作为迁移参考，不以当前限制倒推目标架构。

非目标：

- 不在第一阶段重写 depth/stereo synthesis 算法。
- 不把本地窗口、RTMP、OpenXR 三种输出合并为一个 frame loop。
- 不把 OpenXR Quad Layer 作为虚拟显示器主画面目标；Virtual Desktop 下主画面以 Projection 3D 平面为基线，Quad 只做覆盖层。
- OpenXR view pose 必须按 runtime 提供的真实位置与朝向参与背景、交互和层合成。

## 2. 当前工程锚点

计划实施时主要涉及以下模块：

| 领域 | 现有模块 | 重构定位 |
|---|---|---|
| OpenXR viewer | `src/xr_viewer/*` | OpenXR frame loop、projection layer、quad layer、环境渲染、controller/overlay |
| OpenXR Quad 层 | `src/xr_viewer/core_quad_layer.py` | 改为 Glow、文字面板、虚拟键盘等 2D 覆盖层，不承载虚拟屏幕主画面 |
| OpenXR runtime eye 上传 | `src/xr_viewer/core_runtime_eye.py` | 提供最新显示器纹理，后续接入独立 screen swapchain |
| OpenXR Projection 层 | `src/xr_viewer/projection_layer_presenter.py`, `screen_layer_presenter.py`, `implementation.py` | 常驻场景层，承载房间、controller、laser、虚拟屏幕主画面、边框和墙面反射 |
| OpenXR 环境 | `src/xr_viewer/environment_renderer.py`, `environment_model.py`, `environment_profiles.py` | 由每帧同步复杂房间渲染迁移到 Projection 内部异步房间背景纹理；panorama/cubemap/equirect 是低成本特例，fallback 仍在 Projection Layer 内绘制 |
| Glow/屏幕光 | `src/xr_viewer/core_screen_quality.py`, `src/xr_viewer/environment_effects.py`, `src/xr_viewer/environment_renderer.py`, `docs/20-openxr-gpu-glow-guide.md` | 以现有 GPU glow/downsample/shader 采样为基础，迁移到异步效果结果池；Glow 输出给 Quad 覆盖层，墙面反射由 Projection 消费 safe 结果；禁止 CPU 采样 |
| Runtime pipeline | `src/stereo_runtime/pipeline.py`, `runtime.py` | 继续产出 latest runtime result，不被 viewer present 反压拖慢 |
| OpenXR 状态 | `src/stereo_runtime/openxr_state.py` | 维护 render/source gate 和 hot config，不承载显示层同步细节 |
| 队列语义 | `src/utils/queue_utils.py` | 保留 latest/overwrite/drop 模型，扩展统计而不是改成阻塞队列 |
| 诊断 | `src/utils/breakdown.py`, runtime debug info | 新增 screen/background/effects/submit/present 分段指标 |

## 3. 目标架构

### 3.1 线程与职责

目标运行时拆成四条职责明确的路径：

| 路径 | 实时等级 | 职责 | 阻塞策略 |
|---|---:|---|---|
| Capture + StereoRuntime | 生产者 | 捕获、depth、stereo synthesis 或 RGB+D | 只写 latest result，队列满则覆盖旧帧 |
| OpenXR Projection Screen Presenter | 硬实时 | 在 Projection Layer 内更新虚拟屏幕 3D 平面并提交 OpenXR frame | 不等待 Quad 覆盖层或 soft effects；拿不到新帧则复用上一帧 |
| Projection Scene Presenter | 硬实时提交、软实时内容 | 提交常驻 Projection Layer，绘制房间 fallback、controller、laser、虚拟屏幕、边框，并采样旧 safe 墙面反射 | 不生成当前帧光效；只消费已完成结果 |
| Background Renderer | 软实时 | 异步渲染复杂 3D 房间到 room background texture，或加载/更新外部 panorama/cubemap/equirect 背景资源 | 只发布 completed safe texture；可低频、可延迟、可暂停 |
| Effects Worker | 软实时 | Glow、屏幕平均色、墙面反射纹理 | 只发布 completed result；Quad/Projection 消费旧 safe result |

### 3.2 OpenXR 层结构

目标 `xrEndFrame` 每帧同时提交屏幕层与场景层。它们不是互相 fallback，而是分工合成：

1. 背景层：优先 equirect/cubemap composition layer；不支持时由 Projection Layer 内天空球/全景背景承担。
2. Projection Layer：常驻主场景层，包含房间/背景 fallback、controller、laser、虚拟屏幕主画面、屏幕边框、墙面反射等。
3. HDR 2D image Quad Layer：海报、预览窗口、静态 HDR 图片等 2D 覆盖内容；默认 tone-map 到 `GL_RGBA8`，仅在 runtime 明确支持 HDR/float swapchain 时启用原生 HDR Quad。
4. Glow Quad Layer：使用 `GL_RGBA8` 等 Virtual Desktop 兼容格式，消费异步 safe glow 纹理，不承载主画面。
5. 可选辅助 Quad/overlay layer：调试 OSD、文字面板、键盘等，按交互需求低频更新和叠加。

显示器画面回到 Projection Layer 内 3D 平面，绕开 Virtual Desktop 对 Quad 主画面格式的限制；房间复杂度仍不得阻塞屏幕纹理更新和 `xrEndFrame`。
Projection Layer 即使没有新的光效结果也必须继续提交，直接复用上一帧 safe light texture 或透明默认纹理；Quad Glow/UI 覆盖层不可用时透明禁用并记录原因，不影响 Projection 主屏。

HDR 内容归属：

```text
HDR panorama / 环境背景        -> Projection / equirect / sky sphere
HDR 2D 图像、海报、预览窗口     -> Quad overlay
Glow / UI / 虚拟键盘           -> Quad overlay
```

### 3.3 数据流

```text
Capture
  -> RuntimePipelineLoop
  -> latest OpenXRRuntimeResult
  -> ScreenFrameBridge
  -> Projection screen texture / 3D screen plane
  -> xrEndFrame

ScreenFrameBridge
  -> AsyncEffectsScheduler
  -> glow_result_pool / light_probe_pool / wall_reflection_pool
  -> Glow Quad consumes latest safe glow
  -> Projection shader consumes latest safe wall light result

Async Projection Background Renderer
  -> room_background_texture ring buffer
  -> latestReadyBgIdx / safe background texture
  -> Projection background pass

External Unity/Blender bake or packaged panorama asset
  -> environment profile
  -> mono/SBS HDR panorama or cubemap cache
  -> Background layer or sky sphere
```

关键约束：

- `ScreenFrameBridge` 可以覆盖旧帧，但不得阻塞 runtime producer。
- `AsyncEffectsScheduler` 不向主 OpenXR frame loop 返回 future/promise 等待点；Projection 只读 safe result。
- 主 OpenXR frame loop 只读 `safe_index`，不等待 GPU fence。
- 初始帧所有 effect result 使用黑色/透明默认纹理。
- Projection 屏幕更新完成后的屏幕 GPU 纹理可以作为异步效果输入，但主线程不得等待该输入生成出的当前帧光效。

### 3.4 目标帧时间模型

实现后的 OpenXR 热路径必须接近以下形态。数字是预算目标，不是固定常量；关键是主图形队列没有 Glow/光斑生成等待点。

```text
主图形队列（每帧）：
├── 采样 safe 房间背景纹理 / panorama / equirect          0.5ms
├── 绘制虚拟屏幕 3D 平面 + controller / laser            1.3ms
├── 采样上帧已完成的墙面光斑纹理并合成                  0.7ms
├── 提交 Glow / 文字面板 / 虚拟键盘 Quad 覆盖层          0.5ms
└── 提交 xrEndFrame
目标总耗时：约 3.0-3.5ms

异步计算队列（与主队列并行）：
├── 接收当前帧屏幕 GPU 纹理
├── 降采样 + 模糊生成新 Glow 纹理                       3.0ms
├── 生成新墙面反射光斑纹理                              1.0ms
└── 原子更新 latest safe effect index
```

实现约束：

- 主图形队列只能读取 `latest_safe_glow()` / `latest_safe_light_probe()`，不能等待当前帧 effect 完成。
- 异步计算队列超预算时只能增加 effect age，不能影响 Projection screen 和 `xrEndFrame`。
- 诊断中必须能看到主图形队列耗时、effect worker 耗时、effect age 三者分离。

### 3.5 Projection 内部异步背景渲染

Projection Layer 内部也按硬实时/软实时拆分。主队列每帧只绘制虚拟屏幕、controller、laser、边框等必须跟随当前帧的前景元素；复杂 3D 房间模型由后台图形队列渲染到离屏 `room_background_texture` 环形缓冲，主队列只采样最近一次已经 safe 的背景纹理。

```text
主图形队列（每帧）：
├── xrWaitFrame / xrBeginFrame
├── 采样 latestReadyBgIdx 指向的 safe room background texture
├── 绘制虚拟屏幕 3D 平面，采样最新 screen texture
├── 绘制 controller / laser / 屏幕边框
├── 提交 Glow / UI / 键盘 / HDR 2D image Quad overlay
└── xrEndFrame

异步背景图形队列（软实时）：
├── 从 2~3 张 room background texture ring buffer 选择非读取 slot
├── 使用最新 head pose 渲染完整 3D 房间模型到离屏纹理
├── GPU 完成后 poll fence/signal
└── 原子更新 latestReadyBgIdx
```

约束：

- 主 OpenXR frame loop 不等待背景 fence/signal，只读取已经发布的 safe 背景纹理。
- 背景 worker 慢、暂停或失败时，主队列继续复用上一张 safe 背景或默认背景。
- 首帧必须有黑色、默认 panorama 或预渲染初始背景，不能读取未初始化纹理。
- 背景更新可按每 2~3 帧、头部旋转超过阈值（如 0.5 度）或 GPU 空闲策略触发。
- 当前目标以 3DoF 原地转头为前提，少量背景 pose 延迟可接受；未来 6DoF 需要 depth reprojection 或更精细的背景重投影。

目标预算模型：

```text
主队列 Frame N:
  sample old room background texture  0.5ms
  draw virtual screen                 1.0ms
  draw controllers / laser            0.3ms
  submit xrEndFrame

异步图形队列:
  render complex room model           6-12ms（隐藏在后台）
  publish latestReadyBgIdx
```

## 4. 分阶段实施路线

### 阶段 0：基线和诊断

目标：先建立可对比、可归因的诊断框架，不再把 Virtual Desktop 受限的 Quad 主屏路径作为普通运行目标。

任务：

- 异步 present 是固定主路径；运行参数只允许限制预算或关闭软效果，不允许把旧同步显示器路径作为常规回退目标：
  - `D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS`：screen texture 上传预算，超预算下一帧复用上一张 screen texture。
  - `D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS`：effect source submit + downsample prewarm 预算，超预算下一帧复用 latest safe effect texture。
  - `D2S_OPENXR_BACKGROUND_UPLOAD_BUDGET_MS`：native panorama/cubemap 背景层上传预算，超预算下一帧复用旧背景或 projection fallback。
- GUI/OpenXR config 不再暴露“回到旧显示器路径”的普通开关。
- 在 `FPSBreakdown` 中新增指标：
  - `openxr_upload`
  - `openxr_quad_update_ms`
  - `openxr_background_ms`
  - `openxr_effect_submit_ms`
  - `openxr_effect_ready_age_frames`
  - `openxr_screen_frame_age_frames`
  - `openxr_wait_frame`
  - `openxr_end_frame`
  - `openxr_layer_count`
- 记录每帧使用的是 `new_screen_frame` 还是 `reused_screen_frame`。
- 当前落地：`FPSBreakdown.validate_openxr_async()` 和日志字段 `openxr_async_ok/missing/failed` 汇总 screen present、Quad 失败、D3D11 PBO readback、effect submit/safe reuse 等硬性验收证据，避免只靠人工解读长日志。

验收：

- 默认启动即进入 Projection screen 主路径；预算参数只影响复用策略，不改变架构路径。
- 打开诊断后能分清 runtime FPS、viewer FPS、screen present FPS、submit/present 节奏。

### 阶段 1：ScreenFrameBridge 与 Projection Screen 主路径

目标：把虚拟显示器固定为 Projection Layer 内 3D 屏幕平面主路径；Quad Layer 只作为 Glow/UI 覆盖层。

任务：

- 新增 `ScreenFrameBridge`：
  - 从 `runtime_q` 非阻塞 drain latest。
  - 维护 `latest_frame`, `last_presented_frame`, `frame_id`, `source_timestamp`。
  - 未收到新 runtime result 时复用上一帧，不阻塞 `xrEndFrame`。
- 重构 Projection screen presenter：
  - Projection Layer 每帧绘制虚拟屏幕 3D 四边形。
  - 屏幕纹理支持 stereo eye source、mono fallback 和 full synthesis eye source。
  - screen plane pose/size 由现有 screen placement 配置驱动。
- 重构 `core_quad_layer.py`：
  - Quad 层只承载 Glow、文字信息面板、虚拟键盘等覆盖纹理。
  - Quad 层不承载虚拟显示器主画面。
  - Quad 层创建失败只禁用对应覆盖层，不影响 Projection 主屏。
- 解耦 controller hit/UI：
  - controller raycast 仍使用同一个 screen plane 几何模型。
  - 交互命中不依赖 projection path 是否绘制屏幕。
  - 屏幕边框/焦点/laser hit 可保留在 projection layer 或独立 overlay layer。
- projection layer 是常驻场景层，承载背景 fallback、controller、laser、虚拟屏幕主体、边框、墙面反射等内容。

验收：

- 房间 mesh 与屏幕主体按 Projection depth 规则工作；是否允许遮挡由 screen plane 深度和渲染顺序明确控制。
- 同一 screen pose 下，鼠标/laser 命中区域与 Projection 视觉屏幕一致。
- 环境模型复杂度提高时，显示器 present FPS 不下降到环境 FPS。
- Quad overlay 不可用时进入明确透明/禁用状态并记录原因，不影响 Projection screen 主体。
- Projection layer 始终提交，用于主屏、房间、手柄、墙面反射等场景内容。

### 阶段 2：OpenXR frame loop 硬实时化

目标：主 OpenXR frame loop 不等待 runtime、背景、效果。

任务：

- 将 OpenXR frame loop 的每帧流程固定为：
  1. `xrWaitFrame/xrBeginFrame`
  2. 非阻塞读取最新 screen frame
  3. 更新 Projection screen texture 或复用上一张
  4. Projection 采样 latest safe background/light result
  5. Quad overlay 采样 latest safe Glow/UI texture
  6. 构造 layers
  7. `xrEndFrame`
- 对 runtime result 的处理拆分：
  - screen upload 必须有预算上限。
  - 超预算时可跳过本帧 upload，复用上一帧 screen texture。
  - effect source submit 使用独立预算；超预算时只能跳过 effect submit，不能影响 screen upload 或 `xrEndFrame`。
- 当前落地：screen upload 和 effect submit 已拆成独立预算和独立计时，effect source submit 不再计入 `openxr_upload` 或 `openxr_submit_frame`，并延后到 `xrEndFrame` 后 flush；`screen_age` / `fx_age` 使用帧数 value 统计，不再伪装成 time metric。
- 当前落地：effect source pending 为 latest-only 单槽；flush 跟不上时覆盖旧 pending 并记录 `openxr_effect_submit_overwrite` / `fx_overwrite`。
- 当前落地：OpenXR frame gate 在 source stale 但已有可渲染 last-good Projection screen frame 时继续进入 renderer；只在没有任何 renderable source frame 时才提交 empty frame。该路径保留 `openxr_no_fresh` 诊断，但不阻塞 controller/head pose 刷新和 `xrEndFrame`。
- 当前落地：`FPSBreakdown.validate_openxr_async()` 允许 `openxr_no_fresh` 与 `screen_proj>0` 同时出现；只要 Projection screen 仍在提交并且 effect submit/safe reuse 条件满足，就不把 stale source 诊断误判为 async 失败。
- 当前落地：`poll_screen_frame()` 只负责非阻塞 drain、上传或复用准备；`openxr_new_screen_frame` / `openxr_reused_screen_frame` / screen age / source latency 只在 Projection 两眼主屏成功渲染后记录，避免 render 失败时提前把帧标记为已呈现。
- 当前落地：当 runtime eye 上传路径决定复用上一张已呈现纹理时，会登记 pending reuse present；这保证 `screen_reuse`、screen age 和 source latency 仍跟 Projection 成功提交闭合，而不会让新 dequeued runtime result 丢失呈现记账。
- 避免在 present 路径中执行：
  - 每帧 CPU tensor readback
  - 每帧模型/环境资源加载
  - 当前帧 Glow 生成
  - 当前帧墙面反射计算
- Projection 绘制只能使用已经完成的 safe Glow/light result；worker 慢或失败时复用旧结果或透明默认纹理。
- 保留 latest/overwrite 队列模型，不引入阻塞 backpressure。

验收：

- 人为让 runtime 降到 30 FPS 时，OpenXR frame loop 仍按头显节奏提交，屏幕复用旧帧但头动/层合成不阻塞。
- 人为让环境渲染耗时升高时，Projection screen texture 仍可按最新可用帧刷新。
- 日志能显示 reused frame 比例、effect age 和 swapchain image wait 耗时。
- 主图形队列在 effect worker sleep/超时时仍保持稳定；effect age 增加但 `xrEndFrame` 不等待。

### 阶段 3：Projection 内部异步背景渲染与 panorama/cubemap 特例

目标：把复杂房间背景从 OpenXR 主帧同步 mesh 渲染迁移为 Projection 内部异步背景纹理；外部工具预生成的 panorama/cubemap/equirect 是同一背景通道的低成本特例。项目不再实现从 GLB 房间自动 bake 出 panorama/cubemap；GLB 仍可作为预览/兼容输入，但目标背景资产由 Unity、Blender 或其它离线工具导出。

任务：

- 新增 `RoomBackgroundResultPool`：
  - 2~3 张 `room_background_texture` 环形缓冲。
  - slot 状态至少包含 `idle`, `writing`, `ready`, `safe`, `reading` 或等价语义。
  - 主队列只读取 `latestReadyBgIdx` 指向的 safe slot，不读 writing slot。
- 新增 `AsyncRoomBackgroundRenderer`：
  - 接收最新 head pose 和 environment profile。
  - 可按 interval、rotation delta 或 GPU budget 触发背景更新。
  - GPU 完成后 poll fence/signal，再发布 safe background index。
  - worker 慢或失败只增加背景 age 或复用旧背景，不影响 Projection screen 与 `xrEndFrame`。
- Projection background pass：
  - 优先采样 safe room background texture。
  - 静态 panorama/equirect/cubemap 可作为 safe background source 的低成本特例。
  - 初始状态使用黑色/默认 panorama/预渲染背景。
- 支持外部导出的背景资产：
  - mono equirectangular HDR/LDR panorama。
  - SBS equirectangular HDR panorama，左右眼按 profile 声明的 stereo layout 分区采样。
  - 后续可选 cubemap/cubemap array 输入，但不要求项目内从 GLB 生成。
- `BackgroundBakeService` 只保留轻量辅助职责：
  - 生成/缓存 wall light mask、profile 派生资源等低成本资产。
  - 不负责 GLB PBR 渲染、不负责 cubemap/equirect panorama bake。
- 新增 `BackgroundLayerRenderer`：
  - 首选 OpenXR equirect/cubemap layer（若 runtime 支持）。
  - fallback 为 projection layer 内 sky sphere shader。
- 将动态 controller、laser、overlay 与静态背景分离。
- 当前落地：panorama shader 支持 `stereo_layout: "sbs"`，左右眼分别采样左/右半幅；`.hdr` panorama 优先上传为半浮点纹理，controller panorama IBL 也按同一 eye 分区采样。
- 当前落地：native panorama/equirect 背景层已记录 safe 背景复用诊断；`bg_safe_age_frames` / `bg_reuse` 可用于确认背景上传或更新慢时只复用旧 safe 背景，不阻塞 Projection 主屏提交。
- 新增诊断字段：
  - `bg_async_render_ms`
  - `bg_safe_age_frames`
  - `bg_ready_overwrite`
  - `bg_reuse`
  - `bg_render_trigger=interval|rotation|manual|startup`
  - `bg_safe_idx`

验收：

- 人为让异步房间背景渲染耗时升高到 6-12ms 时，Projection screen present 不下降，只增加 `bg_safe_age_frames` 或 `bg_reuse`。
- `latestReadyBgIdx` 不指向 writing slot，主队列不会读取正在写入的 room background texture。
- 首帧默认背景可见或明确为黑色，不出现未初始化纹理闪烁。
- head rotation delta 超过阈值时能触发背景更新，未超过阈值时可复用旧背景。
- 外部导出的 mono/SBS panorama 背景按 OpenXR view pose 正确采样和合成，无明显方向错误。
- SBS panorama 在左右眼采样正确，不出现左右眼串图或半幅拉伸。
- environment profile 改变后可重新加载对应 panorama/cubemap 和 mask，并热切换。
- GLB 自动 bake 不是验收项；复杂 GLB 房间若未提供 panorama 资产，只走兼容/预览路径，不作为最终异步背景目标。

### 阶段 4：异步 GPU Glow 结果池

目标：在 `docs/20-openxr-gpu-glow-guide.md` 已实现的 GPU glow 技术基础上，把屏幕光采样、downsample、blur 和混合迁移为可延迟消费的 GPU 结果池。这里不是重新引入 CPU 采样；屏幕颜色来源必须来自 GL texture、低分辨率 glow texture、shader/compute pass 或后续 D3D/Vulkan GPU pass。Glow 结果输出给 Quad 覆盖层；墙面反射/light result 由 Projection Layer 采样。

任务：

- 新增 `AsyncEffectResultPool`：
  - 至少 triple buffer。
  - 每个 slot 状态：`idle`, `writing`, `ready`, `safe`。
  - 主线程只读 `safe_slot`。
- 新增 `GlowWorker` / `GpuGlowScheduler`：
  - 复用现有 `_prepare_glow_downsample_texture()`、glow shader、frosted/surround shader 的 GPU 采样思路。
  - 输入必须是 screen/runtime eye GL texture、shared GPU texture 或低分辨率 GPU 中间纹理。
  - 降采样、多 pass blur、区域采样、边缘采样、高亮提取都在 GPU 内完成。
  - 完成后只更新 ready/safe index，不唤醒主线程等待。
  - 禁止通过 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 等 CPU readback 做实时屏幕采样。
- 更新 environment/screen shader：
  - 继续沿用 `20-openxr-gpu-glow-guide.md` 的低分辨率 GPU glow texture、`textureLod()`、边缘采样、区域化采样和 frosted/surround shader 方案。
  - Quad overlay 使用 `safe_glow_texture` 叠加 screen glow / surround glow / frosted glow。
  - 初始或 worker 失败时使用透明纹理。
- 先以 OpenGL FBO/低频 pass 实现逻辑正确性；后续按平台升级到 D3D11 compute、D3D12/Vulkan async compute queue。
- 当前落地：runtime effect source 已抽出轻量 result pool，具备 staging/ready/safe/spare texture 语义和 safe publish 诊断；ready 被 latest-only 新结果覆盖时旧 ready 会转为 spare 复用或释放，避免异步 submit 跟不上时泄漏纹理。
- 当前落地：`EffectWorker` 已从 submitter/source state 中拆出，`xrEndFrame` 后按 `D2S_OPENXR_EFFECT_WORKER_INTERVAL` 低频预热 glow/light downsample，并只向 scheduler 发布 safe downsample；panorama/controller/glow 消费者只读取 safe result，不在 render 消费路径生成 downsample；worker 异常会记录并禁用后续 soft effect 预热，不影响 Projection screen 和 OpenXR submit。
- 当前落地：GPU glow/downsample 复用已有 cache key，新增 `fx_ds_render` / `fx_ds_reuse` 诊断，用于确认 soft effect 是否复用旧结果；no-room 与 environment glow pass 都隔离异常并恢复 GL 状态，effect 失败不污染 screen submit 后续帧。
- 当前落地：screen light source 同帧缓存已接入，环境光、panorama、controller 反射等多消费者复用同一个 prepared light texture，并记录 `light_reuse`。
- 当前落地：`EffectScheduler` 已把 full safe source/downsample 与 safe light probe 分离；屏幕光消费者只读取同 frame id 的 3x3 低频 `safe_light_probe`，不再把 full safe source 或 glow downsample 隐式当作 light probe。
- 当前落地：screen effect source 同帧缓存已接入，Glow/veil/shell 多 pass 复用同一个 safe source lookup，并记录 `fx_source_reuse`。
- 当前落地：`fx_age` 按每帧每个 safe effect texture 去重记录，避免 Glow、screen light、panorama 等多消费者重复放大 age 样本。

验收：

- Glow worker 降频到 30/45 Hz 时，OpenXR screen present 不受影响。
- Glow 纹理 age 可见但不卡顿。
- GPU Glow Off fast path 不触发 downsample，不触发 CPU 采样。
- worker 故障只关闭 Glow，不影响 Projection screen 和 OpenXR submit。

### 阶段 5：墙面反射和屏幕光斑

目标：实现 `35` 中的“预计算掩码 + 动态光斑颜色”路径。墙面反射属于 Projection Layer 场景合成；Projection screen texture 同时作为异步效果输入源。

任务：

- 为 environment profile 增加可选资源：
  - `wall_light_mask`：与 panorama UV 对齐的单通道或 RGB 权重纹理。
  - `screen_light_layout`：屏幕到墙面的投影区域描述。
- 新增 `LightProbeWorker` / `GpuLightProbeScheduler`：
  - 从 screen/runtime eye GPU texture 生成 1x1、3x3、8x5 或 16x9 低频区域色。
  - 必须复用 GPU Glow 降采样链或专用 GPU pass，不允许 CPU 采样。
  - 输出 `screen_light_color_grid` 或低分辨率 GPU light texture。
- Projection/background shader 合成：
  - `final = panorama + mask * delayed_screen_light_color * intensity`
  - 所有输入都来自 safe result。
- 对简单房间提供数学 mask fallback；对复杂房间使用外部工具或 profile 提供的 `wall_light_mask`，项目内只提供 `wall_light_mask: "auto"` 的基础 UV mask 生成。
- 当前落地：panorama shader 已可用数学 mask fallback 或 profile `wall_light_mask` 图片消费 latest safe light probe texture，并使用异步 worker 发布的 3x3 GPU light probe 采样低频屏幕光；复杂房间 mask 由外部工具或 profile 资产提供，项目内不实现复杂房间 bake。
- 当前落地：`BackgroundBakeService` 已支持 profile `wall_light_mask: "auto"`，在配置期生成并缓存与 panorama UV 对齐的灰度 mask PNG；实时渲染路径只加载缓存纹理，不做 CPU 屏幕采样。它不是 GLB panorama/cubemap bake 服务。

验收：

- 屏幕画面颜色变化能延迟一两帧反映到墙面，不影响 screen FPS。
- 没有 mask 时自动退化为现有 glow-only 视觉。
- mask/profile 资源缺失有明确日志，不报错退出。

### 阶段 6：GPU 后端升级路径

目标：在不打断 Python/OpenXR 主流程的情况下，引入真正多队列能力。

分层策略：

1. OpenGL baseline：FBO + low-frequency pass + no-wait safe result，先验证架构语义。
2. D3D11 native：复用现有 D3D11 backend，减少 GL/D3D 互操作成本，支持更稳定的 Windows OpenXR 路径。
3. D3D12/Vulkan 后端：为 async compute/copy queue 建立接口，不把具体 API 绑死在 viewer 业务代码里。

新增接口建议：

```text
EffectScheduler
  submit_screen_frame(frame_handle, frame_id)
  poll_completed()
  latest_safe_glow()
  latest_safe_light_probe()
  release()

ScreenLayerPresenter
  update_or_reuse(frame)
  make_quad_layers(predicted_display_time)
  release()

BackgroundPresenter
  make_background_layer_or_projection()
  update_safe_effect_inputs(effect_state)
  latest_safe_background()
  release()

AsyncRoomBackgroundRenderer
  submit_pose(environment_profile, head_pose, trigger)
  poll_completed()
  latest_safe_background()
  release()
```

验收：

- 后端替换不改变 OpenXR 主 loop 的 no-wait 规则。
- GL/D3D11/D3D12/Vulkan 任一后端失败，都能回退到透明 effect + projection fallback。
- 旧 EXT memory / `render.py` / `d3d11_backend.py` 路径不再参与 OpenXR 热路径；D3D11 projection 不回退 PBO readback，runtime texture upload 的 PBO 仅作为明确告警的 GPU fallback。

## 5. 关键设计约束

### 5.1 主线程禁止等待 effect

禁止在 OpenXR frame loop 中出现以下等待：

- 等待 Glow 完成。
- 等待墙面反射完成。
- 等待外部 panorama/cubemap 资产或派生 mask 准备完成。
- 等待当前 runtime result 必须可用。

可接受行为：

- 复用上一帧 screen texture。
- 使用黑色/透明 effect texture。
- 使用旧 panorama。
- 暂停背景更新但继续提交 Projection screen。

### 5.2 Runtime 与 viewer 只通过 latest result 连接

`RuntimePipelineLoop` 不应被 OpenXR present 速度反压。OpenXR viewer 也不应等待 runtime。两边通过 latest result 桥接：

- producer 快：覆盖旧 result，统计 overwrite/drop。
- producer 慢：viewer 复用旧 screen texture，统计 reuse。
- producer 停：viewer 保持 last good frame 或进入明确 wait/idle 状态。

### 5.3 屏幕交互与屏幕显示同源

显示器视觉和交互都使用同一个 Projection screen plane：

- raycast 使用 screen pose/size。
- UI hover/click 使用 logical UV。
- Projection layer 可绘制边框、hover highlight、controller laser。
- Quad overlay 不参与主屏命中，只用于 Glow/UI/键盘覆盖。

### 5.4 主路径与降级边界

| 功能 | 主路径 | 降级边界 |
|---|---|---|
| 显示器 | Projection screen 3D 平面 | 复用上一帧 screen texture 或明确 wait/idle，不依赖 Quad 主屏 |
| 场景/手柄/墙面反射 | 常驻 Projection layer | 光效透明/复用旧结果；Projection 本身不因 Quad overlay 成功而跳过 |
| Glow/UI/键盘 | Quad overlay layer | 透明/禁用并记录原因，不影响 Projection screen |
| 背景 | equirect/cubemap layer | Projection sky sphere |
| Glow | async safe texture，由 Quad overlay 消费 | transparent/no glow |
| 墙面反射 | mask + delayed light texture，由 Projection 消费 | disabled |
| GPU 后端 | D3D11/D3D12/Vulkan async | OpenGL low-frequency pass |

## 6. 测试计划

### 6.1 单元测试

- `ScreenFrameBridge`：latest drain、reuse、frame id、timestamp、drop 统计。
- `AsyncEffectResultPool`：slot 状态转换、safe index 不读 writing slot。
- GPU Glow 采样约束：测试文档/代码不得出现实时 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 作为屏幕光颜色来源。
- `QuadOverlayPresenter`：Glow/UI/键盘 layer 构造、pose/size、失败原因。
- `BackgroundBakeService`：wall light mask cache key、profile invalidation、missing asset fallback；不测试 GLB panorama/cubemap bake。
- `RoomBackgroundResultPool`：safe index 不读 writing slot，background worker 慢/失败时复用旧 safe slot。
- `AsyncRoomBackgroundRenderer`：interval、rotation delta、manual/startup trigger 语义。

### 6.2 集成测试

- OpenXR 无 headset preview 模式不崩溃。
- Quad overlay 创建失败时仍绘制 Projection screen 主体，并记录明确失败原因。
- Quad overlay 成功时仍提交 Projection layer，用于主屏、controller、房间、墙面反射等场景内容。
- runtime producer 低 FPS 时 viewer submit 不阻塞。
- effect worker 人为 sleep 时 screen FPS 不下降。
- GPU Glow Off fast path 不触发 downsample，不触发 CPU 采样。
- 外部 panorama 背景、异步复杂房间背景或复杂 GLB 兼容预览路径启用时，Projection screen 刷新不被背景路径拖慢。
- 异步房间背景 worker 人为 sleep 时，screen FPS 不下降，只增加 `bg_safe_age_frames` / `bg_reuse`。

### 6.3 性能验收

至少记录：

| 指标 | 目标 |
|---|---:|
| screen present FPS | 接近头显目标刷新率 |
| runtime FPS | 独立记录，不作为 present FPS |
| screen frame reuse rate | 可解释、可接受 |
| effect age | 通常 1-3 帧 |
| OpenXR submit/present stall | 明确归因到 runtime/headset，不归因到 StereoRuntime |
| CPU fallback count | OpenXR 实时路径必须红色告警；屏幕光/Glow 实时采样不得 CPU fallback |
| bg_async_render_ms | 背景 worker 独立耗时，不进入主 frame submit 预算 |
| bg_safe_age_frames / bg_reuse | 背景慢时只增长 age/reuse，不拖慢 screen present |

## 7. 推荐提交顺序

1. `feat(openxr): add async rendering flags and diagnostics`
2. `feat(openxr): introduce screen frame bridge`
3. `feat(openxr): stabilize projection screen presenter`
4. `feat(openxr): decouple frame loop from runtime producer`
5. `feat(openxr): add panorama background presenter`
6. `feat(openxr): add async glow result pool`
7. `feat(openxr): add wall reflection light probe path`
8. `refactor(openxr): split backend effect scheduler interfaces`
9. `docs(openxr): update runtime architecture and degradation matrix`

每个提交都必须保持 OpenXR 可诊断、可启动；显示器主路径固定为 Projection screen，Quad 只做覆盖层。

## 8. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| Quad overlay 与 Projection 场景层视觉不一致 | Glow/UI 错位 | screen pose/size 单一来源，Quad overlay 和 Projection 边框/光效共用参数 |
| OpenXR runtime 对 Quad/equirect 支持差异 | 某些设备 overlay 或背景层失败 | 能力探测 + 明确失败原因；背景可 projection fallback，显示器仍走 Projection screen 主体 |
| Python/OpenGL 难以实现真正 GPU 多队列 | 无法达到最终并行度 | 先实现 no-wait 语义，再引入 D3D11/D3D12/Vulkan 后端 |
| effect 结果读写冲突 | 闪烁或 GPU hazard | triple buffer + safe index，不读 writing slot |
| 诊断口径混乱 | 性能问题误判 | 强制分开 capture/runtime/viewer/screen/effect/submit/present |

## 9. 第一轮落地建议

第一轮不要同时做 panorama、Glow、墙面反射。建议先完成硬实时显示器链路：

1. 加 flags 和分段统计。
2. 做 `ScreenFrameBridge`。
3. 稳定 Projection screen presenter。
4. 保持 Projection Layer 常驻，背景/手柄/边框仍走现有 projection/environment 渲染。
5. 保持 Glow Off / GPU Glow fast path，不引入 CPU 屏幕采样。
6. 验证复杂环境下 screen FPS 与环境成本解耦。

只有当“Projection screen 不被环境拖慢”成立后，再进入异步 Glow 和 panorama 背景，否则会把多个问题混在一起，无法判断收益来源。

## 10. 完成定义

重构完成的判定不是“代码里有 Quad layer 或 async worker”，而是满足以下行为：

- 显示器画面作为 Projection Layer 内 3D 屏幕平面稳定提交。
- Projection Layer 与可用 Quad overlay 同帧提交；Projection 负责主屏、场景、手柄、边框、墙面反射，Quad 负责 Glow/UI/键盘覆盖。
- 背景复杂度变化不改变显示器硬实时路径；复杂 3D 房间渲染只通过 safe room background texture 被主队列消费。
- Glow 和墙面反射都可延迟、可降频、可失败关闭，但屏幕采样必须走 GPU glow/downsample/shader 路径，不能阻塞 screen submit，不能回退到 CPU 实时采样。
- Runtime producer 与 OpenXR presenter 之间没有阻塞等待。
- 日志能清楚解释每一帧的 runtime、viewer、screen、effect、submit/present 状态。
