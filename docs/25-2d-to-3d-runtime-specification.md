# Desktop2Stereo 2D-to-3D Parallax Budget Specification

日期：2026-06-25

> 作废声明：本文已作废，仅作为历史规范记录保留。Desktop2Stereo 当前正式最终运行时流程规范见 `docs/28-Realtime-2d-to-3d-specification.md`；若本文与 docs/28 存在差异，以 docs/28 为准。

本文记录 Desktop2Stereo 新版 2D 转 3D 合成路径的规范决定。目标是把立体强度从经验型 `IPD * stereo_scale * depth_strength * max_shift_ratio` 多重乘法，收敛为清晰、可解释、可复现的视差预算模型，并规定 RGB、depth、mask、补洞、OpenXR 输出在同一 render size 坐标系中的处理路径。

## 结论

Desktop2Stereo 的单目深度模型输出按 normalized / relative depth 处理，不是真实米制深度。因此 runtime 合成路径不得把它当成物理 `Z` 代入双目相机公式。

最终采用：

```text
disparity_px = depth_response(depth, convergence) * max_disparity_px

left_shift_px  = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

其中：

```text
depth_response(depth, convergence): normalized depth 到相对视差权重的映射
max_disparity_px: 左右眼总水平视差预算，单位 pixel
left_shift_px / right_shift_px: 每眼实际位移，分别取总视差的一半
```

`max_disparity_px` 是 normalized-depth 路径的核心立体强度参数。GUI 和 API 的立体强度语义必须收敛到 `max_disparity_px` 或对应的 Parallax Budget 档位。`IPD` 只属于真实 metric depth 路径，不属于 normalized-depth 强度控制。

## 规范处理路径

normalized-depth 2D 转 3D runtime 必须按下面的阶段处理。所有阶段必须显式区分 `capture_size` 和 `render_size`。

```text
1. Capture Input
   输入：显示器 / 窗口捕捉得到的 RGB frame
   坐标系：capture_size
   输出：source_rgb

2. Resolve Render Size
   输入：capture_size、OpenXR render scale、质量档、输出模式
   坐标系：无图像处理，只解析目标尺寸
   输出：render_size

3. Resize RGB To Render Size
   输入：source_rgb
   坐标系：capture_size -> render_size
   输出：render_rgb

4. Depth Estimation
   输入：render_rgb 或由 depth provider 自己预处理后的 render_rgb
   坐标系：depth provider 内部尺寸 -> render_size
   输出：depth_render，必须对齐 render_size

5. Depth Postprocess
   输入：depth_render
   坐标系：render_size
   输出：depth_response_input

6. Resolve Parallax Budget
   输入：render_width、render_height、strength_preset、aspect
   坐标系：render_size
   输出：max_disparity_px

7. Disparity Field
   输入：depth_response_input、convergence、max_disparity_px
   坐标系：render_size
   输出：disparity_px / shift_px

8. Stereo Warp
   输入：render_rgb、disparity_px
   坐标系：render_size
   输出：left_eye、right_eye、raw occlusion / disocclusion areas

9. Mask And Hole Fill
   输入：left_eye、right_eye、depth_render、disparity_px、occlusion mask
   坐标系：render_size
   输出：filled_left_eye、filled_right_eye

10. Temporal Stabilization
    输入：filled eyes、mask、temporal state
    坐标系：render_size
    输出：stable_left_eye、stable_right_eye

11. Output Pack / Viewer Upload
    非 OpenXR：按 Display Mode 打包 mono / half_sbs / full_sbs / half_tab / full_tab / anaglyph / interleaved / leia / depth_map 等
    OpenXR full synthesis：直接上传 left_eye / right_eye
    OpenXR RGB+depth direct：上传 source/render RGB + depth，由 viewer shader 按规范参数生成左右眼
```

核心约束：

```text
进入 stereo synthesis 之后，render_size 是唯一工作坐标系。
RGB、depth、disparity、mask、hole fill、temporal、left/right eye 必须全部对齐 render_size。
```

禁止的混用方式：

```text
RGB 是 2K，depth 仍是 4K。
depth 是 2K，disparity_px 按 4K 预算计算。
mask 在 4K 生成，hole fill 在 2K 执行。
left/right eye 是 2K，但 debug 或 viewer 按 4K eye size 建屏。
OpenXR 下采样后仍沿用 capture_size 的 max_disparity_px。
```

## 工程分层定义

完整工程规范必须把运行目标、质量/算法模式、合成方式、render size、封装格式、传输方式分开。它们是不同层级，不能互相替代。

```text
Capture Source
-> Application Runtime Target
-> Runtime Quality Mode
-> Stereo Synthesis Mode
-> Render Size / 4K Tier
-> Output Transport
-> Output Packing Format
-> Viewer / Device Presentation
```

核心原则：

```text
OpenXR 不是显示封装格式。
half_sbs / full_sbs / TAB 不是运行模式。
网络推流不是立体算法。
3D 显示器不是 capture source。
```

## Capture Source

Capture Source 定义输入从哪里来，只负责提供 RGB frame 和源 metadata。

| Source | 说明 | 输出 |
|---|---|---|
| monitor_capture | 捕捉完整显示器 | source_rgb + capture_size + monitor metadata |
| window_capture | 捕捉指定窗口 | source_rgb + capture_size + window metadata |
| file_image | 单张图片 | source_rgb + capture_size + file metadata |
| file_video | 视频帧 | source_rgb + capture_size + timestamp |
| api_frame | 外部 API 传入帧 | source_rgb + capture_size + caller metadata |

Capture Source 不决定视差公式，不决定 OpenXR 路径，也不决定 SBS/TAB 封装。

## Application Runtime Target

Application Runtime Target 定义程序最终面向的输出目标。

| Target | 用途 | 默认 transport | 典型 packing |
|---|---|---|---|
| local_display | 本地窗口或本地 3D 显示器 | local_window / local_fullscreen | mono / half_sbs / full_sbs / half_tab / full_tab / anaglyph / interleaved / leia |
| network_stream | 网络推流到远端播放器或设备 | encoded_stream | half_sbs / full_sbs / half_tab / full_tab / anaglyph，未来可扩展 left/right 双流 |
| openxr | VR / XR 头显 | openxr_swapchain | none；full synthesis 直接传 left/right eye，RGB+depth direct 传 RGB+depth |
| debug_export | 调试或离线导出 | file_export | left/right/depth/mask/shift/sbs/tensors/metadata |
| headless_api | 无 GUI、批处理或服务调用 | api_result / file_export | 由调用方指定 |
| auto | 由 GUI/设备选择自动映射 | resolved target | resolved packing |

`Application Runtime Target` 不得和 Runtime Quality Mode 混用。`auto/movie/game/image/debug` 这类场景偏好应归入 Runtime Quality Mode，不能替代最终输出目标。

## Runtime Quality Mode

Runtime Quality Mode 定义当前场景对延迟、质量、稳定性的偏好。

| Mode | 语义 | 典型策略 |
|---|---|---|
| auto | 根据 target、设备、GPU 压力自动选择 | 自动选择 render scale、backend、hole fill、temporal |
| movie / cinema | 影院观影，质量优先但仍需实时 | full synthesis eyes、balanced hole fill、temporal on |
| game / game_low_latency | 低延迟优先 | 降低 render scale、减少 temporal 或使用 direct path |
| image / still_image_hq | 静态图像或截图高质量 | 高质量 hole fill、允许更慢合成 |
| debug / debug_export | 调试导出 | 输出 depth/mask/shift/timing/provider metadata |

Quality Mode 可以影响 render scale、hole fill、temporal、backend 选择，但不能改变公式语义。

## Stereo Synthesis Mode

Stereo Synthesis Mode 定义左右眼如何生成。

| Mode | 输入 | 输出 | 适用场景 |
|---|---|---|---|
| rgb_depth_direct | source/render RGB + depth | 由 viewer shader 现场生成左右眼 | OpenXR direct / 低延迟 shader 路径 |
| full_synthesis_eyes | render_rgb + depth_render | left_eye + right_eye + depth | OpenXR full synthesis、本地高质量、静态图像、debug |
| packed_synthesis | left_eye + right_eye | packed frame | 本地显示器、网络推流、录制 |

规则：

```text
rgb_depth_direct 不消费 full synthesis 的 mask / hole fill 结果。
full_synthesis_eyes 不应在 viewer 里重新用 RGB+depth shader 生成左右眼。
packed_synthesis 只能发生在 left/right eye 已经生成之后。
```

## Render Size / 4K Tier

Render Size / 4K Tier 定义实际合成尺寸。它直接影响 `max_disparity_px`、GPU 计算量、OpenXR 上传量和编码带宽。

Render Size / 4K Tier 不向用户暴露 `native` / `fixed` / `dynamic` 策略选择。用户侧只暴露 `stereo_render_scale` / `Render Scale` 作为 4K 级输入的稳定档位选择信号。

| 输入条件 | 规则 | 用途 |
|---|---|---|
| 非 4K 级输入 | `render_size = capture_size` | 避免对普通 1080p/2K 输入做连续缩放，保持预算语义稳定 |
| 4K 级横屏输入 | 按 `stereo_render_scale` 映射到 3840x2160、3200x1800、2560x1440、1920x1080 稳定档位 | OpenXR 降上传量、网络降码率、本地稳帧率 |
| 4K 级竖屏输入 | 按同一档位映射到 2160x3840、1800x3200、1440x2560、1080x1920 | 竖屏截图/窗口保持方向，预算按 render_size 短边解析 |

规则：

```text
max_disparity_px 必须按 render_size 解析。
stereo_render_scale 不是任意输入尺寸的连续缩放比例，而是 4K 级输入的稳定档位选择信号。
4K 级判断必须方向无关，并覆盖常见全屏和近 4K 窗口：3840x2160、2160x3840、4096x2160、3840x2400、3840x1600 都属于 4K 级；2560x1440、3440x1440、1080x1920、1000x3000 不属于 4K 级。
render_size 变化后，depth、mask、temporal state、OpenXR eye texture 都要按新尺寸重建或重置。
动态分辨率稳帧率如果后续加入，也只能跨稳定档位切换；不能每帧连续改变预算。
```

## Output Transport

Output Transport 定义结果送到哪里。

| Transport | 输入 | 输出目标 | 说明 |
|---|---|---|---|
| local_window | packed frame 或 preview frame | 本机窗口 | GUI 预览或普通窗口播放 |
| local_fullscreen | packed frame | 本地 3D 显示器 | 需要匹配显示器支持的 SBS/TAB/interleaved 等格式 |
| encoded_stream | packed frame | 网络接收端 | 编码器可能要求 NV12/BGRA、固定分辨率、固定 FPS |
| openxr_swapchain | left/right eye 或 RGB+depth direct result | OpenXR viewer | full synthesis 上传双眼纹理；direct path 上传 RGB+depth |
| file_export | tensors/images/video/metadata | 文件系统 | debug、评估、回归测试 |
| api_result | structured runtime result | 调用方 | headless/API 模式 |

Transport 不应改变视差公式。编码器或 viewer 的缩放如果不可避免，必须在 metadata 中记录，并避免再次改变 `max_disparity_px` 语义。

## Output Packing Format

Output Packing Format 只定义 left/right eye 如何封装成一帧。它不定义运行模式，不定义合成算法。

| Format | 定义 | packed frame 尺寸 |
|---|---|---|
| mono | 单眼或 2D preview | eye_width x eye_height |
| half_sbs | 左右水平并排，每眼横向压缩 1/2 | eye_width x eye_height |
| full_sbs | 左右水平并排，每眼保留完整宽度 | 2 * eye_width x eye_height |
| half_tab | 左右上下排列，每眼纵向压缩 1/2 | eye_width x eye_height |
| full_tab | 左右上下排列，每眼保留完整高度 | eye_width x 2 * eye_height |
| anaglyph | 用颜色通道编码左右眼 | eye_width x eye_height |
| interleaved | 行/列/像素交错输出 | 依目标显示器协议决定 |
| leia | Leia / lightfield 类显示输出 | 依设备协议决定 |
| depth_map | depth 可视化或灰度深度输出 | render_width x render_height |

命名规则：

```text
统一使用 half_sbs，不使用 half_sb。
统一使用 full_sbs，不使用 full_sb。
TAB 表示 top-and-bottom；内部 key 使用 half_tab / full_tab。
```

OpenXR full synthesis 不使用 SBS/TAB 作为 viewer 输入。OpenXR debug 可以额外导出 SBS/TAB 预览，但那只是 debug artifact。

## Presentation Flags

Presentation Flags 是输出展示层修正，不应改变合成公式。

| Flag | 说明 | 作用层级 |
|---|---|---|
| eye_order | left_first / right_first | packing / display / stream |
| cross_eyed | 交换左右眼 | presentation correction |
| aspect_policy | preserve_aspect / stretch / crop / letterbox | resize / presentation |
| color_format | RGB / BGR / BGRA / NV12 | capture / encoder / upload |
| dtype | uint8 / float16 / float32 | tensor / upload / encoder |
| rgb_range | 0..1 / 0..255 | tensor contract |
| depth_range | near/far 方向和归一化范围 | depth contract |
| latency_policy | quality / balanced / low_latency | quality mode helper |
| debug_outputs | depth / mask / shift_px / timing / provider_info | debug/export |

特别注意：

```text
cross_eyed 只交换左右眼，不改变 disparity 公式。
eye_order 只影响封装或设备协议，不改变 left/right 生成方式。
aspect_policy 如果产生 crop/letterbox，必须在 render_size 解析前明确，否则 depth 和 RGB 会错位。
```

## 模式组合矩阵

| Application Target | Runtime Quality Mode | Synthesis Mode | Render Size / 4K Tier | Transport | Packing Format |
|---|---|---|---|---|---|
| local_display | movie / game / image / debug | full_synthesis_eyes -> packed_synthesis | 非 4K 保持 capture_size；4K 输入按 Render Scale 稳定档位 | local_window / local_fullscreen | mono / half_sbs / full_sbs / half_tab / full_tab / anaglyph / interleaved / leia |
| network_stream | movie / game | full_synthesis_eyes -> packed_synthesis | 非 4K 保持 capture_size；4K 输入按 Render Scale 稳定档位，编码器再按 transport profile 处理 | encoded_stream | half_sbs / full_sbs / half_tab / full_tab / anaglyph |
| openxr_rgb_depth_direct | game / auto | rgb_depth_direct | 非 4K 保持 capture_size；4K 输入按 OpenXR Render Scale 稳定档位 | openxr_swapchain | none |
| openxr full synthesis | movie / game / image / debug | full_synthesis_eyes | 非 4K 保持 capture_size；4K 输入按 OpenXR Render Scale 稳定档位 | openxr_swapchain | none |
| debug_export | debug / image | full_synthesis_eyes | 默认保持 capture_size；离线导出尺寸由调用方显式指定时必须记录 metadata | file_export | left/right/depth/mask/shift/sbs/tensors/metadata |
| headless_api | auto / movie / image / debug | caller selected | caller selected, but must keep render_size and budget metadata explicit | api_result / file_export | caller selected |

工程判断顺序：

```text
1. 先确定 Application Runtime Target。
2. 再确定 Runtime Quality Mode。
3. 根据 target + quality mode 选择 Stereo Synthesis Mode。
4. 根据 capture_size 和 Render Scale 解析 Render Size / 4K Tier。
5. 在 render_size 坐标系完成 depth、disparity、warp、mask、hole fill、temporal。
6. 最后根据 Transport 决定直接上传、封装、编码或导出。
```

## Runtime 参数热更新规范

GUI、本地显示、网络推流和 OpenXR 都必须共享同一套参数语义。GUI 不应直接修改 runtime 内部对象，OpenXR viewer 也不应自己猜测立体参数含义。

核心规则：

```text
所有用户可调立体参数都必须先进入 RuntimeSettingsSnapshot。
所有设置变化只允许在帧边界应用。
所有变化必须按 Hot Reload / Pipeline Rebuild / Session Restart 分级。
OpenXR 和本地模式共享同一参数语义，只允许输出层消费方式不同。
```

### RuntimeSettingsSnapshot

GUI 或 API 只生成不可变配置快照，并通过线程安全队列发送给 runtime。

```text
GUI / API
-> RuntimeSettingsSnapshot(version=N)
-> settings_update_q
-> RuntimePipelineLoop frame boundary
-> active_settings
```

必需字段：

```text
version
timestamp
source
application_runtime_target
runtime_quality_mode
stereo_synthesis_mode
stereo_render_scale
parallax_budget_preset
max_disparity_px
depth_response
convergence
hole_fill_mode
edge_threshold
edge_dilation
mask_feather_radius
temporal_enabled
temporal_strength
presentation_flags
debug_flags
```

禁止方式：

```text
GUI 直接修改 StereoRuntime.stereo_config。
GUI 直接修改 OpenXRViewer.depth_strength / ipd。
多个线程共享可变 settings dict。
viewer 直接读取 GUI settings。
```

### 热更新应用时机

runtime 只在帧边界应用新设置，保证同一帧内不会半旧半新。

```text
while running:
    drain settings_update_q
    merge latest RuntimeSettingsSnapshot
    classify changed fields
    apply hot reload fields
    reset temporal if required
    rebuild resources if required
    process next frame
```

每次 runtime result 应记录：

```text
active_settings_version
hot_reload_changed_fields
hot_reload_class
```

这样调试时可以区分：GUI 没发配置、runtime 没应用配置、viewer 没消费配置，还是参数已生效但视觉变化不明显。

### 参数变更分级

| 参数 | Hot Reload | Reset Temporal | Rebuild Resources | 说明 |
|---|---:|---:|---:|---|
| `max_disparity_px` / parallax preset | 是 | 可选 | 否 | 强度突变时可 reset temporal |
| `convergence` | 是 | 可选 | 否 | 大幅变化建议 reset temporal |
| `depth_response` | 是 | 是 | 否 | 曲线变化会改变全局视差分布 |
| `hole_fill_mode` | 是 | 否 | 否 | quality 模式可能显著影响性能 |
| `edge_threshold` / `edge_dilation` | 是 | 否 | 否 | mask 参数 |
| `mask_feather_radius` | 是 | 否 | 否 | mask 边缘柔化 |
| `temporal_strength` | 是 | 否 | 否 | 0 可表示关闭效果 |
| `temporal_enabled` | 是 | 是 | 否 | 开关变化需要清理历史状态 |
| `cross_eyed` | 是 | 否 | 否 | presentation 层交换左右眼 |
| `anaglyph_method` | 是 | 否 | 否 | packing 层 |
| `output_packing_format` | 部分 | 否 | 可能 | 本地/推流可能需要重建输出缓冲 |
| `stereo_render_scale` | 否 | 是 | 是 | 仅当 4K 档位变化并导致 render_size 变化时重建 |
| `stereo_synthesis_mode` | 否 | 是 | 是 | direct/full synthesis 切换 |
| `depth_backend` | 否 | 是 | 是 | provider/engine 变化 |
| `capture_source` / `capture_target` | 否 | 是 | 是 | 重新捕捉和重建 source metadata |
| `openxr_swapchain_format` | 否 | 是 | 是/重启 | OpenXR session 资源 |
| `environment_model` | 否 | 否 | 可能/重启 | viewer 场景资源 |
| `encoder_profile` | 否 | 否 | 是 | 网络推流编码器资源 |

分级含义：

```text
Hot Reload: 下一帧开始生效，不重建 pipeline。
Pipeline Rebuild: 需要重建 render buffers、provider、packer、encoder 或 OpenXR upload resources。
Session Restart: 需要重启 capture session、OpenXR session 或 viewer。
```

### OpenXR 热更新规则

OpenXR full synthesis 和 RGB+depth direct 的消费方式不同，但参数来源必须相同。

```text
openxr_full_synthesis_eyes:
    runtime 消费 RuntimeSettingsSnapshot。
    runtime 重新合成 left_eye / right_eye。
    viewer 只上传 runtime result，不解释 parallax 参数。
```

```text
openxr_rgb_depth:
    runtime 输出 RGB + depth。
    viewer shader 消费 RuntimeSettingsSnapshot 派生出的 shader uniform snapshot。
    shader uniform snapshot 必须表达 max_disparity_px、depth_response、convergence、render_size、screen_roll 等规范语义。
    viewer 不得把 IPD、depth_strength、stereo_scale、max_shift_ratio 当作 normalized-depth 强度链重新解释。
```

### 本地显示和网络推流热更新规则

本地显示和网络推流都应先更新 synthesis，再按目标输出重新 pack 或 encode。

```text
local_display:
    RuntimeSettingsSnapshot -> left/right eye -> packing format -> local window/fullscreen
```

```text
network_stream:
    RuntimeSettingsSnapshot -> left/right eye -> packing format -> encoder -> stream
```

packing format 改变不应改变 synthesis 公式：

```text
half_sbs -> full_sbs:
    不需要重新计算 depth。
    不需要重新计算 disparity。
    只需要重新 pack；如果输出缓冲/编码器尺寸变化，则重建 transport resources。
```

### 线程安全规范

```text
GUI thread:
    只写 settings_update_q。

Runtime thread:
    只在帧边界 drain queue。
    只持有 immutable active_settings snapshot。

OpenXR viewer thread:
    full synthesis 只读 runtime result。
    direct path 只读 viewer_config_snapshot / shader_uniform_snapshot。

Debug/log:
    每次变更打印 changed fields、version、reload_class。
```

配置快照必须不可变，推荐形态：

```text
RuntimeSettingsSnapshot(frozen=True)
```

### 热更新 debug 字段

runtime result / OpenXR result / debug export 中必须记录：

```text
active_settings_version
hot_reload_changed_fields
hot_reload_class
application_runtime_target
runtime_quality_mode
stereo_synthesis_mode
render_size
max_disparity_px
depth_response
convergence
hole_fill_mode
runtime_output_format
packing_format
transport
provider_info
timing
```

这些字段用于确认 GUI 实时调参是否真正进入 OpenXR、本地显示或网络推流路径。

## 坐标系和数据契约

### capture_size

`capture_size` 是原始输入源尺寸。它来自显示器、窗口、视频帧或截图。

用途：

```text
记录源尺寸
决定默认 render scale 的上限
用于 debug / report / source metadata
```

它不应直接决定最终视差预算，除非 `render_size == capture_size`。

### render_size

`render_size` 是实际 stereo synthesis 和 OpenXR upload 使用的尺寸。它由 capture size、质量档和 4K 稳定档位共同决定。本规范只对 4K 级输入启用档位下采样；低于 4K 的输入默认保持原始 capture size。4K 级判断必须覆盖横屏、竖屏、16:10、DCI 4K 和常见 4K 超宽窗口，同时避免单边很长但面积不足的窗口误入 4K 档位。

用途：

```text
RGB resize 目标尺寸
depth 对齐目标尺寸
max_disparity_px 解析尺寸
warp / mask / hole fill / temporal 工作尺寸
OpenXR left/right eye texture 尺寸
```

### depth_provider_size

`depth_provider_size` 是 depth model 内部推理尺寸，例如 ONNX / TensorRT / PyTorch provider 的输入分辨率。它可以不同于 `render_size`，但 provider 输出必须重新对齐到 `render_size` 后才能进入合成。

规则：

```text
depth provider 可以内部 resize。
runtime 合成只能消费 depth_render。
depth_render.shape[-2:] 必须等于 render_size。
```

### output_display_size

`output_display_size` 是 viewer 或打包输出看到的显示尺寸。对于 OpenXR full synthesis，它通常等于 left/right eye 的 render size。对于 half-SBS 打包，它可能表示 SBS packed frame 的显示尺寸。

规则：

```text
OpenXR full synthesis: output_display_size 应从 left/right eye 推导。
half_sbs: output_display_size 可从 sbs frame 推导。
viewer 建屏不得错误使用被 split 后的半宽 eye size 当作原始显示尺寸。
```

## OpenXR 输出路径规范

OpenXR 保留两条路径，但语义必须明确。

### RGB + depth direct 路径

用途：

```text
低延迟 shader path
OpenXR direct rendering
需要由 viewer shader 按规范参数实时生成左右眼的路径
```

处理方式：

```text
runtime 输出 source_rgb + depth。
viewer shader 根据 depth 做实时双眼位移。
runtime_output_format = openxr_rgb_depth。
```

该路径必须由 shader uniform snapshot 表达本文规范语义，不应混入 full synthesis 的 mask / hole fill 结果。

### Full synthesis eyes 路径

用途：

```text
影院 / 游戏 / 静态图像 / debug export 的质量优先路径
需要 runtime 完整合成左右眼时使用
```

处理方式：

```text
runtime 先生成 left_eye / right_eye / depth。
OpenXRRuntimeResult 直接携带 left_eye 和 right_eye。
viewer 直接上传左右眼纹理。
runtime_output_format = openxr_full_synthesis_eyes。
```

该路径中 viewer 不应再把结果降级成 `(rgb, depth)`，也不应重新按 viewer shader 生成视差。

## 阶段职责

### Depth 阶段

Depth 阶段只负责产生和 render RGB 对齐的 normalized depth。它不负责决定物理 IPD，也不负责最终立体强度。

输出要求：

```text
depth_render: 1 通道或可解释为 1 通道的 depth tensor
shape: render_height x render_width
range: normalized / relative，进入 depth_response 前应有稳定约定
```

### Depth Response 阶段

Depth response 阶段把 normalized depth 转成相对视差权重。它负责深度层次分配，而不是最终强度上限。

输出要求：

```text
response: 与 render_size 对齐
建议范围：[-1, 1]
convergence 附近接近 0
```

### Parallax Budget 阶段

Parallax budget 阶段只解析 `max_disparity_px`。它负责最终最大视差上限。

输入必须是：

```text
render_width
render_height
strength_preset
aspect protection rule
```

不允许继续使用：

```text
capture_width * max_shift_ratio
IPD * stereo_scale * depth_strength
```

作为 normalized-depth 路径的核心公式。

### Warp 阶段

Warp 阶段只做几何位移：

```text
left_shift_px  = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

所有 shift 都应在 render_size 像素坐标系内表达。

### Mask 阶段

Mask 阶段标记 warp 后暴露出来的遮挡空洞、深度断裂边缘、屏幕边缘风险区。Mask 不是补洞本身。

输入：

```text
depth_render
disparity_px / shift_px
edge_threshold
edge_dilation
screen_edge_mask_suppression
```

输出：

```text
occlusion / disocclusion mask
```

### Hole Fill 阶段

Hole fill 阶段根据 mask 修复左右眼图像中的空洞和边缘缺口。它不应该承担“修复过大视差预算”的职责。

实时默认建议：

```text
balanced: edge_aware_fill, radius=3, strength=1.0
```

质量优先或静态图像可以使用：

```text
quality: directional_edge_aware_fill
```

### Temporal 阶段

Temporal 阶段只负责跨帧稳定，不应改变当前帧的视差预算语义。

规则：

```text
scene reset 后应重置 temporal state。
render_size 变化后应重置 temporal state。
source target 切换后应重置 temporal state。
```

## 为什么不用物理 IPD 公式

真实双目几何公式是：

```text
disparity_px = f_px * IPD * (1/Z - 1/Z0)
```

这个公式成立的前提是：

```text
Z: 当前像素对应三维点的 camera-space Z-depth，单位 m
Z0: 零视差 / convergence 平面的 camera-space Z-depth，单位 m
IPD: 双眼瞳距，单位 m
f_px: 当前相机内参中的焦距，单位 pixel
```

也就是说，`Z` 必须是真实米制深度，并且 RGB、depth、相机内参必须对齐。单目 depth model 的输出通常只是相对深度、归一化深度或 inverse-depth-like 信号。它可以表达远近顺序，但不能直接解释为 `0.8m`、`2.0m` 这类物理距离。

因此，Desktop2Stereo 的单目 2D 转 3D 路径必须明确使用：

```text
max_disparity_px / parallax_budget
```

而不是假装 normalized depth 是物理 `Z`。

## 参数定义

### max_disparity_px

`max_disparity_px` 表示最终输出图像上，左右眼之间允许产生的最大水平视差。它是左右眼总视差，不是单眼位移。

例如：

```text
max_disparity_px = 96px
left_shift_px max  = +48px
right_shift_px max = -48px
left/right total   = 96px
```

normalized-depth 路径不得使用下面的经验乘法链作为核心强度公式：

```text
IPD * stereo_scale * depth_strength * max_shift_ratio
```

用户想要更强立体感，应调 `max_disparity_px` 或 GUI 上对应的“立体深度 / Parallax Budget”档位。

### depth_response

`depth_response` 是 normalized depth 到视差权重的曲线。它不是物理参数，而是艺术和工程参数。

它决定：

```text
近景是否更突出
远景是否压缩
中景是否稳定
convergence 附近是否接近零视差
```

示意：

```text
raw = convergence - depth
response = curve(raw)
response = clamp(response, -1, 1)
disparity_px = response * max_disparity_px
```

`depth_response` 控制“深度层次如何分配”，`max_disparity_px` 控制“最大能有多立体”。两者不能混成同一个强度乘法链。

### convergence

`convergence` 表示零视差层，也就是哪一层深度贴近屏幕平面：

```text
response(depth == convergence) ~= 0
```

它不应该被当成单纯的强度参数。调 convergence 会改变前景/背景相对屏幕的位置，而不是只改变整体强弱。

### IPD

在 normalized-depth 路径中，GUI 不应让普通用户调 `IPD` 来追求立体强弱。原因是没有 metric `Z` 和真实 `f_px`，IPD 无法形成严格物理意义。

允许场景：

```text
metric depth path: disparity_px = f_px * IPD * (1/Z - 1/Z0)
```

normalized-depth runtime、GUI、OpenXR direct shader 和 API 边界不得把 `IPD` 作为立体强度参数。

## 分辨率预算解析

内部核心参数使用像素预算，而不是持续使用比例：

```text
max_disparity_px = resolve_parallax_budget(width, height, strength_preset)
```

GUI 可显示为：

```text
舒适 / 标准 / 强 / 极强
```

推荐基础表：

| 分辨率等级 | 舒适 | 标准 | 强 | 极强 |
|---|---:|---:|---:|---:|
| 720p 级 | 24px | 36px | 48px | 64px |
| 1080p 级 | 32px | 48px | 64px | 80px |
| 1440p 级 | 48px | 64px | 88px | 112px |
| 2160p 级 | 64px | 96px | 128px | 160px |

这里的“分辨率等级”由短边决定：

```text
short_side = min(width, height)
```

这样横屏和竖屏可以自然共用同一套预算：

```text
3840x2160 -> short_side = 2160 -> 2160p 级
2160x3840 -> short_side = 2160 -> 2160p 级
1920x1080 -> short_side = 1080 -> 1080p 级
1080x1920 -> short_side = 1080 -> 1080p 级
```

## OpenXR 下采样和 render size

OpenXR 为了降低 GPU 上传量、提高输出帧率，可以把 4K 输入下采样到较低的稳定 eye texture 档位。下采样只对 4K 级输入启用；低于 4K 的输入保持原始 capture size，不得按 `stereo_render_scale` 继续缩小。4K 级输入包含横屏 3840x2160，也包含竖屏 2160x3840。这个因素必须进入 `max_disparity_px` 解析。

核心规则：

```text
max_disparity_px 绑定实际 stereo synthesis / OpenXR upload 的 render size，
不是永远绑定原始 capture size。
```

原因是 `max_disparity_px` 的单位是最终合成图像上的像素。如果 4K 输入下采到 2K 后再合成立体，仍使用 4K 的像素预算，会让实际视差比例过大，边缘拉扯和补洞压力都会明显增加。

尺寸分层：

```text
capture_size: 原始显示器/窗口捕捉尺寸
render_size: 实际用于 depth 对齐、stereo synthesis、OpenXR upload 的 eye texture 尺寸
```

预算应按 `render_size` 解析：

```text
max_disparity_px = resolve_parallax_budget(render_width, render_height, strength_preset)
```

示例：

```text
低于 4K 横屏输入，保持原尺寸: 1920x1080 -> 标准 48px
低于 4K 竖屏输入，保持原尺寸: 1080x1920 -> 标准 48px
4K 横屏输入，Render Scale = 4K / 3840x2160 -> 3840x2160 -> 标准 96px
4K 横屏输入，Render Scale = 3K / 3200x1800 -> 3200x1800 -> 标准约 80px，可由档位插值得到
4K 横屏输入，Render Scale = 2K / 2560x1440 -> 2560x1440 -> 标准 64px
4K 横屏输入，Render Scale = 1K / 1920x1080 -> 1920x1080 -> 标准 48px
4K 竖屏输入，Render Scale = 4K / 3840x2160 -> 2160x3840 -> 标准 96px
4K 竖屏输入，Render Scale = 3K / 3200x1800 -> 1800x3200 -> 标准约 80px，可由档位插值得到
4K 竖屏输入，Render Scale = 2K / 2560x1440 -> 1440x2560 -> 标准 64px
4K 竖屏输入，Render Scale = 1K / 1920x1080 -> 1080x1920 -> 标准 48px
```

处理顺序：

```text
capture RGB
-> resize 到 render_size
-> depth 对齐到 render_size
-> stereo synthesis 在 render_size 坐标系计算 disparity_px
-> OpenXR 上传 render_size 的 left/right eye texture
```

不要在 4K 上用 4K 预算先合成再下采，也不要在 2K 上合成却继续使用 4K 的 `max_disparity_px`。前者浪费算力且让补洞尺度复杂，后者会导致立体强度失真。

4K 输入的档位选择由 `stereo_render_scale` / `Render Scale` 表达。GUI 只展示固定分辨率档位，不展示连续比例或阈值：

```text
Render Scale:
4K / 3840x2160 -> 横屏 3840x2160，竖屏 2160x3840
3K / 3200x1800 -> 横屏 3200x1800，竖屏 1800x3200
2K / 2560x1440 -> 横屏 2560x1440，竖屏 1440x2560
1K / 1920x1080 -> 横屏 1920x1080，竖屏 1080x1920
```

Render Scale 的有效配置值只能是这些固定档位标签或等价分辨率字符串。`1.0`、`5/6`、`2/3`、`0.5` 以及 `0.92`、`0.75`、`0.58` 这类数值都不是档位选择语义；runtime 不得按数值阈值推断档位。

解析方式：

```text
short_side = min(capture_width, capture_height)
long_side = max(capture_width, capture_height)
pixels = capture_width * capture_height
uhd_4k_pixels = 3840 * 2160

is_4k_full_or_ultrawide = long_side >= 3840 and short_side >= 1600
is_near_4k_window = pixels >= uhd_4k_pixels * 0.85 and long_side >= 3200 and short_side >= 1600
is_4k_tier_input = is_4k_full_or_ultrawide or is_near_4k_window

if not is_4k_tier_input:
    render_size = align(capture_size)
else:
    render_size = resolve_4k_tier_size(stereo_render_scale, orientation=capture_orientation)

max_disparity_px = resolve_parallax_budget(render_width, render_height, strength_preset)
```

这里的 `stereo_render_scale` 不是任意输入尺寸的连续缩放比例，而是 4K 级输入的稳定档位选择信号。

如果后续做动态分辨率稳帧率，也不要每帧改变预算。只有在这些情况下重算：

```text
用户切换质量档
OpenXR render scale 改变并导致 4K 档位变化
输入源尺寸跨入或离开 4K 级判断条件，或尺寸变化超过阈值
动态分辨率稳定跨档，例如 3840x2160 -> 3200x1800 -> 2560x1440 -> 1920x1080，或 2160x3840 -> 1800x3200 -> 1440x2560 -> 1080x1920
```

## 宽高比处理

宽高比保护仍然适用，但它属于 Parallax Budget 阶段，必须在 `render_size` 已经解析之后执行。这里的 `width / height` 一律表示实际 stereo synthesis、OpenXR upload 或 packed synthesis 使用的 `render_width / render_height`，不是原始 `capture_width / capture_height`。

常规比例不做修正，包括：

```text
4:3
16:10
16:9
9:16 竖屏
```

这些比例都只按 `render_size` 的短边分辨率决定基础预算。不要为了少见超宽屏污染主公式。

只有最终 `render_size` 超过 2:1 时，才启用保护性降级：

```text
aspect = max(render_width, render_height) / min(render_width, render_height)

if aspect <= 2.0:
    aspect_factor = 1.0
else:
    aspect_factor = clamp(2.0 / aspect, 0.70, 1.0)

max_disparity_px = base_budget * aspect_factor
```

4K 级输入判断和宽高比保护是两个不同阶段：

```text
capture_size 用于判断输入是否进入 4K 固定档位。
render_size 用于解析短边预算和 aspect_factor。
```

因此，`3840x1600` 这类 4K 超宽输入可以进入 4K 级判断，以便按 Render Scale 下采样到稳定档位；但如果最终档位是 `2560x1440` 这类 16:9 render size，就不再触发超宽降预算。反过来，低于 4K 级的超宽输入默认保持 `render_size = capture_size`，如果最终宽高比超过 2:1，仍然触发保护性降级。

示例：

```text
4K 横屏输入: 3840x2160 -> 2560x1440 -> aspect = 1.78 -> factor = 1.0
4K 竖屏输入: 2160x3840 -> 1440x2560 -> aspect = 1.78 -> factor = 1.0
4K 超宽输入: 3840x1600 -> 2560x1440 -> aspect = 1.78 -> factor = 1.0
非 4K 超宽输入: 3440x1440 -> 3440x1440 -> aspect = 2.39 -> factor ~= 0.84
调用方显式保留超宽 render_size: 3840x1600 -> 3840x1600 -> aspect = 2.40 -> factor ~= 0.83
32:9 render_size: 5120x1440 -> aspect = 3.56 -> factor = 0.70
```

核心原则：

```text
短边决定标准预算。
常规 render_size 比例不修正。
只有最终 render_size 的 aspect > 2.0 才降预算。
不要用 raw capture aspect 直接修正已经映射到固定 16:9 / 9:16 档位的预算。
```

## 窗口捕捉处理

窗口程序捕捉的分辨率可能不固定。不要每帧重新计算 `max_disparity_px`，否则窗口尺寸轻微变化会导致立体强度抖动。

规则：

```text
1. capture session 开始时，根据当前 render_size 解析一次预算。
2. 会话内保持 resolved_max_disparity_px 稳定。
3. 只有显示器/窗口目标切换、OpenXR render scale 变化，或尺寸变化超过阈值时才重新解析。
```

重算阈值：

```text
render_size short_side 变化超过 10%
或最终 render_size aspect 跨过 2.0 超宽屏保护阈值
或用户重新选择显示器/窗口
```

## 与补洞和 mask 的关系

视差预算越大，左右眼位移越大，disocclusion 区域越多，补洞压力越高。

因此参数优先级应为：

```text
1. 先定 max_disparity_px，控制整体最大视差。
2. 再定 depth_response / convergence，控制深度层次分布。
3. 最后调 mask / hole fill，修复遮挡边缘和空洞。
```

不要用补洞去弥补过大的视差预算。超过预算产生的边缘虚影、拉扯、重影，后处理很难可靠修复。

## 实现要求

必须提供统一的预算解析函数。这里的 `width/height` 必须是实际合成和上传用的 `render_size`，不是未经下采样的原始捕捉尺寸：

```text
resolve_parallax_budget(render_width, render_height, strength_preset) -> max_disparity_px
```

伪代码：

```text
short_side = min(width, height)
base_budget = lookup_or_interpolate_budget(short_side, strength_preset)

aspect = max(width, height) / min(width, height)
if aspect <= 2.0:
    return base_budget

factor = clamp(2.0 / aspect, 0.70, 1.0)
return base_budget * factor
```

合成路径只使用：

```text
disparity_px = depth_response(depth, convergence) * max_disparity_px
left_shift_px = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

normalized-depth 路径的 runtime 边界只接受 `max_disparity_px` / `parallax_budget_preset`、`depth_response`、`convergence` 这类规范参数。非规范经验字段不得作为合成核心输入，也不得越过导入/解析层进入本文定义的运行时目标流程。

## 参考资料

- [2D to 3D conversion](https://en.wikipedia.org/wiki/2D_to_3D_conversion)：总结 2D 转 3D工作流中的 depth budget、convergence、comfortable disparity、uncovered area filling 等概念。
- [2D-plus-Depth](https://en.wikipedia.org/wiki/2D-plus-depth)：说明 RGB + depth map 是常见 3D显示/转换表示，但 depth map 不等于完整真实几何。
- [Binocular disparity](https://en.wikipedia.org/wiki/Binocular_disparity)：说明双目视差通常表现为左右图像中的水平像素差，且 disparity 与距离相关。
- [Quality Assessment of DIBR-synthesized views: An Overview](https://arxiv.org/abs/1911.07036)：DIBR 合成视图质量评估综述，说明 DIBR 是 3D video / free-viewpoint / VR 中的重要视图合成技术。
- [Hole Filling with Multiple Reference Views in DIBR View Synthesis](https://arxiv.org/abs/1802.03079)：说明 DIBR 视图合成中 disocclusion holes 的来源和补洞问题。
- [3D Photography using Context-aware Layered Depth Inpainting](https://arxiv.org/abs/2004.04727)：说明单张 RGB-D 到新视角合成时，遮挡区域需要 inpainting / hallucinated content。
- [How visual discomfort changes with horizontal viewing angle on stereoscopic display](https://arxiv.org/abs/1811.08639)：说明 stereoscopic display 中视差和观看条件会影响视觉不适。
