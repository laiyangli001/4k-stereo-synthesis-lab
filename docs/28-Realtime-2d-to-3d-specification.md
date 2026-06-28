# 实时立体视觉合成管线设计规范书

**文档版本**：3.0
**发布日期**：2026年6月
**规范地位**：Desktop2Stereo 当前正式最终运行时流程规范
**编制依据**：ISO/IEC、Khronos OpenXR、UWA 等国际/行业标准，以及 Desktop2Stereo 当前工程实现


## 1. 引言与范围

### 1.1 背景
实时立体视觉合成（Real-time Stereoscopic Synthesis）是将单目RGB视频流实时转换为双目立体视觉内容的关键技术，广泛应用于AR/VR头显、裸眼3D显示、远程呈现、沉浸式媒体等领域。本规范书定义 Desktop2Stereo 当前正式最终运行时流程规范，涵盖从捕获输入到最终输出的十一个核心处理步骤。历史文档 `25-2d-to-3d-runtime-specification.md` 自本文档生效后作废；若两者存在差异，以本文档为准。

### 1.2 范围
本规范适用于：
- 基于单目RGB输入的实时双目立体视觉合成系统
- 支持OpenXR标准的AR/VR设备端立体渲染
- 裸眼3D显示终端的立体内容生成
- 沉浸式媒体应用中基于渲染系统的视觉内容合成

### 1.3 规范性引用文件

| 编号 | 标准/规范 | 名称 | 发布机构 |
|------|----------|------|---------|
| [1] | ISO/IEC TR 23090-27:2025 | Information technology — Coded representation of immersive media — Part 27: Media and architectures for render-based systems and applications | ISO/IEC JTC 1/SC 29 |
| [2] | T/UWA 035-2025 | 基于双目视差的裸眼3D系统参考架构与通用技术要求 | 世界超高清视频产业联盟 |
| [3] | OpenXR 1.1 | API Specification for XR Applications | Khronos Group |
| [4] | SMPTE ST 2070 | Stereoscopic 3D in MXF | SMPTE |
| [5] | ITU-R BT.2025 | 3DTV Program Production and Exchange | ITU-R |


## 2. 术语与定义

| 术语 | 定义 |
|------|------|
| **视差（Disparity）** | 同一场景点在左右眼图像中的水平像素位置差异 |
| **深度图（Depth Map）** | 本规范中的默认深度图为 normalized / relative depth，不等同于真实米制 Z-depth |
| **DIBR** | Depth-Image-Based Rendering，基于深度图像的渲染 |
| **捕获尺寸（capture_size）** | 输入源原始分辨率，来自显示器、窗口、文件或 API 帧 |
| **渲染尺寸（render_size）** | 立体合成管线内部的唯一工作分辨率 |
| **4K档位（4K Tier）** | 4K级输入映射到 4K/3K/2K/1K 固定 render_size 档位的规则 |
| **最大视差（max_disparity_px）** | 软件层面允许的左右眼总视差像素预算，用于控制立体感强度（非物理测量值） |
| **视差预算（Parallax Budget）** | 根据 `render_size` 和用户强度档位解析出的 `max_disparity_px` 及其响应曲线 |
| **深度响应（depth_response）** | normalized depth 到相对视差权重的映射函数 |
| **OpenXR** | Khronos Group发布的XR应用开放API标准 |
| **时域稳定化** | 利用历史帧信息减少帧间闪烁和抖动的处理技术 |


## 3. 总体架构与约束

### 3.1 系统架构图

```
Capture Input → Resolve Render Size → Resize RGB To Render Size → Depth Estimation →
Depth Postprocess → Resolve Parallax Budget → Disparity Field →
Stereo Warp → Mask and Hole Fill → Temporal Stabilization → Output Pack / Viewer Upload
```

### 3.2 全局核心约束

1. **工作坐标系一致性**：进入立体合成（Stereo Synthesis）后，`render_size` 是唯一工作坐标系。
2. **数据对齐约束**：RGB、Depth、Disparity、Mask、Hole Fill、Temporal、Left/Right Eye 全部必须严格对齐 `render_size`。
3. **视差控制约束**：`max_disparity_px` 必须按 `render_size` 解析，其值由软件层根据用户偏好和固定预算表计算，不受物理显示设备参数的直接影响。
4. **Normalized-depth约束**：默认单目深度路径不得把 normalized / relative depth 当作真实米制 `Z` 代入物理 IPD 公式。
5. **Render Scale约束**：`Render Scale` 只表示 4K级输入的固定档位选择，不是任意输入尺寸的连续缩放比例。
6. **坐标系规则**：所有图像坐标系采用**左上角为原点**（0,0），x轴向右，y轴向下。
7. **输出分层约束**：运行目标、质量模式、合成方式、render size、transport、packing format 必须分层表达，不能互相替代。

### 3.3 工程分层定义

完整运行时必须按以下层级解析，任何层级都不能替代其它层级：

```text
Capture Source
→ Application Runtime Target
→ Runtime Quality Mode
→ Stereo Synthesis Mode
→ Render Size / 4K Tier
→ Output Transport
→ Output Packing Format
→ Viewer / Device Presentation
```

核心原则：

```text
OpenXR 不是显示封装格式。
half_sbs / full_sbs / TAB 不是运行模式。
网络推流不是立体算法。
3D 显示器不是 capture source。
```

| 层级 | 规范值示例 | 说明 |
|------|------------|------|
| Capture Source | monitor_capture / window_capture / file_image / file_video / api_frame | 只负责提供 RGB frame 和 source metadata |
| Application Runtime Target | local_display / network_stream / openxr / debug_export / headless_api / auto | 定义最终输出目标 |
| Runtime Quality Mode | auto / movie / cinema / game / game_low_latency / image / debug | 定义延迟、质量、稳定性偏好 |
| Stereo Synthesis Mode | rgb_depth_direct / full_synthesis_eyes / packed_synthesis | 定义左右眼如何生成 |
| Output Transport | local_window / local_fullscreen / encoded_stream / openxr_swapchain / file_export / api_result | 定义结果送到哪里 |
| Output Packing Format | mono / half_sbs / full_sbs / half_tab / full_tab / anaglyph / interleaved / leia / depth_map | 只定义 left/right eye 如何封装成一帧 |

### 3.4 RuntimeSettingsSnapshot 与热更新边界

GUI 或 API 必须生成不可变配置快照，并通过线程安全队列交给 runtime。所有用户可调立体参数只允许在帧边界应用。

```text
GUI / API
→ RuntimeSettingsSnapshot(version=N)
→ settings_update_q
→ RuntimePipelineLoop frame boundary
→ active_settings
```

`RuntimeSettingsSnapshot` 必需字段：

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

参数变更分级：

| 参数 | Hot Reload | Reset Temporal | Rebuild Resources | 说明 |
|------|:----------:|:--------------:|:-----------------:|------|
| `max_disparity_px` / `parallax_budget_preset` | 是 | 可选 | 否 | 强度突变时可 reset temporal |
| `convergence` | 是 | 可选 | 否 | 大幅变化建议 reset temporal |
| `depth_response` | 是 | 是 | 否 | 曲线变化会改变全局视差分布 |
| `hole_fill_mode` / mask 参数 | 是 | 否 | 否 | 影响 mask 与 hole fill 行为 |
| `temporal_enabled` | 是 | 是 | 否 | 开关变化需要清理历史状态 |
| `temporal_strength` | 是 | 否 | 否 | 0 可表示关闭效果 |
| `cross_eyed` / `eye_order` | 是 | 否 | 否 | presentation 层修正 |
| `output_packing_format` | 部分 | 否 | 可能 | 本地/推流可能需要重建输出缓冲 |
| `stereo_render_scale` | 否 | 是 | 是 | 仅当 4K 档位变化并导致 `render_size` 变化时重建 |
| `stereo_synthesis_mode` | 否 | 是 | 是 | direct/full synthesis 切换 |
| `depth_backend` | 否 | 是 | 是 | provider/engine 变化 |
| `capture_source` / `capture_target` | 否 | 是 | 是 | 重新捕捉和重建 source metadata |
| `openxr_swapchain_format` | 否 | 是 | 是/重启 | OpenXR session 资源 |
| `encoder_profile` | 否 | 否 | 是 | 网络推流编码器资源 |

每次 runtime result / OpenXR result / debug export 必须记录：

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


## 4. 分步骤详细规范

### 步骤 1：Capture Input（捕获输入）

#### 4.1.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | 显示器/窗口/文件/API的RGB帧 |
| **坐标系** | `capture_size`（原始捕获分辨率） |
| **输出** | `source_rgb` + `capture_metadata`（含分辨率、色彩空间、时间戳、帧率） |

#### 4.1.2 处理语义

系统从多种来源（显示器捕获、窗口捕获、视频文件解码、图形API帧缓冲）获取RGB帧，同时采集元数据以供下游步骤使用。

#### 4.1.3 行业依据

- **ISO/IEC TR 23090-27:2025**：定义了媒体直接交付给基于渲染的应用（如游戏引擎）的架构框架。
- **T/UWA 035-2025**：规定了基于双目视差的裸眼3D系统内容层（3D视频采集、2D视频生成3D视频）的通用要求。
- **SMPTE ST 2070**：提供了立体3D内容的通用规定。

#### 4.1.4 参考论文

- Cheng, Z. et al. "RTS-Mono: A Real-Time Self-Supervised Monocular Depth Estimation Method for Real-World Deployment." arXiv:2511.14107, 2025.

#### 4.1.5 前沿技术说明

**选型方案**：采用统一捕获抽象层（Unified Capture Abstraction Layer），支持Windows DXGI/DirectCapture、Linux DRM/KMS、macOS ScreenCaptureKit等多后端。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 单一平台捕获API（如仅Windows GDI） | 无法满足跨平台部署需求 |
| 基于FFmpeg的软件解码直连 | 延迟高，缺乏与渲染管线的深度集成 |


### 步骤 2：Resolve Render Size（解析渲染尺寸）

#### 4.2.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `capture_size`、Application Runtime Target、Runtime Quality Mode、`Render Scale` / `stereo_render_scale` |
| **输出** | `render_size`（宽度×高度） |

#### 4.2.2 处理语义

`render_size` 是 stereo synthesis、OpenXR upload、depth 对齐、mask、hole fill、temporal 的唯一工作尺寸。当前规范不向用户暴露 `native` / `fixed` / `dynamic` 策略选择；用户侧只暴露 `Render Scale` 作为 4K级输入的固定档位选择信号。

| 输入条件 | 规则 | 用途 |
|----------|------|------|
| 非 4K级输入 | `render_size = capture_size` | 避免普通 1080p/2K 输入被连续缩放，保持视差预算稳定 |
| 4K级横屏输入 | 按 `Render Scale` 映射到 3840x2160、3200x1800、2560x1440、1920x1080 | 降低 OpenXR 上传量、网络码率、本地 GPU 压力 |
| 4K级竖屏输入 | 按同一档位映射到 2160x3840、1800x3200、1440x2560、1080x1920 | 保持竖屏方向，预算按 `render_size` 短边解析 |

4K级判断必须方向无关，并覆盖常见全屏和近 4K 窗口：3840x2160、2160x3840、4096x2160、3840x2400、3840x1600 属于 4K级；2560x1440、3440x1440、1080x1920、1000x3000 不属于 4K级。

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
```

`Render Scale` 的有效配置值只能是固定档位标签或等价分辨率字符串：

```text
4K / 3840x2160 -> 横屏 3840x2160，竖屏 2160x3840
3K / 3200x1800 -> 横屏 3200x1800，竖屏 1800x3200
2K / 2560x1440 -> 横屏 2560x1440，竖屏 1440x2560
1K / 1920x1080 -> 横屏 1920x1080，竖屏 1080x1920
```

`1.0`、`5/6`、`2/3`、`0.5` 以及 `0.92`、`0.75`、`0.58` 这类数值不是档位选择语义；runtime 不得按数值阈值推断档位。

#### 4.2.3 行业依据

- **OpenXR 1.1规范**：定义了`XrViewConfigurationType`，包括`XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO`，提供了硬件无关的立体显示视图配置。
- **ISO/IEC TR 23090-27:2025**：第8.3.1节涵盖单面板平面显示器的渲染分辨率管理。

#### 4.2.4 参考论文

- "Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency." arXiv e-prints, May 2025.

#### 4.2.5 前沿技术说明

**当前项目采用**：4K级输入固定档位解析。`Render Scale` 只在输入跨入 4K级条件后生效，非 4K 输入保持 `capture_size`。

**未来候选**：动态分辨率稳帧率可以加入，但只能在 4K/3K/2K/1K 等稳定档位之间切换，不能每帧连续改变预算。

**未采用/不适用**：

| 方案 | 原因 |
|------|------|
| 任意连续缩放因子 | 会造成 `max_disparity_px` 预算随输入尺寸连续漂移 |
| 用户可选 `native/fixed/dynamic` 策略 | 当前产品规范只保留固定 4K 档位选择 |
| 每帧动态改 render size | 会导致 depth、mask、temporal、OpenXR texture 频繁重建 |


### 步骤 3：Resize RGB To Render Size（RGB缩放至渲染尺寸）

#### 4.3.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `source_rgb`（`capture_size`） |
| **转换** | `capture_size` → `render_size` |
| **输出** | `render_rgb`（`render_size`） |

#### 4.3.2 处理语义

将原始捕获的 RGB 图像转换到管线工作分辨率 `render_size`。若 `render_size == capture_size`，不得做无意义缩放；若 4K级输入映射到固定档位，则 RGB 必须先缩放到 `render_size`，后续 depth、disparity、mask、hole fill、temporal、left/right eye 均以该尺寸为准。`aspect_policy` 如果需要 crop / letterbox，必须在 `render_size` 解析前明确，否则 RGB 与 depth 会错位。

#### 4.3.3 行业依据

- **ITU-R BT.2025**：规定了3DTV节目制作与交换的数字图像系统标准。
- **SMPTE标准体系**：为数字电影全流程提供技术规范。

#### 4.3.4 参考论文

- "NTIRE 2025 Challenge on HR Depth from Images of Specular and Transparent Surfaces." arXiv, 2025.

#### 4.3.5 前沿技术说明

**当前项目采用**：capture preprocess / runtime preprocess 负责把输入转为 RGB、float 0..1、CHW/BCHW tensor，并在需要时 resize 到 `render_size`。

**未来候选**：质量优先路径可评估 Lanczos / Catmull-Rom；实时路径继续优先使用 GPU 友好的 bilinear / bicubic / area resize。

**未采用/不适用**：不得在 depth provider 内隐式改变 runtime 工作坐标系；provider 内部 resize 后必须回到 `render_size`。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| Nearest-neighbor插值 | 锯齿严重，不适合立体合成 |
| 仅Bilinear插值 | 高频细节保留不足，影响深度估计精度 |


### 步骤 4：Depth Estimation（深度估计）

#### 4.4.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `render_rgb`（`render_size`，RGB float 0..1，CHW/BCHW tensor） |
| **处理** | Depth provider 内部可 resize 到模型输入尺寸，但最终必须回到 `render_size` |
| **输出** | `depth_render`（`render_size`，normalized / relative depth，float tensor） |

#### 4.4.2 处理语义

从单目 RGB 图像估计每像素 normalized / relative depth。Depth provider 可以内部 resize 到 ONNX / TensorRT / PyTorch / MIGraphX / MPS / XPU 等后端要求的模型输入尺寸，但返回 runtime 前必须 upsample / align 到 `render_size`。

Depth provider 必须满足以下契约：

```text
predict_profile(render_rgb) -> DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)
depth.shape[-2:] == render_size
depth range / near-far direction 必须可通过 provider_info / debug_info 追踪
```

默认 normalized-depth 路径不得把 provider 输出解释为真实米制 `Z`。只有明确进入 metric depth path，且具备真实相机内参与 metric depth 定义时，才允许使用物理 IPD 公式。

#### 4.4.3 行业依据

- **ITU/MPEG标准委员会**：正在定义虚拟视图系统的数据压缩（传输格式），并提供非规范性的深度估计和视图合成工具。
- **T/UWA 035-2025**：2D视频生成3D视频是内容层的核心功能。

#### 4.4.4 参考论文

- Cheng, Z. et al. "RTS-Mono: A Real-Time Self-Supervised Monocular Depth Estimation Method for Real-World Deployment." arXiv:2511.14107, 2025.
  - 可作为未来轻量级实时 provider 候选，不是当前项目默认 provider。
- Cheng, J. et al. "MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors." CVPR 2025.
  - 属于 stereo / multiview / stereo matching 与 monodepth prior 融合方向，可作为研究参考，不是当前单目 RGB depth provider 选型。
- "CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach." arXiv, 2025.
  - 可作为未来自监督深度模型参考。

#### 4.4.5 前沿技术说明

**当前项目采用**：provider-agnostic 深度估计架构，而不是单一固定模型。当前 ModelRegistry 覆盖 Depth-Anything、Video-Depth-Anything、DA3、Metric Depth-Anything、InfiniDepth、Distill-Any-Depth、DPT、ZoeDepth、DepthPro 等模型族；Depth provider 后端覆盖 PyTorch CUDA、ONNX Runtime CUDA、TensorRT native、ORT TensorRT、PyTorch ROCm、MIGraphX、PyTorch MPS、PyTorch XPU。

当前 provider 规则：

```text
ModelRegistry 解析模型 ID 和 family。
provider 内部负责模型输入尺寸、normalize、artifact 加载、ONNX/TensorRT/MIGraphX 构建或加载。
ONNX CUDA 必须实际启用 CUDAExecutionProvider，否则报错。
IOBinding / DLPack / CUDA tensor / ROCm tensor / MPS / XPU 是后端优化，不改变 depth_render 合同。
provider 输出必须回到 render_size。
```

**未来候选**：RTS-Mono 等轻量级实时单目深度模型可作为新增 provider 评估；Video Depth / temporal depth 模型可作为减少深度闪烁的候选，但必须保持 `depth_render` 对齐 `render_size` 的合同。

**未采用/不适用**：

| 方案 | 原因 |
|------|------|
| 把 RTS-Mono 写成当前主引擎 | 当前代码没有 RTS-Mono provider，不能作为正式实现描述 |
| 把 MonSter++ 写成当前备用单目 provider | 其核心方向是 stereo / multiview / matching，不是当前单目 RGB runtime 的直接替换 |
| 传统 SGM | 需要双目输入，不适用于当前单目 RGB 输入路径 |
| 把 provider 输出当作 metric Z | 当前 normalized-depth 路径没有真实相机内参与米制深度合同 |


### 步骤 5：Depth Postprocess（深度后处理）

#### 4.5.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `depth_render`（`render_size`） |
| **坐标系** | `render_size` |
| **输出** | `depth_response_input`（`render_size`，归一化深度响应） |

#### 4.5.2 处理语义

对 provider 输出的 normalized / relative depth 进行合成前后处理，包括：
- **范围归一化与方向确认**：明确 near/far 方向和归一化范围，并在 debug/provider metadata 中可追踪。
- **render_size 对齐**：确保 `depth_render.shape[-2:] == render_size`。
- **边缘与前景响应控制**：根据 runtime 参数应用 foreground scale、antialias / edge-aware upsample 等处理。
- **状态边界**：深度后处理不得改变 `render_size`，也不得直接决定最终 `max_disparity_px`。

#### 4.5.3 行业依据

- **MPEG沉浸式视频标准**：定义了深度图的编码与传输格式。

#### 4.5.4 参考论文

- "Depth-guided Hole-filling Algorithm for View Synthesis." KCI, 2025.
  - 利用局部深度信息生成方向向量图进行图像修补

#### 4.5.5 前沿技术说明

**当前项目采用**：provider 输出归一化、上采样到 `render_size`，并在 synthesis 阶段使用 `postprocess_depth()`、edge-aware upsample、foreground/antialias 等轻量处理。

**未来候选**：Guided Filter / bilateral filter / temporal depth refinement 可以作为质量路径增强，但必须作为 depth postprocess 插件，不得改变 `depth_render` 坐标合同。

**未采用/不适用**：不能把 depth postprocess 写成“绝对深度转 [0,1]”，因为当前默认路径本来就是 normalized / relative depth。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 高斯滤波 | 过度平滑，深度边缘模糊，影响视差图质量 |
| 仅裁剪+归一化 | 缺乏对深度估计噪声的抑制能力 |


### 步骤 6：Resolve Parallax Budget（解析视差预算）

#### 4.6.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `render_width`、`render_height`、`parallax_budget_preset`（comfort / standard / strong / extreme）、aspect protection rule |
| **输出** | `max_disparity_px`（左右眼总视差预算，单位 pixel） |

#### 4.6.2 处理语义

本步骤根据 `render_size` 和用户选择的 Parallax Budget 档位解析 `max_disparity_px`。`max_disparity_px` 是左右眼总视差预算，不是单眼位移，也不是物理 IPD。

推荐基础表：

| 分辨率等级 | comfort | standard | strong | extreme |
|------------|--------:|---------:|-------:|--------:|
| 720p级 | 24px | 36px | 48px | 64px |
| 1080p级 | 32px | 48px | 64px | 80px |
| 1440p级 | 48px | 64px | 88px | 112px |
| 2160p级 | 64px | 96px | 128px | 160px |

分辨率等级由短边决定：

```text
short_side = min(render_width, render_height)
base_budget = lookup_or_interpolate_budget(short_side, parallax_budget_preset)
```

宽高比保护属于 Parallax Budget 阶段，必须在 `render_size` 已解析之后执行。常规 4:3、16:10、16:9、9:16 不做修正；只有最终 `render_size` 超过 2:1 时才启用保护性降级：

```text
aspect = max(render_width, render_height) / min(render_width, render_height)

if aspect <= 2.0:
    aspect_factor = 1.0
else:
    aspect_factor = clamp(2.0 / aspect, 0.70, 1.0)

max_disparity_px = base_budget * aspect_factor
```

窗口捕捉的预算不得每帧重算。只有在用户切换质量档、OpenXR render scale 改变并导致 4K 档位变化、输入源跨入/离开 4K级判断条件、最终 `render_size` 短边变化超过 10%、最终 aspect 跨过 2.0 保护阈值或用户重新选择显示器/窗口时，才重新解析预算。

normalized-depth 路径不得使用下面的经验乘法链作为核心强度公式：

```text
IPD * stereo_scale * depth_strength * max_shift_ratio
```

#### 4.6.3 行业依据

- **ISO/IEC TR 23090-27:2025**：支持渲染系统应用中的媒体自适应，允许根据应用场景调整渲染参数。
- **T/UWA 035-2025**：定义了裸眼3D内容制作中视差控制的通用要求，强调内容自适应。
- **OpenXR 1.1**：虽未强制指定视差控制方式，但提供了视图配置和合成层，允许应用自由决定视差幅度。

#### 4.6.4 参考论文

- Kim, S. et al. "Perceptual Disparity Limits for Stereoscopic 3D Content Based on Viewing Distance and Screen Size." IEEE Trans. on Visualization and Computer Graphics, vol. 28, no. 5, 2022.
  - 提供了视差范围与感知舒适度的映射数据，可作为设定 `base_disparity` 和限幅的依据。
- Wang, J. et al. "Content-adaptive Disparity Control for Stereoscopic Video." ACM Trans. on Graphics (Proc. SIGGRAPH), 2024.
  - 提出基于场景语义的视差调节方法，可动态调整视差幅度。
- "Real-time Disparity Control for Monocular Depth-based 3D Synthesis." IEEE VR 2025 Workshop, 2025.
  - 展示了纯软件视差控制策略在实时管线中的应用。

#### 4.6.5 前沿技术说明

**当前项目采用**：`resolve_parallax_budget(render_width, render_height, preset)` 表驱动预算解析，支持 comfort / standard / strong / extreme、短边插值和超宽保护。

**未来候选**：内容自适应、UI/人像/风景语义降预算可以作为上层 preset 选择器，但不得绕过 `max_disparity_px` 合同。

**未采用/不适用**：不采用 `short_side × 0.035 × content_factor × perf_factor` 作为正式公式；该公式不等价于当前实现的固定预算表。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| IPD × stereo_scale × depth_strength × max_shift_ratio 公式 | 参数物理意义不清，且依赖不存在的物理测量值；该公式过度简化，缺乏对内容类型的考量。 |
| 固定视差像素值（如30px） | 无法适应不同分辨率，在4K下立体感弱，在720p下易造成不适。 |
| 完全依赖EDID等硬件信息 | 假设固定观看环境，不适合软件播放器和通用应用；且EDID信息在多数场景不可靠。 |


### 步骤 7：Disparity Field（视差场生成）

#### 4.7.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `depth_response_input`（归一化深度响应）、`convergence`（汇聚平面深度，软件可配置）、`max_disparity_px`（来自步骤6） |
| **输出** | `disparity_px` / `shift_px`（`render_size`，浮点视差图） |

#### 4.7.2 处理语义

**核心公式**：
```
disparity_px = depth_response(depth, convergence) × max_disparity_px
```

其中 `depth_response(depth, convergence)` 将 normalized / relative depth 映射到相对视差权重，建议范围为 `[-1, 1]`，并在 convergence 附近接近 0。near/far 方向由 provider 输出约定和 depth_response 曲线共同决定，必须在 debug metadata 中可追踪，不得在下游阶段重新猜测。

#### 4.7.3 行业依据

- **MPEG虚拟视图系统**：视差估计是虚拟视图系统的三大核心组件之一。
- **T/UWA 035-2025**：定义了双目成像的三种视差关系。

#### 4.7.4 参考论文

- "DEFOM-Stereo: Depth Foundation Model Based Stereo Matching." arXiv, 2025.
  - 在KITTI 2012、KITTI 2015、Middlebury、ETH3D基准上排名第一
- "Trans embedded encoding volume for stereo matching." Applied Intelligence, 2025.
  - 在SceneFlow数据集上EPE达0.42

#### 4.7.5 前沿技术说明

**当前项目采用**：`depth_response(depth, convergence) * max_disparity_px` 的显式视差预算模型。当前默认响应曲线是可追踪的规范曲线名称，后续可替换为更复杂曲线，但输出仍必须是 `render_size` 对齐的 `disparity_px`。

**未来候选**：非线性 depth response、场景自适应 convergence、前景保护曲线可以加入，但必须保持 `max_disparity_px` 是唯一最终强度上限。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 线性深度-视差映射 | 缺乏对感知非线性的建模，立体感不足 |
| 固定汇聚平面 | 无法适应不同场景的深度分布 |


### 步骤 8：Stereo Warp（立体扭曲/视图合成）

#### 4.8.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `render_rgb`、`disparity_px`（`render_size`） |
| **输出** | `left_eye`、`right_eye`（`render_size`）、原始遮挡/去遮挡区域掩码 |
| **位移规则** | `left_shift_px = +disparity_px / 2`，`right_shift_px = -disparity_px / 2` |

#### 4.8.2 处理语义

基于 DIBR（Depth-Image-Based Rendering）技术生成左右眼：
1. `left_shift_px = +disparity_px / 2`
2. `right_shift_px = -disparity_px / 2`
3. 所有 shift 都在 `render_size` 像素坐标系内表达。
4. Warp 阶段只负责几何位移和 raw occlusion / disocclusion 区域暴露，不负责用补洞掩盖过大的视差预算。

#### 4.8.3 行业依据

- **MPEG沉浸式视频标准**：支持通过多个真实或虚拟摄像机捕获的沉浸式视频内容的存储与分发。
- **DIBR是虚拟视图合成的标准技术**。

#### 4.8.4 参考论文

- "GenStereo: Towards Open-World Generation of Stereo Images and Unsupervised Matching." arXiv, 2025.
  - 基于扩散模型，将扩散过程条件化为视差感知坐标嵌入和扭曲输入图像
  - 在11个立体数据集上训练，展现强泛化能力
- "Novel view synthesis with wide-baseline stereo pairs based on local–global information." 2025.
- "High efficiency depth image-based rendering with simplified inpainting-based hole filling." Springer.

#### 4.8.5 前沿技术说明

**当前项目采用**：baseline / layered DIBR 合成路径。`fast` / `fast_plus` 使用 baseline shift，`quality_4k` / `hq_4k` 使用 layered synthesis；OpenXR RGB+depth direct 路径由 viewer shader 使用规范 uniform snapshot 现场生成双眼。

**未来候选**：真实 3D warping、相机内参反投影、Z-buffer 或 mesh-based reprojection 可以作为 metric-depth / calibrated-camera 路径探索，但不属于当前 normalized-depth runtime 的默认实现。

**未采用/不适用**：当前默认路径没有真实相机内参和 metric `Z`，因此不能把“3D反投影 + Z-buffer”写成已采用实现。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 前向映射（Forward Mapping） | 产生大量空洞，后处理复杂 |
| 简单像素复制平移 | 无遮挡处理，立体感差 |


### 步骤 9：Mask and Hole Fill（掩码与空洞填充）

#### 4.9.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `left_eye`、`right_eye`、`depth_render`、`disparity_px`、遮挡掩码 |
| **输出** | `filled_left_eye`、`filled_right_eye`（`render_size`） |

#### 4.9.2 处理语义

1. **遮挡区域识别**：利用遮挡掩码标记在视图合成中暴露的背景区域
2. **空洞填充策略**：根据 mask、depth_render、disparity_px / shift_px 对左右眼暴露区域进行修补。Mask 只负责标记风险区，不负责修复本身。Hole fill 不得承担“修复过大视差预算”的职责。

#### 4.9.3 行业依据

- **DIBR空洞填充是虚拟视图生成中最关键的问题**。

#### 4.9.4 参考论文

- "Depth-guided Hole-filling Algorithm for View Synthesis." KCI, 2025.
  - 基于局部深度信息为每个空洞像素生成方向向量图进行图像修补
- "AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting." CVPR 2025.
  - 深度感知不可见掩码生成与自适应引导深度扩散
- "An efficient hole filling for depth image based rendering."

#### 4.9.5 前沿技术说明

**当前项目采用**：`edge_aware_fill` / `directional_edge_aware_fill`，并通过 `hole_fill_mode`、`hole_fill_radius`、`hole_fill_strength`、`mask_feather_radius`、`edge_threshold`、`edge_dilation` 等参数控制实时质量路径。OpenXR 实时默认推荐 balanced；静态图或导出可以使用 quality。

**未来候选**：AI inpainting 可以作为离线/质量增强路径，但不得进入默认实时路径，也不得掩盖过大的 `max_disparity_px`。

**未采用/不适用**：当前没有把大面积 AI 修补网络作为默认 runtime 依赖。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 仅背景色填充 | 视觉质量差，产生明显伪影 |
| 各向同性扩散填充 | 跨边缘扩散，产生模糊 |
| 仅依赖单一策略 | 无法处理多样化的空洞形态 |


### 步骤 10：Temporal Stabilization（时域稳定化）

#### 4.10.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `filled_left_eye`、`filled_right_eye`、掩码、时域状态 |
| **输出** | `stable_left_eye`、`stable_right_eye`（`render_size`） |
| **重置条件** | `render_size`变化 / source target变化 / scene reset时重置时域状态 |

#### 4.10.2 处理语义

1. **时域滤波**：对左右眼输出与历史状态进行稳定化处理。
2. **遮挡/掩码边界**：mask 区域可降低时域依赖，避免拖影。
3. **状态重置**：scene reset、`render_size` 变化、source target 切换、temporal 关键设置变化时必须清空相关历史状态。
4. **语义边界**：Temporal 只负责跨帧稳定，不得改变当前帧的 `max_disparity_px` 预算语义。

#### 4.10.3 行业依据

- **OpenXR规范**：定义了视图配置和合成层，时域稳定化是提升XR体验的关键技术。

#### 4.10.4 参考论文

- "StereoFG: Generating Stereo Frames from Centered Feature Stream." SIGGRAPH Asia 2025.
  - 新颖循环网络将中心特征传播到下一帧以提升时域稳定性
- "PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching." NeurIPS 2025.
  - 时域一致的深度估计，对AR等应用至关重要
- "Stable Sample Caching for Interactive Stereoscopic Ray Tracing." 2025.
  - 时域抗锯齿（TAA）与基于哈希的着色缓存

#### 4.10.5 前沿技术说明

**当前项目采用**：runtime temporal state + scene reset / render_size reset / source target reset 机制，使用 `temporal_enabled`、`temporal_strength`、`scene_reset_threshold`、`reset_cooldown_frames` 等参数控制。

**未来候选**：运动矢量、optical flow、temporal depth model 可以作为质量增强，但必须显式记录依赖和 reset 条件。

**未采用/不适用**：当前默认实现不依赖 motion-vector 驱动的 MA-EMA，因此不能把运动矢量场写成已采用输入。

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 无时域处理 | 帧间闪烁严重，立体感不稳定 |
| 固定系数EMA | 运动场景产生拖影，静态场景噪声抑制不足 |
| 仅单帧处理 | 无法利用时域信息提升质量 |


### 步骤 11：Output Pack / Viewer Upload（输出打包与视图上传）

#### 4.11.1 输入/输出

| 项目 | 规格 |
|------|------|
| **输入** | `stable_left_eye`、`stable_right_eye`（`render_size`） |
| **输出** | 多种输出格式（见下文） |

#### 4.11.2 处理语义

**A. 本地/网络输出格式**：
- **Mono**：单目（取左眼或平均）
- **Half-SBS**：左右水平并排，每眼横向压缩 1/2，packed frame 总尺寸为 `render_width × render_height`
- **Full-SBS**：左右水平并排，每眼保留完整宽度，packed frame 总尺寸为 `2 × render_width × render_height`
- **Half-TAB**：左右上下排列，每眼纵向压缩 1/2，packed frame 总尺寸为 `render_width × render_height`
- **Full-TAB**：左右上下排列，每眼保留完整高度，packed frame 总尺寸为 `render_width × 2 × render_height`
- **Anaglyph**：红蓝/红青互补色
- **Interleaved**：像素/行交错
- **Leia**：Leia光场显示格式
- **Depth Map**：RGB + 深度图输出

**B. OpenXR Full Synthesis**：
- 上传 `left_eye` / `right_eye` 到OpenXR交换链
- 使用 `XrCompositionLayerProjection` 类型

**C. OpenXR RGB+D Depth Direct**：
- 输出RGB + Depth，由Viewer Shader消费
- Viewer Shader 消费 RuntimeSettingsSnapshot 派生出的 shader uniform snapshot。该 snapshot 必须表达 `max_disparity_px`、`depth_response`、`convergence`、`render_size`、`screen_roll` 等规范语义；viewer 不得把 IPD、depth_strength、stereo_scale、max_shift_ratio 当作 normalized-depth 强度链重新解释。

#### 4.11.3 行业依据

- **OpenXR 1.1规范**：
  - `XrCompositionLayerProjection`：表示从每只眼的视点使用透视投影渲染的平面投影图像
  - 支持 `XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO` 主立体显示配置
  - Quad Layer Shape在所有OpenXR运行时上均受支持
- **ISO/IEC TR 23090-27:2025**：定义了渲染系统应用的端到端互操作性用例
- **SMPTE ST 2070**：立体3D在MXF中的存储格式
- **ITU-R BT.2025**：3DTV节目制作与交换标准

#### 4.11.4 前沿技术说明

**选型方案**：采用**统一输出抽象层（Unified Output Abstraction Layer）** ，支持：
1. 所有主流立体格式的动态切换
2. OpenXR原生集成（Full Synthesis模式）
3. 灵活扩展新格式（插件化架构）

**淘汰方案对比**：

| 淘汰方案 | 淘汰理由 |
|---------|---------|
| 仅单一输出格式（如仅SBS） | 缺乏灵活性，无法适配多种显示设备 |
| 硬编码OpenXR输出 | 无法支持非XR应用场景（如裸眼3D屏） |
| 无RGB+D Direct模式 | 无法利用Viewer端Shader的灵活性 |


## 5. 整体约束与一致性规则

### 5.1 分辨率一致性

| 约束项 | 规则 |
|--------|------|
| 工作分辨率 | 所有中间产物必须对齐 `render_size` |
| 输入分辨率 | `capture_size` 可任意，但进入合成前必须统一 |
| 输出分辨率 | 按输出格式规范，左右眼各自为 `render_size` |

### 5.2 数据类型一致性

| 数据类型 | 规格 |
|---------|------|
| RGB | 8-bit per channel, sRGB |
| Depth | float tensor，normalized / relative depth，range 与 near/far 方向必须可追踪 |
| Disparity | 32-bit float 或等效 tensor，像素单位 |
| Mask | boolean / float / 8-bit 均可，但语义必须是 occlusion / disocclusion 风险区 |
| Temporal State | 与 left/right eye 或内部历史状态对齐，`render_size` 变化时必须重置 |

### 5.3 坐标系一致性

- **图像坐标**：左上角原点，(0,0)在左上角
- **深度坐标**：near/far 方向由 depth provider 与 depth_response 约定，不得由下游阶段猜测。
- **视差坐标**：`disparity_px` 是左右眼总视差预算下的像素位移场；展示层的正负方向必须通过 eye generation / presentation contract 明确。

### 5.4 时序一致性

- 帧序列必须保持时间戳单调递增
- 时域状态在场景切换时必须重置
- 分辨率变化时必须重置时域状态

### 5.5 视差控制约束（新增）

- `max_disparity_px` 完全由软件层按 `render_size` 和 Parallax Budget 档位解析，不依赖任何物理测量值。
- normalized-depth 路径不得把 `IPD`、`stereo_scale`、`depth_strength`、`max_shift_ratio` 作为核心强度链。
- 应用层可根据具体显示设备特性选择不同 `parallax_budget_preset`，但管线内部不做硬件参数解析。


## 6. 测试与验证建议

### 6.1 单元测试

| 测试项 | 验证方法 | 通过标准 |
|--------|---------|---------|
| 分辨率解析 | 输入各种capture_size，验证render_size输出 | 符合4.2节规则 |
| 图像缩放 | 对比缩放前后分辨率 | 精确匹配render_size |
| 深度估计 | 验证 provider 输出尺寸、dtype、range metadata、timing/provider_info | `depth_render.shape[-2:] == render_size`，并可追踪 provider 信息 |
| 视差预算计算 | 输入不同 `render_size` / preset / aspect，验证 `max_disparity_px` | 符合4.6.2节预算表、插值和超宽保护规则 |
| 视差计算 | 验证公式 `disparity_px = depth_response × max_disparity_px` | 数值误差 < 0.01px |
| 立体扭曲 | 验证左右眼位移方向与幅度 | left_shift = +d/2, right_shift = -d/2 |

### 6.2 集成测试

| 测试项 | 验证方法 | 通过标准 |
|--------|---------|---------|
| 端到端延迟 | 从capture到输出的总耗时 | < 33ms（30 FPS）/ < 16ms（60 FPS） |
| 分辨率一致性 | 检查所有中间buffer | 全部为render_size |
| 格式兼容性 | 测试所有输出格式 | 所有格式正常输出 |
| OpenXR兼容性 | 在主流OpenXR运行时测试 | 正常显示立体图像 |
| Parallax Budget 可调性 | 调整 `parallax_budget_preset`，观察视差变化 | `max_disparity_px` 随 preset 单调变化 |

### 6.3 主观质量评估

| 评估维度 | 方法 | 参考标准 |
|---------|------|---------|
| 立体感 | 用户评分（1-5分） | 平均分 > 3.5 |
| 空洞伪影 | 专家评审 | 无明显可见空洞 |
| 时域稳定性 | 视频回放评审 | 无明显闪烁/抖动 |
| 视觉舒适度 | 用户舒适度问卷 | 无不适报告率 > 90% |
| Parallax Budget 偏好 | 用户偏好测试 | 多数用户选择 comfort / standard / strong 中的稳定档位 |

### 6.4 性能基准

| 平台 | 目标分辨率 | 目标帧率 | 测试条件 |
|------|-----------|---------|---------|
| 桌面端（RTX 4060 或同级） | 1080p/1440p/4K固定档位 | 以实际 depth backend 和 synthesis mode 记录 | 本地 / OpenXR / stream 分路径测试 |
| 高端桌面端（RTX 4080/4090 或同级） | 4K / 3K / 2K 固定档位 | 以实际 headset / viewer / stream 目标记录 | OpenXR full synthesis 与 RGB+depth direct 分开测试 |
| 移动/边缘端 | 1080p 或更低固定档位 | 只在对应 provider 实现后声明 | 不得引用未接入 provider 的论文 FPS 作为项目性能指标 |

### 6.5 标准符合性验证

- **ISO/IEC TR 23090-27:2025**：验证渲染系统媒体架构的符合性
- **OpenXR 1.1**：使用OpenXR Conformance Test Suite验证
- **T/UWA 035-2025**：验证裸眼3D系统参考架构的符合性


## 7. 参考文献

### 7.1 标准与规范

1. ISO/IEC TR 23090-27:2025. Information technology — Coded representation of immersive media — Part 27: Media and architectures for render-based systems and applications. ISO/IEC JTC 1/SC 29, 2025.
2. T/UWA 035-2025. 基于双目视差的裸眼3D系统参考架构与通用技术要求. 世界超高清视频产业联盟, 2025.
3. Khronos OpenXR 1.1 Specification. The Khronos Group Inc.
4. SMPTE ST 2070-1. Stereoscopic 3D in MXF — Common Provisions. SMPTE.
5. ITU-R BT.2025. 1280 × 720 digital image systems for the production and international exchange of 3DTV programs for broadcasting. ITU-R.

### 7.2 学术论文

6. Cheng, Z., et al. "RTS-Mono: A Real-Time Self-Supervised Monocular Depth Estimation Method for Real-World Deployment." arXiv:2511.14107, 2025.
7. Cheng, J., et al. "MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors." CVPR 2025.
8. "StereoFG: Generating Stereo Frames from Centered Feature Stream." SIGGRAPH Asia 2025.
9. "PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching." NeurIPS 2025.
10. "GenStereo: Towards Open-World Generation of Stereo Images and Unsupervised Matching." arXiv, 2025.
11. "DEFOM-Stereo: Depth Foundation Model Based Stereo Matching." arXiv, 2025.
12. "Depth-guided Hole-filling Algorithm for View Synthesis." KCI, 2025.
13. "AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting." CVPR 2025.
14. "CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach." arXiv, 2025.
15. "Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency." arXiv e-prints, May 2025.
16. "Glasses-free 3D display with ultrawide viewing range using deep learning." Nature, 2025.
17. Kim, S. et al. "Perceptual Disparity Limits for Stereoscopic 3D Content Based on Viewing Distance and Screen Size." IEEE Trans. on Visualization and Computer Graphics, vol. 28, no. 5, 2022.
18. Wang, J. et al. "Content-adaptive Disparity Control for Stereoscopic Video." ACM Trans. on Graphics (Proc. SIGGRAPH), 2024.
19. "Real-time Disparity Control for Monocular Depth-based 3D Synthesis." IEEE VR 2025 Workshop, 2025.


## 附录A：缩略语表

| 缩略语 | 全称 |
|--------|------|
| DIBR | Depth-Image-Based Rendering |
| FPS | Frames Per Second |
| HMD | Head-Mounted Display |
| MDE | Monocular Depth Estimation |
| MPEG | Moving Picture Experts Group |
| SBS | Side-By-Side |
| SMPTE | Society of Motion Picture and Television Engineers |
| SOTA | State-of-the-Art |
| TAB | Top-And-Bottom |
| TAA | Temporal Anti-Aliasing |
| UWA | Ultra High-Definition Video Industry Alliance |
| XR | Extended Reality |

---