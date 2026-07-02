# VisionDepth3D 架构与功能对比报告

## 结论

VisionDepth3D 是一个偏“创作套件”的 2D-to-3D 工具，核心优势在离线视频生产工作流、丰富的 3D 调参控件、FPS 插帧/超分增强、深度图生产与融合、短片段预览测试和最终视频导出。

Desktop2Stereo 当前更偏“实时运行时产品”，核心优势在 OpenXR Link、实时桌面捕获、多后端深度运行时、模型 artifact 准备、Flet GUI、Windows 启动器和运行状态管理。

两者不是同一类产品，但 VisionDepth3D 有大量值得借鉴的工程设计。最值得我们照抄或改造吸收的是：

1. 离线 3D 生成工作流：源视频 + 深度视频 + 预览 + 短片段测试 + 编码输出。
2. Live stereo controls：把复杂 3D 参数组织成可实时调节的创作控件。
3. FPS / Upscale Enhancer：抽帧、RIFE 插帧、ESRGAN 超分、输出视频配置、硬件编码。
4. 任务队列/进度反馈：进度、FPS、ETA、日志、暂停/恢复/取消。
5. 线程化视频增强管线：RIFE、ESRGAN、写视频并行，用 buffer 保持离线输出帧顺序。
6. 深度稳定策略：Temporal depth normalization、EMA depth smoothing、subject/convergence EMA。
7. 离线质量优先设计：clip range 测试、preview modes、edge repair quality、depth blending。

关于“VisionDepth 是否支持输出缓存”：

- 它确实有多处缓存/缓冲概念。
- FPS/Upscale 的 threaded pipeline 明确使用 frames buffered and synchronized 来保证 RIFE/ESRGAN/写视频并行时输出顺序正确；这是离线转换缓存，不是实时显示缓存。
- VideoDepthAnything 模型内部有 temporal cache / sliding window，用于时序深度推理。
- render_3d 有 grid cache、depth/convergence EMA 等运行中缓存。
- Live 输出更接近实时 MJPEG/virtual camera 推送，不是我们前面设计的 presentation delay buffer。

因此，VisionDepth3D 可以证明“离线视频生产工作流里用缓存换吞吐/质量”是合理路线，但它的缓存实现不能直接等同于我们要做的 OpenXR/本地输出延迟显示缓存。我们的实时高画质缓冲只能吸收它的保序思想，不能照搬它的离线缓存方式，更不能把实时画面缓存到硬盘。

## 源码位置

本次对比源码：

```text
E:\AI_2D_to_3D\4.LC700X_Desktop2Stereo\VisionDepth3D
```

关键文件：

```text
VisionDepth3D/app.py
VisionDepth3D/UserGuide.md
VisionDepth3D/core/vd3d_live.py
VisionDepth3D/core/merged_pipeline.py
VisionDepth3D/core/render_3d.py
VisionDepth3D/core/render_depth.py
VisionDepth3D/services/depth_service.py
VisionDepth3D/ui/pages/fps_upscale_page.py
VisionDepth3D/ui/queue_dock.py
VisionDepth3D/core/models/video_depth_anything/video_depth_stream.py
```

本项目关键对比文件：

```text
4k-stereo-synthesis-lab/src/gui/gui.py
4k-stereo-synthesis-lab/src/app_runtime/runtime_callbacks.py
4k-stereo-synthesis-lab/src/stereo_runtime/pipeline.py
4k-stereo-synthesis-lab/src/stereo_runtime/model_artifacts.py
4k-stereo-synthesis-lab/src/stereo_runtime/model_registry.py
4k-stereo-synthesis-lab/src/xr_viewer/core_source_state.py
4k-stereo-synthesis-lab/src/xr_viewer/openxr_runtime.py
4k-stereo-synthesis-lab/src/utils/queue_utils.py
```

## 顶层架构对比

| 维度 | VisionDepth3D | Desktop2Stereo |
|---|---|---|
| 产品定位 | 视频 2D-to-3D 创作套件 | 实时桌面/OpenXR 2D-to-3D 运行时 |
| GUI 技术 | PySide6 | Flet |
| 主入口 | `app.py` 创建 QApplication、Splash、Controller、Services、MainWindow | `src/gui/gui.py` + launcher + runtime 子进程 |
| 架构风格 | controller/service/page/core 分层，保留较多 legacy core 函数 | GUI、app_runtime、stereo_runtime、xr_viewer、capture、streaming 分层 |
| 实时输出 | VD3D Live preview、HTTP MJPEG、virtual camera | OpenXR Link、本地窗口、MJPEG/RTMP legacy stream |
| 离线输出 | 完整：深度视频、3D 视频、FPS/超分、编码、clip range | 当前弱项，未来待做 |
| 模型后端 | CUDA 优先，DirectML/ROCm/CPU 说明较多，但实现更偏 PyTorch 应用层 | 模型 registry + ONNX + TensorRT + MIGraphX + 多后端 artifact 准备 |
| 用户工作流 | 先生成 depth，再调 3D，再短片测试，再最终渲染 | 选择参数后直接运行实时输出 |
| 质量调参 | 非常丰富，偏创作者 | 当前更偏运行时必要参数 |
| 日志/任务 | Job Queue、progress、FPS、ETA、copy/clear log | Flet 原生日志面板、状态栏、模型下载/转换进度 |

## VisionDepth3D 的代码架构

### 入口和服务组装

`app.py` 使用 PySide6，启动时先显示 splash，再延迟导入重模块：

```text
QApplication
-> QSplashScreen
-> AppState / DepthState
-> SettingsService
-> RenderService
-> DepthService
-> PreviewService
-> PresetService
-> LiveService
-> AppController
-> MainWindow
```

这个设计和我们最近对 `run_windows.bat` / GUI 启动顺序的要求方向一致：先让用户看到 GUI 或 splash，再加载重模块。

可借鉴点：

- 启动早期反馈明确。
- 重模块导入放在 splash 之后。
- AppState / DepthState 与 service 分离，便于离线任务长期运行。

不建议照搬点：

- PySide6 技术栈不适合直接并入我们的 Flet GUI。
- 其 services 里有不少 legacy/Tk 兼容 proxy，说明代码历史包袱较重。

### GUI 页面结构

VisionDepth3D 的用户功能按页面组织：

```text
Depth Estimation Tab
Depth Blender Tab
3D Generator Tab
FPS / Upscale Enhancer Tab
VD3D Live
Job Queue / Progress / Log
```

它的 UI 是创作软件思路：每个页面对应一个完整工作流，而不是只给一组实时参数。

我们当前 GUI 更像运行控制台：参数区 + 运行状态 + log。未来做离线转换时，建议新增独立页面或 tab，而不是把所有离线功能塞进当前实时参数页。

推荐我们未来 GUI 分区：

```text
实时模式
  - OpenXR Link
  - 本地实时预览
  - 高画质缓冲输出

离线转换
  - 输入视频
  - 深度生成
  - 3D 生成
  - FPS/超分增强
  - 输出视频配置
  - 短片段测试

任务中心
  - 当前任务
  - 队列
  - 进度/FPS/ETA
  - 日志
```

## VD3D Live 对比

UserGuide 中的 VD3D Live 面向实时 2D-to-3D，典型流程是：

```text
Capture source
-> Depth Anything model
-> Pixel Shift CUDA pipeline
-> SBS output
-> preview / HTTP MJPEG / virtual camera
```

### Live 控件

UserGuide 中 Live stereo controls 包括：

```text
Capture FPS
Infer W / Infer H
Depth FPS
Smooth (EMA + median)
EMA alpha
Enable SBS 3D
FG shift
MG shift
BG shift
Show preview window
HTTP stream host:port
Virtual camera
VCam FPS
Audio device
Audio delay ms
```

推荐 Live preset 示例：

```text
Capture FPS: 30
Depth FPS: 4 to 6
Smooth Depth: On
Foreground Shift: -6.0
Midground Shift: -0.8
Background Shift: +2.2
Max Pixel Shift: 0.020 to 0.030
Parallax Balance: 0.70
Depth Pop Gamma: 1.05 to 1.15
Subject Tracking: Off for testing, On for stability
Dynamic Convergence: On
Edge Masking: On
Floating Window: Off for testing, On if edge violations appear
```

### 值得我们借鉴的部分

1. 把实时 3D 参数分层：捕获、深度、3D/Pixel Shift、预览/输出、音频。
2. 前景/中景/背景 shift 比单一 depth strength 更容易被创作者理解。
3. Depth FPS 与 Capture FPS 分离，允许低频深度 + 高频显示。
4. Smooth/EMA 是明确的用户可见控制，便于解释“稳定 vs 延迟”。
5. Virtual camera / HTTP stream 是简单外部输出路线。

### 不适合照搬的部分

1. VisionDepth3D Live 主要是 MJPEG/virtual camera/preview 输出，不是 OpenXR 原生输出。
2. 它的 live output 不解决 VR swapchain、head pose、OpenXR frame pacing。
3. 它的音频 monitor 仍偏 ffplay/DirectShow 参数，不适合我们规划的通用音频同步层。
4. 它的 Live 控件很多，如果原样塞进我们 GUI，会破坏当前实时模式的简洁性。

### 对我们的建议

我们应照抄“控件组织思想”，不要照搬输出实现。

建议在我们 GUI 中把实时调参升级为两层：

```text
基础模式：
  Depth Strength
  Convergence
  Screen Distance
  Quality Preset
  High Quality Buffer checkbox

高级模式：
  Capture FPS / Target FPS
  Depth FPS
  Temporal smoothing
  FG/MG/BG shift
  Max pixel shift
  Parallax balance
  Dynamic convergence
  Edge masking
  Floating window
```

内部仍映射到我们现有 `RuntimeSettingsSnapshot`、OpenXR runtime config、stereo runtime 参数，不直接采用 VisionDepth3D 的参数命名。

## VisionDepth3D 的缓存/缓冲实现

这里必须区分两类缓存：

```text
VisionDepth3D 离线转换缓存：允许使用磁盘帧、depth video、输出视频中间结果，目标是保序、复用和吞吐。
Desktop2Stereo 实时高画质缓冲：禁止画面帧落盘，目标是 OpenXR/本地实时输出的低抖动延迟播放。
```

我们的实时缓冲优先级必须是：

```text
GPU resident buffer -> system memory ring buffer -> 失败
```

如果 GPU 缓存和内存缓存都撑不住目标 frame window，就应直接判定高画质缓冲不可用，提示用户降低延迟、分辨率或补洞质量。不能自动退化成硬盘帧缓存。

### 1. FPS/Upscale threaded pipeline 的帧缓冲

UserGuide 明确说明 threaded pipeline：

```text
One thread generates interpolated frames (RIFE)
One thread upscales frames (ESRGAN)
One thread writes frames to the output video
Frames are buffered and synchronized to maintain correct ordering.
```

这是一种典型离线生产管线：

```text
read frame pair
-> RIFE interpolation worker
-> ESRGAN upscale worker
-> ordered writer
-> output video
```

它的 buffer 目标是提升离线处理吞吐并保持输出顺序，不是为了 OpenXR 低延迟显示，也不是实时 presentation delay buffer。

对我们的借鉴：

- 离线转换必须用 ordered frame buffer，不能简单多线程乱序写视频。
- 离线转换可以使用磁盘帧、depth cache、临时视频文件，因为它不承担实时 OpenXR presentation。
- 实时高画质缓冲只能使用 GPU resident buffer，其次 system memory ring buffer；如果必须依赖硬盘缓存才能运行，应判定失败。
- 任务状态必须包含 frame index、completed/total、FPS、ETA。
- 稳定模式和高性能线程模式可以并存：
  - Merged Pipeline：低内存、稳定。
  - Threaded Pipeline：高吞吐、占用更多内存。

### 2. VideoDepthAnything 模型内部 temporal cache

`core/models/video_depth_anything/video_depth_stream.py` 中 `VideoDepthAnything` 维护：

```text
frame_id_list
frame_cache_list
cached_hidden_state_list
sliding window
```

`infer_video_depth_one()` 首帧会初始化缓存，后续帧会拼接历史 cached hidden states，再更新 sliding window。

这说明 VisionDepth3D 已经利用模型内部时序缓存来稳定视频深度。这不是输出缓存，但对我们“高画质缓冲输出”非常有启发：如果未来用户选择 Video Depth Anything 或其它视频模型，应优先利用模型原生 temporal cache，而不是只在外部做 RGB/depth 后处理。

### 3. render_3d 中的运行时缓存和 EMA

`core/render_3d.py` 包含：

```text
get_base_grid_cached(H, W, device, dtype)
TemporalDepthFilter
DepthPercentileEMA
ConvergenceEMA
SubjectDepthEMA
```

这些属于图像生成中的缓存/平滑：

- grid cache 避免每帧重复创建采样网格。
- depth EMA 降低深度跳动。
- percentile EMA 稳定 depth range。
- convergence/subject EMA 稳定舒适度。

这与我们当前实时模式的 latest-frame 队列不同。我们可以吸收这些算法策略，但要放入 `stereo_runtime` 内部，而不是 presentation queue。

### 4. Live MJPEG 只保留最新 JPEG

`core/vd3d_live.py` 的 MJPEG 输出逻辑更像：

```text
push_bgr(frame)
-> encode jpeg
-> replace self._jpeg
-> /video.mjpg loop yield latest jpeg
```

这和我们的 `put_latest()` 思路接近：实时输出保最新帧，不保证每帧都显示。

因此，VisionDepth3D Live 并没有实现我们设想的“延迟播放 presentation buffer”。真正接近缓存输出的是离线 threaded pipeline 和视频深度模型 temporal cache。

## FPS / Upscale Enhancer 对比

VisionDepth3D 的 FPS/Upscale Enhancer 是非常完整的离线增强模块。

UserGuide 描述其能力：

- Extract Frames from Video。
- Configure Output Video。
- RIFE interpolation，支持 2x / 4x / 8x。
- Real-ESRGAN upscaling。
- PySceneDetect 自动分割长视频。
- 硬件编码重建输出视频。
- 进度、FPS、ETA、日志、取消。

`ui/pages/fps_upscale_page.py` 的结构是：

```text
页面收集 settings
-> _start_standard() 调 core.merged_pipeline.start_merged_pipeline
-> _start_threaded() 调 core.merged_pipeline.start_threaded_pipeline
-> 后台 thread 执行
-> _TkProgressProxy 把 legacy progress 转成 Qt signal
```

这套结构对我们未来离线转换非常有价值。

### 我们应借鉴的模块边界

建议未来新增：

```text
src/offline/video_jobs.py
src/offline/frame_extract.py
src/offline/fps_interpolation.py
src/offline/upscale.py
src/offline/video_encode.py
src/offline/job_queue.py
src/gui/offline_page.py
```

不要把离线转换逻辑塞进 `stereo_runtime/pipeline.py`。实时管线和离线管线的调度目标不同：

```text
实时：优先最新帧、低延迟、允许丢帧
离线：按帧完整处理、顺序输出、不可丢帧
```

### 可直接照抄的产品设计

1. “Extract Frames from Video” 单独按钮。
2. “Configure Output Video” 单独区域。
3. RIFE 和 ESRGAN 可以独立开关，也可以同时启用。
4. 输出分辨率、原始 FPS、目标 FPS、编码器、容器格式都在 GUI 中明确配置。
5. 提供两种管线：稳定模式与高性能线程模式。
6. 输出前允许用户先跑短片段。

### 不建议照抄的实现

1. 不建议用 Tk proxy 兼容层；我们应设计 Flet 原生 progress event。
2. 不建议直接把 `core/merged_pipeline.py` 的巨型函数结构搬进来；我们应拆成小模块。
3. 不建议一开始就支持太多模型；先打通 FFmpeg 抽帧/编码和一个插帧/超分 backend。

## 3D Generator / 离线 3D 生成对比

VisionDepth3D 的 3D Generator 是我们未来离线转换最应该参考的部分。

它的用户工作流是：

```text
输入源视频
-> 生成或选择深度视频
-> 选择输出格式
-> 预览调参
-> 短 clip 测试
-> 全片渲染
```

UserGuide 强调的关键能力：

- Preview Modes：Anaglyph、Shift Heatmap、Overlay Arrows。
- Screen Plane Offset / zero parallax。
- Subject Lock。
- Dynamic Convergence。
- Foreground Curvature。
- Edge Repair Quality。
- Layered Depth and Background Depth。
- Codec Presets。
- VR180 Output Settings。
- Clip Range Rendering。

### 对我们的离线转换设计建议

我们未来的离线转换不应只是“把实时输出录下来”，而应是独立生产管线：

```text
Input video
-> Frame extraction / decode
-> Depth generation or load depth video
-> Optional depth stabilization / blending
-> Stereo synthesis
-> Preview selected frames
-> Clip range render
-> Full encode with audio mux
```

第一版建议优先做：

```text
输入视频
深度模型选择
输出 SBS 格式
输出分辨率/FPS/编码器
预览当前帧
Clip range 测试
最终视频渲染
```

第二版再加：

```text
Depth video cache
Depth blending
Dynamic convergence
Subject lock
Edge repair quality
FPS/upscale enhancer
VR180
```

## Depth Estimation / Depth Blender 对比

VisionDepth3D 把深度生产作为独立步骤，而不是实时管线里的隐式过程。

优点：

- 深度视频可以重复使用。
- 用户可以检查 depth 是否稳定。
- 可以用两个 depth source 做 blending。
- 长视频可以先稳定 depth，再做 3D。

我们当前实时模式不需要这个流程，但离线转换强烈建议采用。

建议未来新增离线 depth cache：

```text
outputs/depth_cache/<source_hash>/<model>/<resolution>/depth.mkv
outputs/depth_cache/<source_hash>/<model>/<resolution>/frames/*.png
outputs/depth_cache/<source_hash>/metadata.json
```

metadata 至少包含：

```text
source_path
source_duration
source_fps
source_width/source_height
model_id
inference_resolution
depth_normalization
frame_count
created_at
```

这会让离线 3D 调参变快：用户改 convergence/shift/edge repair 时，不需要重新跑深度模型。

## Job Queue / Progress 对比

VisionDepth3D 的 `ui/queue_dock.py` 提供：

```text
Job Queue title
ProgressBar
Status label
Telemetry label
Log list
Copy Log
Clear Log
max_log_lines = 500
```

`services/depth_service.py` 会解析 legacy status：

```text
56/7188 | FPS: 1.2 | ETA: 01:43:18
```

并转成结构化 progress payload：

```text
progress
status_text
elapsed
eta
fps_like
rate_label
```

我们已经有 Flet 原生日志窗口和 D2S_PROGRESS 思路，但离线任务中心还不完整。

建议我们离线任务统一使用：

```text
JobProgressEvent:
  job_id
  stage
  completed
  total
  percent
  fps
  eta_seconds
  elapsed_seconds
  status_text
  severity
```

实时 log 面板可以继续轻量；离线任务中心需要更像 job queue。

## 模型与后端对比

### VisionDepth3D

UserGuide 中支持路线：

```text
NVIDIA CUDA
Windows AMD/Intel DirectML
Linux AMD ROCm
CPU fallback
FFmpeg hardware encoders: NVENC / AMF / QSV / CPU
```

实现上更偏 PyTorch 应用层和创作工具集成。模型包含 Depth Anything、Video Depth Anything、Depth Anything 3、RIFE、Real-ESRGAN 等。

### Desktop2Stereo

我们当前更强的是模型 artifact 和运行时后端：

```text
ModelRegistry
ONNX artifact
TensorRT engine
MIGraphX graph
CUDA/ROCm/XPU/CPU 方向
OpenXR runtime
```

`model_artifacts.py` 已有通用 artifact 准备层，能根据模型、尺寸、dtype、backend 检查/生成 `.onnx`、`.trt`、`.mgx`。

### 对我们的建议

离线转换不应绕开现有 `ModelRegistry` 和 `model_artifacts.py`。未来离线 depth generation 应复用同一套模型选择、下载、ONNX/TRT/MIGraphX artifact 准备逻辑。

但离线视频增强模型如 RIFE/ESRGAN 可以先单独建 registry：

```text
DepthModelRegistry
StereoRuntimeArtifactRegistry
EnhancementModelRegistry
```

避免把深度模型和插帧/超分模型混在一个列表里。

## 双方优劣项

### VisionDepth3D 优势

1. 离线创作工作流完整。
2. UserGuide 很成熟，用户知道每一步该做什么。
3. 3D Generator 参数丰富，适合电影/视频调参。
4. FPS/Upscale Enhancer 已经覆盖抽帧、插帧、超分、编码。
5. 有稳定管线和线程高性能管线两种思路。
6. 支持 short clip range，避免全片试错成本高。
7. 深度图可以独立生成、保存、混合、复用。
8. Live 控件面向创作者，FG/MG/BG shift 等控制更直观。
9. 任务进度包含 FPS/ETA，离线任务体验好。
10. 有 temporal depth normalization 和多类 EMA 稳定策略。

### VisionDepth3D 劣势

1. 实时输出架构偏 MJPEG/virtual camera，不是 OpenXR 原生。
2. GUI 与核心有 legacy/Tk proxy 痕迹，架构不够纯净。
3. 部分 core 文件功能很大，维护成本高。
4. 多后端 artifact 准备不如我们系统化。
5. 对 VR/OpenXR swapchain、head pose、frame pacing 没有我们项目深入。
6. Live 缓存不是 presentation delay buffer，不能直接解决 OpenXR 高画质缓冲。
7. 音频 monitor 更偏外部 ffplay/DirectShow，不适合作为通用音频同步层。

### Desktop2Stereo 优势

1. OpenXR Link 是核心优势，目标更清晰。
2. 运行时分层较适合长期维护：GUI、app_runtime、stereo_runtime、xr_viewer。
3. 模型 registry 和 artifact 准备层已经比较系统。
4. TensorRT/MIGraphX/ONNX 等部署路径更适合实时产品。
5. Flet GUI 与启动器已围绕用户运行状态优化。
6. 已有日志面板、进度条、模型准备检测和下载逻辑。
7. latest-frame 策略适合低延迟实时场景。

### Desktop2Stereo 劣势

1. 离线转换能力基本缺失或尚未产品化。
2. 缺少源视频 -> 深度视频 -> 3D 视频的完整工作流。
3. 缺少 FPS 插帧、超分、编码配置等创作工具。
4. 实时参数暴露还不如 VisionDepth3D 的 Live controls 细致。
5. 缺少 depth cache、depth blending、clip range test。
6. 高画质缓冲输出还停留在设计阶段。
7. 音频同步和独立音频采集还没有实现。

## 哪些功能值得我们照抄

### 第一优先级：离线转换工作流

建议照抄产品流程，不照抄代码：

```text
Extract Frames from Video
Generate Depth Video
Preview 3D
Clip Range Render
Configure Output Video
Render Final Video
```

这是我们未来“本地模式输出”和“离线转换”的基础。

### 第一优先级：Live stereo controls 的组织方式

建议引入高级模式：

```text
Depth FPS
Temporal smoothing
FG/MG/BG shift
Max pixel shift
Parallax balance
Depth pop gamma
Subject tracking
Dynamic convergence
Edge masking
Floating window
```

这些参数不要一次性全放默认界面，应放到高级面板或专家模式。

### 第一优先级：任务进度标准

照抄思想：

```text
progress percent
frames completed / total
FPS
ETA
stage status
cancel/pause/resume
copy log
```

我们可以把它统一到 Flet 原生任务中心。

### 第二优先级：FPS / Upscale Enhancer

建议未来做成独立离线增强页：

```text
RIFE interpolation
ESRGAN upscaling
hardware encoder
threaded pipeline
stable merged pipeline
```

第一版可先只实现 frame extraction + encode + one enhancement backend。

### 第二优先级：Depth cache 和 depth blending

离线转换中深度生成是昂贵步骤，必须可复用。

建议先做 depth cache，再做 depth blending。

### 第二优先级：Preview modes

VisionDepth3D 的 Anaglyph、Shift Heatmap、Overlay Arrows 非常适合调参。

我们未来本地离线转换至少应支持：

```text
SBS preview
Anaglyph preview
Depth preview
Shift heatmap
```

### 第三优先级：VR180 / advanced output formats

这属于后续功能，不建议第一阶段做。

## 哪些不建议照抄

1. 不照搬 PySide6 GUI。
2. 不照搬 legacy Tk proxy。
3. 不照搬巨型 core pipeline 文件结构。
4. 不把 MJPEG/virtual camera 当作 OpenXR 高画质缓冲实现。
5. 不把 RTMP/ffplay 风格音频 monitor 当作通用音频同步方案。
6. 不把所有高级参数一次性暴露到基础 GUI。

## 对高画质缓冲输出的启发

VisionDepth3D 证明了三件事：

1. 视频工作流可以接受缓存和延迟，只要输出顺序正确、进度清晰。
2. 时序深度模型内部 cache 能显著提升视频深度稳定性。
3. 离线/高画质路径应该和实时 latest-frame 路径分开。

我们前一份 `33-quality-buffered-output-feasibility-report.md` 的方向仍然成立：

```text
实时模式：latest-frame，低延迟，允许丢帧。
高画质缓冲模式：frame window，延迟输出，lookahead 补洞。
离线转换：完整逐帧处理，严格保序，不丢帧。
```

VisionDepth3D 的 threaded pipeline 只能作为离线保序 buffer 的参考；VideoDepthAnything temporal cache 可作为深度模型内部缓存参考；Live MJPEG latest frame 则说明实时输出不应强行保每帧。Desktop2Stereo 的实时高画质缓冲必须坚持 GPU 优先、内存次之、不能落盘，无法满足则判失败。

## 建议的 Desktop2Stereo 后续路线

### 阶段 1：离线转换页面骨架

新增 GUI 页面：

```text
离线转换
  输入视频
  输出目录/输出文件
  深度模型
  输出格式 SBS/Anaglyph/Depth only
  编码器
  Clip range
  开始/暂停/恢复/取消
```

内部先实现：

```text
FFmpeg/OpenCV decode
Depth generation using existing runtime/model registry
Stereo synthesis using existing stereo_runtime where possible
FFmpeg encode
Job progress events
```

### 阶段 2：Depth cache

新增可复用 depth cache：

```text
source_hash + model + resolution + settings -> depth video / depth frames
```

离线调参时优先复用 depth cache。

### 阶段 3：预览和短片测试

实现：

```text
single frame preview
short clip render
SBS/anaglyph/depth/shift heatmap preview
```

这是减少用户试错成本的关键。

### 阶段 4：高画质缓冲输出

把前一份可行性报告中的 presentation buffer 做成实时可选模式：

```text
Quality Buffered Output checkbox
delay ms
lookahead frames
hole fill quality
```

注意它不同于离线转换：它仍是实时输出，只是允许延迟。

### 阶段 5：FPS/Upscale Enhancer

参考 VisionDepth3D：

```text
Extract Frames from Video
RIFE interpolation
ESRGAN upscaling
Configure Output Video
Threaded high-performance pipeline
Stable merged pipeline
```

建议先独立于实时 runtime 实现。

### 阶段 6：独立音频同步

不要复用 RTMP 音频采集。

新增：

```text
AudioCapture
AudioClock
AudioDelayBuffer
AudioOutput
Mux audio into offline output
Sync audio delay with quality buffer delay
```

Windows 优先调研 WASAPI loopback。

## 推荐优先级表

| 优先级 | 功能 | 借鉴 VisionDepth3D 的内容 | 对我们价值 |
|---|---|---|---|
| P0 | 离线转换页面 | 3D Generator workflow | 补齐产品短板 |
| P0 | Job progress event | progress/FPS/ETA/log | 改善长任务体验 |
| P0 | Clip range render | 短片段测试 | 降低试错成本 |
| P1 | Depth cache | depth video / save frames | 避免重复跑模型 |
| P1 | Live advanced controls | FG/MG/BG、Depth FPS、EMA | 提升实时调参能力 |
| P1 | Preview modes | Anaglyph/Heatmap/Depth | 提升调参效率 |
| P2 | FPS/Upscale | RIFE/ESRGAN/编码 | 提升离线创作能力 |
| P2 | Threaded pipeline | buffered ordered stages | 提升吞吐 |
| P2 | Depth blending | 多 depth source 融合 | 提升画质 |
| P3 | VR180 | 输出格式扩展 | 后续高级功能 |

## 最终建议

VisionDepth3D 最值得我们学习的不是某个单点算法，而是“视频创作工作流”的完整性。它把深度生成、深度稳定、3D 调参、短片预览、最终渲染、插帧、超分、编码、任务进度组织成一个连续流程。

Desktop2Stereo 不应放弃实时/OpenXR 优势去复制 VisionDepth3D，而应吸收它的离线创作能力，形成双模式产品：

```text
实时模式：Desktop2Stereo 当前强项，低延迟 OpenXR / 本地实时输出。
高画质缓冲模式：可选延迟，服务视频观看，提升补洞质量。
离线转换模式：完整视频生产工作流，服务最终高质量 SBS/3D 视频输出。
```

短期最该做的是离线转换骨架和任务进度系统；中期加入 depth cache、preview modes、clip range；长期再加入 FPS/upscale enhancer、depth blending、VR180 和完整音频同步。
