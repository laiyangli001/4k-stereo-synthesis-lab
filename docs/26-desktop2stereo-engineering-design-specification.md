# Desktop2Stereo 工程设计规范

日期：2026-06-25

本文在 `docs/28-Realtime-2d-to-3d-specification.md` 的正式最终运行时流程规范基础上，对照当前工程代码，定义 Desktop2Stereo 的完整工程设计规范。本文关注模块职责、现有实现路径、硬件加速边界、数据契约、热更新、调试与后续演进。历史 `docs/25-2d-to-3d-runtime-specification.md` 已作废，仅作为背景记录。

## 文档定位与 docs/28 对齐方式

`docs/28-Realtime-2d-to-3d-specification.md` 是正式最终运行时流程规范，负责定义目标语义和十一阶段处理流程。本文不重复定义另一套运行时规范，而是按 docs/28 的流程顺序记录当前工程实现、迁移边界、兼容清理项和验证责任。

写作规则：

```text
1. docs/28 决定应该做什么，以及每个阶段的输入/输出语义。
2. docs/26 记录当前工程由哪些模块实现这些阶段，以及哪些兼容路径还没有清理。
3. 若 docs/26 和 docs/28 的运行时语义冲突，以 docs/28 为准，并回写 docs/26。
4. docs/00-api-handoff-progress.md 只记录当前推进状态、任务清单和验证结果，不承载详细工程规范。
```

## 设计目标

Desktop2Stereo 的核心目标是把桌面、窗口、图片或视频源转换为可实时观看的立体输出，并支持本地 3D 显示器、网络推流和 OpenXR 头显。

工程目标：

```text
1. 输入源统一为 RGB frame + capture metadata。
2. 2D-to-3D runtime 统一使用 render_size 坐标系。
3. normalized depth 路径统一使用 parallax budget，而不是物理 IPD 强度链。
4. OpenXR、本地显示、网络推流共享同一 runtime 参数语义。
5. GPU 数据路径优先避免不必要的 CPU 往返，但必须显式标注当前实现是否真正零拷贝。
6. GUI 只负责配置、启动、热参数保存和用户交互，不直接修改 runtime 内部对象。
7. 所有输出路径必须可 debug、可定位、可回归测试。
```

## 当前代码模块地图

| 子系统 | 主要文件 | 当前职责 |
|---|---|---|
| GUI | `src/gui/*.py` | Flet UI、配置收集、settings.yaml 读写、启动/停止子进程、热参数保存 |
| 应用运行上下文 | `src/app_runtime/*.py` | 创建 queues、runtime、OpenXR state、capture config、callbacks、cleanup |
| Capture | `src/capture/*.py`, `src/capture/backends/*.py` | 显示器/窗口捕捉、事件式或 polling 式 capture runner、`CapturedFrame` metadata、raw_q 输入 |
| Runtime adapter | `src/stereo_runtime/adapter.py` | Desktop2Stereo settings -> `StereoRuntimeConfig`，模型/输出/参数归一化 |
| Depth providers | `src/stereo_runtime/depth_provider.py`, `src/stereo_runtime/providers/*` | PyTorch/ONNX/TensorRT/MIGraphX/ROCm/MPS/XPU 深度推理 |
| Model artifacts | `src/stereo_runtime/model_registry.py`, `model_artifacts.py` | 模型 ID 映射、Hugging Face 下载、ONNX/TensorRT/MIGraphX artifact 路径 |
| Stereo synthesis | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `layers.py` | RGB+depth -> left/right/sbs，遮挡、mask、补洞、temporal |
| Runtime pipeline | `src/stereo_runtime/runtime.py`, `pipeline.py` | 每帧 depth、synthesis、OpenXR result、timing、debug_info |
| Output packing | `src/stereo_runtime/output.py`, `output_triton.py`, `output_convert.py` | SBS/TAB/anaglyph/interleaved/leia/depth_map 打包与 numpy 转换 |
| Local viewer | `src/viewer/viewer.py`, `viewer_runtime.py`, `gl_texture_uploader.py` | GLFW/ModernGL 本地显示、SBS/TAB/Anaglyph/Interleaved/Leia、共享 CUDA/GL texture upload 后端 |
| OpenXR viewer | `src/xr_viewer/*.py` | OpenXR session、swapchain、D3D11/OpenGL render、controller/environment、runtime eye 上传 |
| Streaming | `src/streaming/*.py` | MJPEG legacy streaming、RTMP config/legacy hooks |
| Bench/test tools | `scripts/tools/*`, `tests/*` | 4K benchmark、visual regression、provider/runtime/OpenXR tests |

## 顶层运行流程

当前 GUI 启动方式：

```text
Flet GUI
-> 收集控件值
-> 写入 src/settings.yaml
-> 启动 python -u -X faulthandler src/main.py 子进程
-> pump 子进程 stdout 到控制台
-> Stop 时写 STOP_REQUEST_FILE 或结束进程
```

当前 runtime 主数据流：

```text
CaptureSessionLoop
-> raw_q(maxsize=1)
-> RuntimePipelineLoop
-> runtime_q(maxsize=1)
-> local viewer / network streamer / OpenXR viewer
```

队列规范：

```text
raw_q 只保留最新 capture frame，旧帧可丢弃。
runtime_q 只保留最新 runtime result，避免渲染端积压。
丢帧优先于增加交互延迟。
CUDA runtime 后段慢于 capture cadence 时，必须按 GPU 完成节奏推进 latest frame，不能无限异步排队。
```

## docs/28 流程到工程实现映射

| docs/28 阶段 | 当前工程实现 | 当前状态 / 迁移边界 |
|---|---|---|
| 步骤 1：Capture Input | `src/capture/session.py`, `src/capture/runners.py`, `src/capture/types.py`, `src/capture/backends/*` | 已有 `CapturedFrame` metadata、copy mode、source/capture size 记录；真实 CUDA/ROCm zero-copy 仍需硬件验证 |
| 步骤 2：Resolve Render Size | `src/stereo_runtime/render_size.py`, `src/stereo_runtime/pipeline.py`, GUI Render Scale 配置 | 已按 4K/3K/2K/1K 固定 scale 档位解析；4K级输入按比例缩放并保持输入宽高比；历史 numeric / short alias Render Scale 输入已清理并有测试防回归 |
| 步骤 3：Resize RGB To Render Size | `src/capture/preprocess.py`, `src/stereo_runtime/pipeline.py` | 已在 pipeline 边界把 capture frame 解析到 render_size；跨设备 fallback 仍需继续扩展验证矩阵 |
| 步骤 4：Depth Estimation | `src/stereo_runtime/depth_provider.py`, `src/stereo_runtime/depth_onnx_provider.py`, `src/stereo_runtime/providers/*`, `model_registry.py` | 已支持多 provider / backend；provider 内部尺寸可以不同，但输出必须回到 render_size；实时 provider 主路径不得 CPU numpy 往返，TensorRT/MIGraphX 需区分 GPU event timing 和 CPU enqueue timing |
| 步骤 5：Depth Postprocess | `src/stereo_runtime/runtime.py`, `src/stereo_runtime/synthesis.py` | 当前使用 normalized / relative depth 处理和轻量 postprocess；不能改写为 metric Z 路径 |
| 步骤 6：Resolve Parallax Budget | `src/stereo_runtime/parallax.py`, `adapter.py`, `settings_snapshot.py` | 已有预算表、短边插值和超宽保护；legacy multiplier 只应留在显式兼容入口 |
| 步骤 7：Disparity Field | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `openxr_render.py` | 主语义为 `depth_response * max_disparity_px * depth_strength`；near/far、response 和 depth_strength 必须在 metadata / shader uniforms 中可追踪 |
| 步骤 8：Stereo Warp | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `layers.py`, OpenXR shader paths | full synthesis 已有 baseline/layered DIBR；OpenXR D3D11 direct shader 仍需追平 OpenGL direct 的核心 DIBR 质量语义 |
| 步骤 9：Mask and Hole Fill | `src/stereo_runtime/synthesis.py`, `layers.py` | 已有 edge-aware / directional fill 路径；AI inpainting 只作为未来离线/质量候选 |
| 步骤 10：Temporal Stabilization | `src/stereo_runtime/runtime.py`, `pipeline.py`, `settings_snapshot.py` | 已有 scene/render_size/source/settings reset 机制；temporal 只做稳定，不改变 parallax budget |
| 步骤 11：Output Pack / Viewer Upload | `src/stereo_runtime/output.py`, `output_triton.py`, `src/viewer/*`, `src/xr_viewer/*`, `src/streaming/*` | 本地/网络消费 packed output，OpenXR 支持 full synthesis 与 RGB+depth direct；OpenXR/本地窗口共享 CUDA/GL texture uploader；D3D11 direct parity 和 RTMP/低延迟 stream 是后续工程 |

## GUI / Flet 规范

当前 GUI 基于 Flet，主要职责分布：

| 文件 | 职责 |
|---|---|
| `src/gui/gui.py` | GUI 类组装入口 |
| `src/gui/builders.py` | 控件布局和分组构建 |
| `src/gui/controls.py` | 控件 helper |
| `src/gui/config_mgr.py` | settings 应用、收集、热参数保存 |
| `src/gui/handlers.py` | 控件事件、可见性、设备变化、模式联动 |
| `src/gui/localization.py` | 中英文 UI 文案 |
| `src/gui/process.py` | 子进程启动/停止、日志、状态栏、ESC 长按停止 |
| `src/gui/devices.py` | 设备枚举、平台优化器可见性 |
| `src/gui/capture_sources.py` | 显示器/窗口源选择 |

GUI 设计规则：

```text
1. GUI 控件值是 user-facing，内部 key 必须保持稳定。
2. GUI 不直接调用 StereoRuntime.process_rgb_frame()。
3. GUI 不直接修改 StereoRuntime.stereo_config 或 OpenXRViewer 属性。
4. GUI 启动时写完整 settings.yaml；运行中热参数变更可写 settings.yaml，由 runtime hot reload 读取。
5. 所有平台不可用选项必须隐藏或禁用，不能写入非法组合。
6. Flet 控件显示值和内部枚举值必须通过 config_mgr 显式转换。
```

当前已实现的热参数保存路径：

```text
on_stereo_hot_param_change()
-> _schedule_stereo_hot_save()
-> _save_stereo_hot_params()
-> 更新 settings.yaml 中 Stereo Preset / Stereo Quality / Convergence / Depth Strength / Parallax Budget / Temporal / Hole Fill / Edge 等规范字段
```

后续规范要求：

```text
settings.yaml hot save 是持久化配置写入路径，不承担线程间实时共享语义。
运行时热更新语义收敛到 RuntimeSettingsSnapshot + settings_update_q。
GUI 写 settings.yaml 后，runtime 由 StereoHotReloader 读取规范字段；不再接受 IPD / Stereo Scale / Max Shift Ratio 作为兼容强度入口。
```

## 配置与 settings.yaml 契约

当前配置入口：

```text
src/settings.yaml
-> src/main.py 读取
-> create_runtime_context()
-> runtime_config_from_d2s_settings()
-> StereoRuntimeConfig
-> StereoRuntime
```

关键字段分组：

| 分组 | 字段示例 | 说明 |
|---|---|---|
| Capture | `Capture Mode`, `Monitor Index`, `Window Title`, `Capture Tool` | 输入源与捕捉后端 |
| Runtime target | `Run Mode` | Local Viewer / 3D Monitor / Stream / OpenXR Link 等 GUI 运行目标 |
| Depth model | `Depth Model`, `Depth Resolution`, `Model List` | 模型选择和推理尺寸 |
| Device/backend | `Computing Device`, `TensorRT`, `MIGraphX`, `CoreML`, `OpenVINO`, `torch.compile`, `FP16` | 平台加速选择 |
| Stereo | `Stereo Preset`, `Stereo Quality`, `Synthetic View`, `Depth Strength`, `Convergence`, `Parallax Budget Preset`, `Max Disparity Px` | 当前规范立体参数；Depth Strength 是用户连续强度 gain，Parallax Budget 是档位预算，Convergence 是零视差平面 |
| Synthesis postprocess | `Temporal Strength`, `Scene Reset Threshold`, `Edge Dilation`, `Mask Feather Radius`, `Hole Fill Mode`, `Edge Threshold`, `Foreground Scale`, `Anti-aliasing` | 稳定、mask、补洞、depth postprocess |
| Output | `Display Mode`, `Anaglyph Method`, `Cross Eyed`, `Fill 16:9`, `Fix Viewer Aspect`, `VSync` | 本地/推流封装与显示 |
| OpenXR | `XR Preview Window`, `XR Headset Model`, `Controller Model`, `Environment Model` | OpenXR viewer 行为；头显型号解析推荐观看距离并按 60° 水平视角自动计算屏幕尺寸 |
| Streaming | `Stream Protocol`, `Streamer Port`, `Stream Quality`, `Stream Key`, `Stereo Mix`, `CRF`, `Audio Delay` | 网络推流配置 |

规范要求：

```text
1. settings.yaml 是持久化配置，不是线程间实时共享对象。
2. runtime 内部使用规范化后的 dataclass，不直接依赖原始 YAML 字段名。
3. normalized-depth 路径只读取 parallax budget / max_disparity_px 规范字段；IPD / Stereo Scale / Max Shift Ratio 不再作为配置、热更新或 adapter 兼容入口。
4. 保存配置时不得清空用户未显示的有效字段。
```

## Capture 子系统设计

### Capture 抽象

当前抽象文件：

```text
src/capture/types.py
src/capture/factory.py
src/capture/session.py
src/capture/runners.py
```

核心契约：

```text
CaptureConfig:
    output_resolution
    fps
    window_title
    capture_mode
    monitor_index
    capture_tool
    os_name

CaptureRunner.run(...):
    on_frame(captured_frame: CapturedFrame)
```

CaptureSessionLoop 当前负责：

```text
1. 创建 runner。
2. 接收 `CapturedFrame`，兼容包装旧三元组输入。
3. 首帧打印 raw size、capture size 和 target size。
4. 写入 raw_q，若队列满则覆盖旧帧。
5. capture paused / hard idle 时清 raw queue。
```

### Windows capture 后端

当前 Windows 后端：

| Capture Tool | 文件 | 类型 | 说明 |
|---|---|---|---|
| `WindowsCapture` | `windows_capture_event.py` | event runner | 使用 `windows_capture.WindowsCapture` 事件回调 |
| `WindowsCaptureCUDA` | `windows_capture_event.py` | event runner | 使用 `wc_cuda.WindowsCapture`，CUDA 插件路径 |
| `WindowsCaptureROCm` | `windows_capture_event.py` | event runner | 使用 `wc_rocm.WindowsCapture`，AMD/ROCm 插件路径 |
| `DesktopDuplication` | `windows_desktop_duplication.py` | polling source | DXGI Desktop Duplication session |
| default Windows fallback | `windows_dxcamera.py` | polling source | DXCamera 风格后备路径 |

事件式 WindowsCapture 路径：

```text
create_capture_runner()
-> WindowsCaptureEventRunner
-> _load_windows_capture(capture_tool)
-> cap = WindowsCapture(window_name=...) 或 WindowsCapture(monitor_index=...)
-> @cap.event on_frame_arrived(frame, control)
-> raw = frame.frame_buffer.copy() 或 frame.frame_buffer.clone()
-> CapturedFrame(raw, capture_size, timestamp, copy_mode, frame_raw_device, metadata)
-> on_frame(captured_frame)
```

零拷贝/少拷贝规范：

```text
WindowsCaptureCUDA / WindowsCaptureROCm 是零拷贝候选 capture backend。
当前 event runner 仍会对 frame.frame_buffer 调 copy() 或 clone()，因此不能把当前全链路宣称为严格 zero-copy。
如果 frame_buffer 是 GPU tensor，clone/copy 应尽量保持 device-to-device，不应退回 CPU。
如果 frame_buffer 是 CPU numpy，后续 runtime preprocess 才负责上传到 GPU。
```

必须记录的 capture metadata：

```text
capture_tool
capture_mode
monitor_index
window_title / hwnd
capture_size
frame_raw_type
frame_raw_device
frame_raw_dtype
copy_mode: none / clone / copy / cpu_numpy / gpu_tensor
```

后续优化目标：

```text
1. 对 WindowsCaptureCUDA 定义并验证 GPU tensor contract，避免 CPU numpy 中转。
2. 对 WindowsCaptureROCm 定义并验证 ROCm tensor contract，避免 CPU numpy 中转。
3. 继续补充真实硬件 zero-copy / device-to-device copy 验证，只有实测成立后才标记 `zero_copy=True`。
4. runtime preprocess 根据 CapturedFrame metadata / frame_raw.device 选择 CUDA/ROCm/CPU 路径并记录 transfer。
```

### macOS capture 后端

当前 macOS 后端：

```text
src/capture/backends/macos_coregraphics.py
src/capture/backends/macos_screencapturekit.py
```

规范：

```text
CoreGraphics 作为兼容路径。
ScreenCaptureKit 作为较新系统的高性能路径。
macOS MPS depth provider 可用时，capture 输出仍需明确 CPU->MPS 上传边界。
```

### Linux capture 后端

当前 Linux 后端：

```text
src/capture/backends/linux_mss.py
```

规范：

```text
Linux 当前以 MSS/CPU capture 为主。
后续如接入 PipeWire/DMABUF，应按 capture metadata 标注 zero-copy eligibility。
```

## Capture preprocess 与 device contract

当前 preprocess 模块：

```text
src/capture/preprocess.py
src/capture/preprocess_triton.py
```

规范目标：

```text
输入 frame_raw 可以是 numpy、CPU torch tensor、CUDA torch tensor、ROCm torch tensor。
输出 render_rgb 必须是 B/C/H/W 或 C/H/W RGB tensor，range 0..1，device 与 depth provider 对齐。
```

颜色格式规则：

```text
capture 后端必须声明原始格式：RGB / BGR / BGRA / RGBA / NV12。
runtime 前必须统一为 RGB。
BGR/BGRA-to-RGB 属于 host/capture preprocess 职责，不属于 depth provider 职责。
```

GPU 上传规则：

```text
CPU numpy -> torch tensor -> target device。
CUDA tensor -> CUDA runtime 直接消费。
ROCm tensor -> ROCm runtime 直接消费。
若目标 depth backend 与 capture device 不同，必须显式 device transfer 并记录。
```

## 模型下载与 artifact 管理

当前实现：

```text
ModelRegistry.default()
DepthModelSpec(name, model_id, family)
resolve_model_dir(model_id, cache_dir)
artifact_paths_for_model()
prepare_model_artifacts()
```

模型缓存目录规则：

```text
cache_dir / models--{org}--{repo}
```

下载规则：

```text
prepare_model_artifacts()
-> 先查当前后端可执行的本地 artifact
-> 再查本地权重文件：model.safetensors / model.pt / model.ckpt
-> 本地权重存在则本地导出 ONNX / 构建后端 artifact
-> 本地 artifact 和权重都不存在时才探测 Hugging Face endpoint 并下载
-> ensure_model_downloaded()
-> huggingface_hub.snapshot_download(...) 或 hf_hub_download(...)
```

endpoint 规则：

```text
默认顺序： https://hf-mirror.com -> https://huggingface.co
如果用户显式设置 HF_ENDPOINT，则只使用该 endpoint。
下载前必须记录模型下载 URL，并通过轻量 HEAD / Range GET 探测记录 HTTP 状态、大小、类型、重定向后 URL 或连接错误。
两个默认 endpoint 都不可访问时，必须停止运行并提示用户检查网络或开启 VPN 后重试。
不得在启动阶段做 Google 网络探测；网络检查只发生在缺少本地 artifact 和本地权重、确实需要下载模型时。
```

artifact 命名：

```text
model_fp16_{height}x{width}.onnx
model_fp32_{height}x{width}.onnx
model_fp16_{height}x{width}.trt
model_fp16_{height}x{width}.mgx
```

artifact 后端归属：

```text
ONNX / XPU / CPU / generic ONNXRuntime 路径：只把 .onnx 视为可执行 artifact。
TensorRT / NVIDIA CUDA native engine 路径：只把 .trt 视为可执行 artifact，缺失时从匹配 dtype 的 ONNX 构建。
MIGraphX / AMD ROCm 路径：只把 .mgx 视为可执行 artifact，缺失时从 ONNX 构建。
.trt 不能让 ONNX/XPU/CPU/ROCm 路径跳过 ONNX 或权重检查。
.mgx 不能让 CUDA/TensorRT/ONNX/XPU/CPU 路径跳过对应 artifact 或权重检查。
```

尺寸规则：

```text
Distill/Depth-Anything 系列默认 patch multiple = 14。
InfiniDepth 系列默认 patch multiple = 16。
export_size_for_model() 负责对 export_height/export_width 取最近合法倍数。
```

规范要求：

```text
1. GUI 只选择模型名称和 depth resolution。
2. runtime 通过 ModelRegistry 解析 model_id。
3. 下载、ONNX 导出、TensorRT/MIGraphX engine 构建由 model_artifacts/provider 层负责。
4. local_files_only=True 时不得静默联网；ONNX 导出内部加载模型也必须继承 local_files_only。
5. force_download/force_rebuild 必须由 GUI 显式选项或命令行显式指定。
6. artifact 检测必须与当前 depth backend / GPU 类型绑定，不能用任意本地产物作为所有后端的 cache hit。
7. MIGraphX 准备必须在确认 MIGraphX 可用后执行；不可破坏 unsupported ROCm 环境回退 PyTorch ROCm 的路径。
```

## Depth provider 设计

当前 provider factory：

```text
src/stereo_runtime/depth_provider.py:create_depth_provider()
```

支持后端：

| Backend | 文件/类 | 平台 | 说明 |
|---|---|---|---|
| `pytorch_cuda` | `DistillAnyDepthBase518`, `GenericAutoDepthProvider`, `InfiniDepthProvider` | NVIDIA CUDA | PyTorch/Transformers 路径 |
| `onnx_cuda` | `providers/nvidia/onnx_cuda.py`, `depth_onnx_provider.py` | NVIDIA CUDA | ONNX Runtime CUDA，可用 IOBinding/DLPack |
| `tensorrt_native` | `providers/nvidia/tensorrt_native.py` | NVIDIA CUDA | Native TensorRT engine |
| `tensorrt_ort` | `providers/nvidia/tensorrt_ort.py` | NVIDIA CUDA | ORT TensorRT provider |
| `pytorch_rocm` | `providers/amd/pytorch_rocm.py` | AMD ROCm | PyTorch ROCm |
| `migraphx_rocm` | `providers/amd/migraphx.py` | AMD ROCm | MIGraphX graph，可 fallback PyTorch ROCm |
| `pytorch_mps` | `providers/apple/pytorch_mps.py` | Apple Silicon | PyTorch MPS |
| `pytorch_xpu` | `providers/intel/pytorch_xpu.py` | Intel XPU | PyTorch XPU |

Depth provider contract：

```text
predict_profile(rgb) -> DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)
```

Depth 输出规则：

```text
depth 必须是 normalized / relative depth。
depth 必须最终对齐 render RGB 的 H/W。
depth provider 可以内部 resize 到模型输入尺寸，但返回 runtime 前必须 upsample 到 render_size。
depth range 和 near/far 方向必须在 debug_info/provider_info 中可追踪。
```

ONNX CUDA 规范：

```text
ModelOnnxPreprocessor:
    RGB 0..1 -> resize 到模型 input_size -> normalize -> dtype
ONNX Runtime:
    CUDAExecutionProvider 必须实际启用，否则报错。
IOBinding:
    优先把 output 绑定到 CUDA，减少 CPU 往返。
DLPack:
    只有 use_iobinding 时允许作为进一步优化。
```

AMD/ROCm 规范：

```text
ROCm capture backend 和 ROCm depth backend 应尽量保持同 device。
MIGraphX artifact 使用 .mgx。
MIGraphX ROCm7 构建优先尝试 FP8 autocast，失败再回退 FP16；force-FP32 模型不得进入 FP8/FP16 量化。
MIGraphX 失败时是否 fallback PyTorch ROCm 必须在 provider_info.fallback_reason 中记录。
```

实时 provider 零拷贝规则：

```text
TensorRT / MIGraphX / ONNX Runtime 等实时 provider 不得在主路径把 CUDA/ROCm tensor 转 CPU numpy 再回 GPU。
如后端限制导致 CPU 回传，必须红色控制台告警并记录 provider_info / fallback_reason。
TensorRT native / MIGraphX 等后端应优先记录 CUDA/HIP event timing，区分真实 GPU preprocess/model/postprocess 时间和 CPU enqueue 时间。
```

## StereoRuntime 处理路径

当前核心类：

```text
src/stereo_runtime/runtime.py:StereoRuntime
```

主要方法：

```text
process_rgb_frame(rgb_frame) -> StereoRuntimeResult(
    depth,
    left_eye,
    right_eye,
    sbs,
    output_eye_size,
    output_display_size,
    output_format,
    output_dtype,
    output_pack_backend,
    debug_info,
    timing,
    provider_info,
)
process_openxr_frame(rgb_frame, openxr_config) -> OpenXRRuntimeResult(
    depth,
    left_eye,
    right_eye,
    source_rgb,
    output_eye_size,
    output_display_size,
    output_format,
    output_dtype,
    output_pack_backend,
    debug_info,
    timing,
    provider_info,
)
openxr_result_from_stereo_result(stereo_result) -> OpenXRRuntimeResult
```

普通 full synthesis 路径：

```text
rgb_frame
-> depth_provider.predict_profile()
-> synthesize_stereo(rgb, depth, stereo_config, temporal_state)
-> optional runtime uint8 pack
-> StereoRuntimeResult
```

OpenXR direct 路径：

```text
rgb_frame
-> depth_provider.predict_profile()
-> _prepare_openxr_rgb_depth(depth)
-> OpenXRRuntimeResult(source_rgb=rgb_frame, depth=prepared_depth, left_eye=rgb_frame, right_eye=rgb_frame, shader_uniforms=max_disparity_px/depth_strength/convergence/render_size/screen_roll)
-> runtime_output_format = openxr_rgb_depth
```

OpenXR full synthesis 路径：

```text
rgb_frame
-> process_rgb_frame()
-> openxr_result_from_stereo_result()
-> OpenXRRuntimeResult(left_eye, right_eye, depth, source_rgb=None)
-> runtime_output_format = openxr_full_synthesis_eyes
```

Structured output contract：

```text
output_eye_size: (width, height)
output_display_size: (width, height)
output_format
output_dtype
output_pack_backend
timing
provider_info
```

Compatibility debug contract:

```text
debug_info["runtime_output_format"]
debug_info["runtime_output_dtype"]
debug_info["runtime_output_eye_size"]
debug_info["runtime_output_display_size"]
debug_info["runtime_output_pack_backend"]
debug_info["runtime_depth_backend"]
debug_info["cuda_memory_*"] when enabled
```

## Stereo synthesis 设计

当前合成入口：

```text
src/stereo_runtime/synthesis.py:synthesize_stereo()
```

后端：

| Backend | 说明 |
|---|---|
| `fast` | baseline shift 快速路径 |
| `fast_plus` | baseline + occlusion mask + directional/edge fill，部分可 fused Triton |
| `quality_4k` | layered synthesis 质量路径 |
| `hq_4k` | 更高层数/更重质量路径 |

当前规范位移参数：

```text
depth_strength
convergence
parallax_budget_preset
max_disparity_px
```

shader 派生参数：

```text
depth_response(depth, convergence)
max_disparity_px
depth_strength
disparity_px = depth_response * max_disparity_px * depth_strength
left_shift_px = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

迁移规则：

```text
当前：parallax_budget resolver 已实现，并把 preset / max_disparity_px 作为 normalized-depth 主语义。
当前：legacy 字段仍可读取，必须通过 adapter / legacy preset 映射或隔离，不作为默认核心强度链。
后续：清理 legacy 字段、debug-only 兼容键和旧测试入口，只保留显式 legacy adapter。
```

## Mask 与 hole fill 规范

当前模块：

```text
src/stereo_runtime/occlusion.py
src/stereo_runtime/occlusion_triton.py
src/stereo_runtime/hole_fill.py
src/stereo_runtime/hole_fill_triton.py
```

当前 GUI hole fill mode 映射：

| GUI 模式 | 内部值 | 技术 | 默认参数 |
|---|---|---|---|
| 柔和 / 低重影 | `soft_low_ghost` | edge_aware_fill | radius=1, strength=0.6 |
| 均衡 / 标准 | `balanced` | edge_aware_fill | radius=3, strength=1.0 |
| 锐利 / 高细节 | `sharp_test` | edge_aware_fill | radius=1, strength=1.0 |
| 内容感知 / 最高质量 | `quality` | directional_edge_aware_fill | radius=3, strength=1.0 + directional/content-aware |

规范：

```text
mask 标记需要修复的位置，不负责修复。
hole fill 根据 mask 修补空洞，不负责兜底过大的视差预算。
OpenXR 实时默认建议 balanced。
静态图/导出可以使用 quality。
```

## Output packing 规范

当前实现：

```text
src/stereo_runtime/output.py:make_sbs()
src/stereo_runtime/output_triton.py
```

支持格式：

```text
mono
half_sbs
full_sbs
half_tab
full_tab
anaglyph
interleaved
leia
depth_map
```

打包规则：

```text
full_sbs: cat left/right on width。
half_sbs: each eye area-resize to half width then cat。
full_tab: cat left/right on height。
half_tab: each eye area-resize to half height then cat。
anaglyph: channel composite。
interleaved: row interleave。
leia: column interleave / device-specific output。
depth_map: depth repeated to RGB channels。
```

Triton 规则：

```text
sbs_backend() 根据 output_format、fused、tensor support、env disable flag 选择 Triton 或 torch fallback。
Triton 不可用时必须 fallback，不应中断普通输出。
Debug info 必须记录 sbs_backend。
```

## 本地 viewer 设计

当前实现：

```text
src/viewer/viewer.py:StereoWindow
```

技术栈：

```text
GLFW window
ModernGL context
OpenGL texture upload
optional CUDA PBO upload
shader display modes: SBS/TAB/Depth/Anaglyph/Interleaved/Leia/Mono
```

CUDA/GL upload 当前规则：

```text
共享 CudaGlTextureUploader 负责 CUDA tensor -> OpenGL texture。
优先 CUDA/GL image texture copy；image texture 不可用或失败时允许 PBO GPU fallback。
CPU upload 只允许作为显式 fallback，必须红色控制台告警并记录失败原因。
离散 GPU 可使用 pinned host staging / PBO 降低上传开销；集成 GPU 避免不必要 pinned staging overhead。
```

规范：

```text
本地 viewer 接收 packed frame，不重新定义立体公式。
本地 viewer 可以做 display mode shader/presentation，但不能改变 synthesis 的 max_disparity_px 语义。
CUDA/GL image texture 与 PBO 都是 upload 后端，不是立体算法；PBO 是 fallback 时必须标记为 fallback。
glGenerateMipmap 不是 CUDA 功能；实时 viewer/OpenXR 上传路径默认不得每帧生成 mipmap，除非实际使用 mip level 采样并已验证收益。
```

## OpenXR viewer 设计

当前模块：

```text
src/xr_viewer/base.py
src/xr_viewer/openxr_runtime.py
src/xr_viewer/d3d11_native_renderer.py
src/xr_viewer/core_* modules
src/xr_viewer/environment*.py
```

OpenXR runtime 启动：

```text
runtime_q.get() 首帧
-> frame_size_from_runtime_result()
-> OpenXRViewer(frame_size=(width,height), depth_q=runtime_q, ...)
-> viewer.run(first_runtime_result=...)
```

OpenXR 两条渲染路径：

| 路径 | runtime_output_format | 输入 | viewer 职责 |
|---|---|---|---|
| RGB+depth direct | `openxr_rgb_depth` | source_rgb + prepared depth | viewer shader 根据 depth 现场生成双眼 |
| Full synthesis eyes | `openxr_full_synthesis_eyes` | left_eye + right_eye | viewer 直接上传左右眼纹理 |

D3D11 native renderer：

```text
render_eye(): RGB+depth shader path。
render_runtime_eye(): runtime eye texture path，depth_srv=None。
CUDART_D3D11 支持 cudaGraphicsD3D11RegisterResource + cudaMemcpy2DToArrayAsync device-to-device copy。
```

OpenGL texture upload path：

```text
runtime_direct_opengl_image_texture 优先用于 CUDA tensor -> OpenGL image texture copy。
runtime_direct_opengl_pbo 是 GPU fallback，不代表 image texture 成功。
CPU GL upload 只允许作为显式 fallback，必须红色控制台告警。
debug/log 必须记录 upload backend、fallback reason、eye texture size。
```

规范：

```text
OpenXR full synthesis 不接受 SBS/TAB 作为 viewer 输入。
OpenXR full synthesis 的 runtime 应优先产出 OpenXR 可直接上传的 uint8/RGBA CUDA tensor，viewer 只负责 texture upload 和 present。
OpenXR debug 可额外导出 SBS 预览，但不作为 swapchain 输入。
viewer 建屏优先使用结构化 `output_display_size`，legacy `debug_info["runtime_output_display_size"]` 只作为兼容 fallback；eye texture 优先使用结构化 `output_eye_size`。
Direct path 的 legacy uniforms 必须由 adapter 从规范参数转换。
OpenXR `xr_wait` / `xr_poll` / `xr_submit` / present 时间属于设备运行时调度，不得混同为 StereoRuntime depth/synthesis/SBS 生成耗时。
```

### OpenXR 虚拟屏幕、头显预设与 OSD 规范

头显屏幕预设属于 OpenXR presentation 层，不改变 runtime 立体合成语义。GUI 只保存 `XR Headset Model`，运行时通过 `src/utils/xr_headset_presets.py` 解析设备推荐观看距离，并按统一水平视角计算屏幕尺寸。

当前规则：

```text
XR_HEADSET_HORIZONTAL_FOV_DEG = 60.0
screen_width_m = distance_m * 2 * tan(radians(60.0) / 2)
screen_height_m = screen_width_m * 9 / 16
diagonal_in = hypot(screen_width_m, screen_height_m) / 0.0254
```

规范要求：

```text
1. 头显预设只维护设备型号、分类和推荐观看距离；不得手工维护每个预设的宽度/高度/英寸派生值。
2. 推荐距离为无穷远的设备在当前 viewer 中取 20.0 m 作为实用距离；例如 Pico 4 / 4 Ultra 和 HTC VIVE XR Elite。
3. OpenXR screen width clamp 必须允许 20.0 m / 60° 预设生成的 23.09 m 宽屏幕；当前上限为 30.0 m。
4. 距离显示必须使用头部位置到屏幕中心的真实欧氏距离 `_screen_view_distance()`，不得混用内部 `screen_distance` 投影值作为用户可见观看距离。
5. Y 短按恢复默认屏幕预设，Y 长按切换预设；应用预设时必须重置 preset OSD key，使恢复同一 preset 也能重新显示 OSD。
6. preset OSD 显示 5.0 s，live distance 只更新显示文本，不得进入触发 key 以免头部微动持续刷新倒计时。
7. 虚拟键盘与屏幕下边缘的间距按屏幕高度 15% 计算。
8. 屏幕边缘吸附释放角为 6°。
9. 屏幕与键盘的激光命中光圈共用同一 cursor-ring model，并按 eye-to-hit distance 缩放。
```

手柄屏幕操作规则：

```text
Left grip: 平移屏幕，保持距离和朝向。
Right grip: 保留 sphere-orbit drag，围绕头部移动屏幕中心；移动后自动更新 yaw/pitch 让屏幕朝向头部。
D2S_OPENXR_RIGHT_GRIP_SCREEN_ROTATION: 默认关闭，只控制右手柄 wrist roll 是否映射到 screen_roll；不得用该开关阻止 right-grip orbit 后的自动 yaw/pitch 朝向头部。
Both grips: 继续作为双手系统移动/调整路径，不由单手规则替代。
```

### OpenXR 手柄光照规范

手柄光照属于 OpenXR viewer 的 presentation 层，不改变 runtime 立体合成语义。屏幕灯光是所有手柄模型共用的统一光源；Pico、Quest、Valve、YVR 等模型只提供 mesh、texture、profile offset 和按钮动画，不各自定义独立灯光模型。

环境 profile 使用 `controller_hdr_lighting` 控制手柄 HDR 环境反射，`controller_hdr_reflection` 仅作为兼容别名：

```text
controller_hdr_lighting = true:
    启用 HDR panorama 环境采样。
    关闭手柄顶灯补光。
    手柄主要由屏幕灯光 + HDR diffuse/specular + 屏幕镜面反射共同照亮。

controller_hdr_lighting = false:
    禁用 HDR 环境采样。
    启用手柄顶灯补光。
    手柄主要由屏幕灯光 + 基础环境光 + 顶灯补光 + 屏幕镜面反射共同照亮。
```

当前建议值：

| 项 | 建议值 | 说明 |
|---|---:|---|
| 基础环境光 | `baseColor * 0.30` | 只保留最低可见度，避免背光面发白 |
| 屏幕漫反射主光 | `1.00 * u_screen_light_intensity` | 主光源，使用屏幕采样颜色 `screen_tint` |
| 屏幕方向项 | `pow(max(dot(N, screen_light_dir), 0.0), 0.75)` | 保留方向性，朝向屏幕的表面更亮 |
| 顶灯位置 | `u_camera_pos + vec3(0.0, 0.45, -0.18)` | 近似头顶/面板上方补光，只在 HDR 关闭时启用 |
| 顶灯亮度 | `0.40 * top_fill` | 辅助补光，不能盖过屏幕主光 |
| 顶灯颜色 | `vec3(0.95, 0.97, 1.0)` | 轻微冷白，避免染色过重 |
| HDR diffuse mix | `0.36` | HDR 开启时给材质低频环境色 |
| HDR specular | `0.30 * u_env_intensity` | HDR 开启时给镜面反射强度 |
| 屏幕镜面混合 | `mix(baseColor * screen_tint, screen_col, 0.72)` | 以真实屏幕颜色为主 |
| 屏幕镜面强度 | `(0.38 + 0.95 * fresnel) * u_screen_light_intensity` | 斜视角更明显，正视角保留基础反射 |
| 输出限制 | `clamp(color, 0.0, 1.0)` | 维持 LDR swapchain 输出稳定 |

规范要求：

```text
1. 屏幕灯光和屏幕镜面反射必须对所有 controller model 共用。
2. `max(dot(N, light_dir), 0.0)` 方向项必须保留，不能退回无方向全亮 ambient。
3. HDR 开启时不得叠加顶灯，避免 HDR 反射和固定白光同时抬亮背光面。
4. HDR 关闭时必须保留顶灯，补偿无 panorama IBL 时手柄表盘过暗的问题。
5. 新环境 profile 必须显式写 `controller_hdr_lighting`；`.hdr` panorama 可自动开启，但 profile 显式值优先。
```

## 网络推流设计

当前实现主要是 MJPEG legacy stream：

```text
src/streaming/mjpeg_streamer.py
src/streaming/legacy_runtime.py
```

MJPEG path：

```text
runtime_q -> runtime_output_to_numpy(runtime_result.sbs) -> MJPEGStreamer.set_frame() -> JPEG encode -> /stream.mjpg
```

MJPEG 特性：

```text
HTTP WSGI server
multipart/x-mixed-replace MJPEG stream
固定 JPEG quality
浏览器端 canvas/video auto-resize
```

RTMP 相关：

```text
src/streaming/config.py
src/streaming/rtmp.py
```

规范：

```text
network_stream 默认消费 packed frame。
编码器不可改变 synthesis 公式。
若 encoder 要求固定尺寸或 NV12/BGRA，转换发生在 transport 层，并记录 metadata。
RTMP/低延迟编码路径应复用 RuntimeSettingsSnapshot 和 Output Packing Format，不单独定义立体参数。
```

## Application Runtime Target 具体映射

| GUI Run Mode / Target | 当前/目标实现 | Runtime result | Presentation |
|---|---|---|---|
| Local Viewer | 本地 GLFW/ModernGL viewer | StereoRuntimeResult.sbs | window display |
| 3D Monitor | 本地 fullscreen 3D 显示器 | packed SBS/TAB/interleaved/leia | local_fullscreen |
| Stream | MJPEG legacy / RTMP planned | packed frame | encoded_stream |
| OpenXR Link traditional | OpenXR RGB+depth direct | OpenXRRuntimeResult(source_rgb, depth) | viewer shader |
| OpenXR Link full synthesis | runtime 完整左右眼 | OpenXRRuntimeResult(left_eye, right_eye) | eye texture upload |
| Debug Export | 文件/metadata 输出 | left/right/depth/mask/shift/sbs | file_export |

## 热更新设计

当前实现：

```text
GUI hot save -> settings.yaml
StereoHotReloader(settings_path=src/settings.yaml)
RuntimeCallbacks.apply_stereo_hot_reload_if_needed()
OpenXRStateController.update_runtime_config(snapshot/depth_strength/convergence/parallax_preset/max_disparity_px/screen_roll)
```

现状说明：

```text
当前 OpenXR hot reload 使用 RuntimeSettingsSnapshot 语义。
旧 IPD / Stereo Scale / Max Shift Ratio 兼容入口已清理。
adapter 只负责规范字段归一化，不再把旧强度链转换为 parallax budget。
```

目标路径：

```text
GUI / API
-> RuntimeSettingsSnapshot(version=N)
-> settings_update_q
-> RuntimePipelineLoop frame boundary
-> apply hot reload / rebuild / restart classification
-> runtime result debug_info 记录 active_settings_version
```

参数分级遵循 `docs/28-Realtime-2d-to-3d-specification.md` 中的 Hot Reload / Pipeline Rebuild / Session Restart 表。

RuntimeSettingsSnapshot 分级执行规则：

```text
Hot Reload 字段可由 StereoRuntime 在帧边界直接应用。
Depth provider rebuild 字段可由 StereoRuntime 重建 provider 后继续运行。
render_size_policy / stereo_render_scale / stereo_synthesis_mode / output_transport 等 pipeline-owned 字段不得在 StereoRuntime 内静默合并；必须抛出 pipeline rebuild 信号，由上层重建 pipeline/context 后再继续。
Session Restart 字段必须抛出 restart 信号，由外层重启 capture/session/viewer。
```

## 性能与资源管理

### GPU 优先级

```text
1. Capture 若能输出 GPU tensor，runtime 应保持 GPU tensor。
2. Depth provider 输出应尽量在目标 device。
3. Stereo synthesis、mask、hole fill、packing 优先 torch/Triton GPU path。
4. Viewer upload 优先共享 CUDA/GL image texture；失败才走 PBO/pinned staging；CPU upload 必须红色 fallback 告警。
5. 只有网络编码、CPU-only backend、离线保存/报告或显式诊断导出需要时才转 CPU numpy；实时主路径任何 CPU 回传都按 bug 处理。
```

### 内存与队列

```text
raw_q maxsize=1，防止 capture 堆积。
runtime_q maxsize=1，防止 render/stream 堆积。
OpenXR hard idle 时清 raw_q/runtime_q 并停止 active capture session。
render_size 变化时必须释放或重建 textures/buffers/temporal state。
```

### 实时调度与 GPU 反压

```text
实时链路以 latest-frame / low-latency 为优先目标，旧 raw frame 可被 overwrite/drop。
CUDA runtime 不得把每帧 depth/synthesis/pack GPU work 无界异步提交后继续消费下一帧。
D2S_RUNTIME_SYNC_AFTER_FRAME=auto 时，只要 runtime 使用 CUDA 后端，就在完整 runtime frame 后同步到 GPU 完成边界。
该规则由 runtime 后段决定，不由 OpenXR、Local Viewer、3D Monitor、MJPEG/RTMP 等 presentation target 决定。
```

诊断解释：

```text
capture_fps 表示 capture callback 到达速率。
runtime FPS 表示后段实际消费速率。
viewer_fps / submit_fps / present_fps 表示显示提交侧节奏，不能反推 StereoRuntime compute FPS。
raw overwrite/drop 是 producer-side latest-frame 丢帧，属于预期反压控制。
drain_drop=0 不代表没有丢帧，只表示旧帧已在 raw_q 入口被覆盖。
当 StereoRuntime total_ms 远低于显示帧间隔但 SBS/display FPS 低时，优先排查 viewer upload、runtime_q handoff、OpenXR submit/present、vsync 或 FPS 统计口径。
```

### Warmup

当前 runtime 支持：

```text
StereoRuntime.warmup_stereo_kernels_for_frame()
```

规范：

```text
首次真实 frame 后按实际 frame shape warmup。
Triton/kernel warmup 不应阻塞 GUI 主线程。
Debug log 应记录 warmup configs、resolution、elapsed_ms。
```

## Debug、日志和回归测试

每帧 result 应优先通过结构化字段暴露 host 需要消费的输出合同：

```text
output_format
output_eye_size
output_display_size
output_dtype
output_pack_backend
timing
provider_info
```

每帧 debug_info 继续保留诊断与兼容字段：

```text
application_runtime_target
runtime_quality_mode
stereo_synthesis_mode
capture_tool
capture_size
render_size
runtime_depth_backend
depth_provider_size
depth_render_size
sbs_backend
packing_format
transport
output_transport
hole_fill_backend
occlusion_mask_backend
depth_response
convergence
hole_fill_mode
edge_threshold
edge_dilation
mask_feather_radius
temporal_enabled
temporal_strength
temporal_reset_reason: scene_reset / settings_changed / render_size_changed / source_target_changed, comma-separated when multiple causes apply
active_settings_version
```

现有测试/工具覆盖方向：

```text
tests/test_runtime_pipeline.py
tests/test_runtime_openxr.py
tests/test_openxr_runtime.py
tests/test_capture_factory.py
tests/test_capture_session.py
tests/test_output_convert.py
tests/test_synthesis.py
tests/test_gui_config.py
scripts/tools/local_4k_pipeline_benchmark.py
scripts/tools/local_4k_sbs_visual_regression.py
scripts/tools/openxr_visual_regression.py
```

新增功能必须补充：

```text
1. capture metadata contract tests。
2. RuntimeSettingsSnapshot hot reload tests。
3. render_size 下采样 + max_disparity_px resolver tests。
4. OpenXR full synthesis eye size/display size regression。
5. OpenXR/local viewer CUDA/GL image texture、PBO fallback、CPU fallback warning tests。
6. network_stream packed frame resize/encoder metadata tests。
```

## 兼容清理与迁移边界

兼容清理以 docs/28 的最终运行时合同为目标；新代码默认不再增加别名、debug-only 字段或旧参数链路。历史兼容只允许留在明确命名的 import/parsing adapter 或 legacy fallback 中。

优先清理项：

| 清理项 | 目标状态 | 备注 |
|---|---|---|
| RuntimeSettingsSnapshot 字段别名 | 边界字段只保留 `parallax_budget_preset`、`temporal_enabled` 等 docs/28 名称 | `parallax_preset`、`temporal` 等旧名只允许在 legacy parser 中转换 |
| debug-only 兼容键 | host/viewer 消费结构化 result 字段 | `runtime_output_*`、flat OpenXR `openxr_*` shader-uniform debug keys 只作为短期 fallback |
| legacy parallax multiplier | normalized-depth 主路径只使用 `max_disparity_px` / `parallax_budget_preset` | `ipd_mm`、`stereo_scale`、`max_shift_ratio`、`depth_strength` 不再作为核心强度链 |
| render-size/render-scale 旧路径 | 用户侧只保留固定 scale 档位语义，并保持输入宽高比 | 删除历史 numeric scale thresholds、`native/fixed/dynamic` 用户策略语言、固定输出分辨率映射和非 4K 连续缩放行为 |
| D3D11 native direct shader | 与 OpenGL RGB+depth direct 的核心 DIBR 语义一致 | 独立 cleanup track，完成前不得删除 OpenGL-only fallback 假设 |

估算边界：代码兼容清理加自动测试约 4-8 小时 AI 时间；若同时清理更广的 parallax/debug/render-size 遗留面，约 1 个 AI 工作日。OpenXR/D3D11 真机头显验证不包含在该估算内。

## 当前实现与规范符合状态

本文以 `docs/28-Realtime-2d-to-3d-specification.md` 为正式最终运行时语义来源。若本文和 `docs/28` 在 parallax budget、render_size、OpenXR 输出语义上出现差异，以 `docs/28` 为准，并回写本文保持一致。历史 `docs/25-2d-to-3d-runtime-specification.md` 已作废，不再作为当前规范裁决来源。

| 领域 | 当前状态 | 剩余要求 |
|---|---|---|
| Parallax formula | 已有 `resolve_parallax_budget(render_width, render_height, preset)`，normalized-depth 主路径使用 `max_disparity_px` / `parallax_budget_preset` 解析档位预算，并使用 `depth_strength` 连续缩放实际视差位移；旧 `IPD/stereo_scale/max_shift_ratio` 配置、热更新和 adapter 兼容入口已清理 | 继续通过测试防止旧强度链回流；Depth Strength 保留为用户可调强度 gain，不作为旧物理 IPD 链的一部分 |
| RuntimeSettingsSnapshot | 已有 `RuntimeSettingsSnapshot`、`settings_update_q`、帧边界应用、热更新分级、结构化 result 字段 | GUI live hot-save 仍需进一步收敛为直接发送 snapshot；settings.yaml + StereoHotReloader 只保留为兼容路径 |
| Capture metadata | 已有 `CapturedFrame` / `FrameCopyMode`，event/polling runner 会携带 source、device、dtype、copy_mode、capture_size 等 metadata，并进入 runtime debug | 真实硬件 CUDA/ROCm zero-copy 仍需设备验证后才能把路径标成 true zero-copy |
| Capture preprocess device contract | 已显式处理 numpy / CPU tensor / CUDA tensor / ROCm tensor 形态，并记录 origin/output device 与 transfer metadata | 跨设备 fallback 和硬件路径仍需按目标机器补充验证矩阵 |
| OpenXR direct uniforms | 已输出规范 `shader_uniforms`，字段以 `max_disparity_px`、`depth_strength`、`depth_response`、`convergence`、`render_size`、`screen_roll` 为主；OpenGL 与 D3D11 RGB+D direct 调用层均按 `max_disparity_px / render_width` 派生每眼 shader offset，并使用同一 `depth_strength` 放大实际视差位移，不再消费 IPD / Stereo Scale / Max Shift Ratio 旧强度链 | D3D11 native direct shader 仍需追平 OpenGL direct shader 的完整 DIBR 质量语义；这是独立 follow-up |
| OpenXR headset screen presets / OSD | 已新增 `XR Headset Model` 设置和 `src/utils/xr_headset_presets.py`，按推荐距离 + 60° 水平视角自动计算屏幕尺寸；`_screen_view_distance()` 统一用户可见距离；preset OSD 显示 5 秒且不被 live distance 刷新；Y 恢复同一 preset 会重新显示 OSD；右手柄保留 sphere-orbit 并自动朝向头部 | 后续如增加水平视角 GUI slider，只应调整统一 FOV 参数或显式设置，不应回到每个预设手工维护宽高 |
| GPU texture upload | 已抽出共享 `CudaGlTextureUploader`，OpenXR runtime eye 与本地 viewer runtime RGBA texture 复用同一 CUDA/GL upload 语义；image texture 优先、PBO fallback、CPU fallback 红色告警 | 继续真机验证 CUDA/GL image texture 失败根因和各显示模式 fallback 日志；RGB+depth direct texture path 可按同一 uploader 继续收敛 |
| Depth provider GPU timing / zero-copy | TensorRT native 已记录真实 CUDA event timing；MIGraphX 构建已导入 ROCm7 FP8-first/FP16 fallback/force-FP32 skip 规则；TensorRT ORT CPU staging 已作为下一优化目标记录 | TensorRT ORT / ONNX Runtime realtime provider 仍需移除 CPU numpy input/output 往返，保持 iobinding output 在 GPU 并直接返回 CUDA tensor |
| Render Size / 4K scale tier | 已收敛为固定 scale 档位 Render Scale：非 4K 保持 capture_size；4K 级输入按 4K/3K/2K/1K 稳定 scale 档位解析，并保持横屏、竖屏、16:10、DCI 4K、常见 4K 超宽比例；判断排除面积不足的窄高/1440p 超宽；旧 numeric / short alias Render Scale 输入已清理为默认回退 | 继续通过测试防止重新引入 `0.75`、`75%`、`2K` 等用户侧别名；无新的运行时语义待办 |
| Runtime scheduling / backpressure | 已确认 capture 前段可到高刷 cadence，完整 CUDA runtime 后段若无 GPU 完成边界会因异步队列积压反压 WGC / CUDA interop；规范要求 latest-frame、raw overwrite/drop 和 `D2S_RUNTIME_SYNC_AFTER_FRAME=auto` 的 CUDA runtime 同步策略 | 继续用真实高刷显示器验证 CUDA/ROCm/非 CUDA 后端矩阵；不得把该问题误归因到 OpenXR presentation target |
| Debug / result contract | `StereoRuntimeResult` / `OpenXRRuntimeResult` 已暴露 output/timing/provider 结构化字段；每帧 debug_info 已补齐 application_runtime_target、stereo_synthesis_mode、transport、output_transport、capture/render/depth size 和 active settings metadata | debug-only 兼容键仍按兼容清理表逐步移除；host/viewer 新消费路径应继续优先读结构化字段 |
| Network stream | MJPEG/legacy stream 已消费 packed frame，并引入 `EncoderProfile` 描述 transport 侧 resize、pixel format、quality/FPS 等 | RTMP/更低延迟编码仍是后续工程；不能重新定义立体参数语义 |

## 后续实施优先级

1. 完成 GUI live hot-save 到 `RuntimeSettingsSnapshot` 的直接发送路径，把 settings.yaml polling 降为持久化同步路径。
2. 深挖 CUDA/GL image texture 失败根因，确保 OpenXR 和本地 viewer 都优先走 image texture，PBO/CPU fallback 都有明确日志。
3. 移除 TensorRT ORT / ONNX Runtime depth provider 的 CPU numpy 往返，实现真正 GPU zero-copy input/output。
4. 做 D3D11 native OpenXR direct shader parity，使其核心 DIBR 质量语义追平 OpenGL direct shader。
5. 做 CUDA/ROCm capture zero-copy 硬件验证，只有实测无 CPU 中转后才允许把 metadata 标为 `zero_copy=True`。
6. 做 runtime scheduling/backpressure 回归验证：CUDA 默认同步、latest-frame overwrite/drop、非 CUDA 后端无误触发。
7. 清理兼容冗余：旧 snapshot/API 字段、debug-only 兼容字段；legacy parallax 乘数字段和 render-scale 数值/短写别名已清理，后续只需防回归。
8. 继续完善 network_stream 的 encoder transport contract，尤其是 RTMP/低延迟编码路径，但保持其只消费 packed synthesis 输出。

## 结论

Desktop2Stereo 的工程设计应以 `render_size` 坐标系、规范化视差预算、统一 runtime settings、可追踪 GPU 数据路径为核心。当前项目已经具备 Flet GUI、跨平台 capture、多后端 depth provider、Triton 合成/打包、本地 viewer、MJPEG 推流和 OpenXR 双路径基础。下一阶段的重点不是继续增加散落参数，而是把现有功能收敛到统一的工程契约：

```text
CaptureFrame metadata
RuntimeSettingsSnapshot
RenderSize / 4K scale tier
ParallaxBudgetResolver
StereoRuntimeResult / OpenXRRuntimeResult debug contract
Transport-specific presentation contract
```
