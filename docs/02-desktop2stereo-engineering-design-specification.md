# Desktop2Stereo 工程设计规范

日期：2026-07-16

本文在 `docs/01-Realtime-2d-to-3d-specification.md` 的正式最终运行时流程规范基础上，对照当前工程代码，定义 Desktop2Stereo 的完整工程设计规范。本文关注模块职责、现有实现路径、硬件加速边界、数据契约、热更新、调试与后续演进。历史 `docs/25-2d-to-3d-runtime-specification.md` 已作废，仅作为背景记录。

## 文档定位与 docs/01 对齐方式

`docs/01-Realtime-2d-to-3d-specification.md` 是正式最终运行时流程规范，负责定义目标语义和十一阶段处理流程。本文不重复定义另一套运行时规范，而是按 docs/01 的流程顺序记录当前工程实现、迁移边界、兼容清理项和验证责任。

写作规则：

```text
1. docs/01 决定应该做什么，以及每个阶段的输入/输出语义。
2. docs/02 记录当前工程由哪些模块实现这些阶段，以及哪些兼容路径还没有清理。
3. 若 docs/02 和 docs/01 的运行时语义冲突，以 docs/01 为准，并回写 docs/02。
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
6. 实时 CUDA 主路径不得为了标量控制、日志诊断或采样颜色调用 `.item()` / `.cpu()` / `.numpy()` 造成每帧同步。
7. GUI 只负责配置、启动、热参数保存和用户交互，不直接修改 runtime 内部对象。
8. 所有输出路径必须可 debug、可定位、可回归测试。
9. OpenXR 主画面立体呈现以 projection layer 为可靠路径；Quad layer 在 VDXR 下只作为实验、诊断或 overlay 路径，背景、Glow、墙面反射仍是软实时路径，不得阻塞 screen submit。
10. OpenXR 启动时必须先完成 runtime preparation / TensorRT build，再启动 capture_loop，避免 WindowsCaptureCUDA 与 TensorRT/PyTorch CUDA 初始化或 engine build 并发触发 CUDA stream/capture 冲突。
11. OpenXR / local preview 的 glTF 资源渲染以 glTF 2.0 renderer compliance layer 为目标：parser、合规适配、backend upload/render 明确分层，controller 与 environment 共享同一 mesh/material/texture/render-pass contract。
12. Panda3D glTF renderer 是 `docs/39` 定义的成熟 runtime 迁移候选：它只能作为 opt-in 路径接入现有 OpenXR Projection swapchain，不能替换 depth/runtime/latest-frame 调度、OpenXR session、controller input 或屏幕拖动状态机；D3D11/NV_DX 是主路径，OpenGL shared-context 是 fallback 路径；默认 renderer 在真实 OpenXR swapchain GPU-only gate 通过前保持 native/OpenGL 测试路径。
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
| Local viewer | `src/viewer/viewer.py`, `viewer_runtime.py`, `gl_texture_uploader.py` | GLFW/ModernGL 本地显示、SBS/TAB/Anaglyph/Interleaved/Leia、OpenGL 路径共享 GPU tensor -> GL texture upload 后端 |
| OpenXR viewer | `src/xr_viewer/*.py`, `src/xr_viewer/gltf/*.py`, `src/xr_viewer/panda_runtime/*.py` | OpenXR session、swapchain、Windows 默认 D3D11 native / OpenGL forced-or-auto fallback render、controller/environment、runtime eye 上传；主画面 3D 以 projection layer 为可靠路径，Quad layer 保留为实验/诊断/overlay；glTF 资源经 `xr_viewer.gltf` parser/contract/render-plan/color-management 层输出共享 primitive/material/texture/render-pass contract，OpenGL/D3D11/preview 只负责 API 绑定、颜色空间和 swapchain/窗口呈现；`panda_runtime` 是 opt-in Panda3D scene renderer facade，负责 Panda scene graph、animation clock、screen/controller/ray snapshot、offscreen targets、D3D11/NV_DX 主 bridge、OpenGL shared-context fallback bridge contract 和 diagnostics，真实 OpenXR swapchain GPU-only gate 通过前不得成为默认路径 |
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
普通模式：
CaptureSessionLoop
-> raw_q(maxsize=1)
-> RuntimePipelineLoop
-> runtime_q(maxsize=1)
-> local viewer / network streamer / OpenXR viewer

OpenXR 启动顺序：
RuntimePipelineLoop.prepare()
-> stereo_runtime.load()
-> model cache / ONNX / TensorRT artifact preparation
-> start CaptureSessionLoop
-> latest-frame runtime loop
-> OpenXR viewer
```

队列规范：

```text
raw_q 只保留最新 capture frame，旧帧可丢弃。
runtime_q 只保留最新 runtime result，避免渲染端积压。
丢帧优先于增加交互延迟。
CUDA runtime 后段慢于 capture cadence 时，必须按 GPU 完成节奏推进 latest frame，不能无限异步排队。
```

## docs/01 流程到工程实现映射

| docs/01 阶段 | 当前工程实现 | 当前状态 / 迁移边界 |
|---|---|---|
| 步骤 1：Capture Input | `src/capture/session.py`, `src/capture/runners.py`, `src/capture/types.py`, `src/capture/backends/*` | 已有 `CapturedFrame` metadata、copy mode、source/capture size 记录；真实 CUDA/ROCm zero-copy 仍需硬件验证 |
| 步骤 2：Resolve Render Size | `src/stereo_runtime/render_size.py`, `src/stereo_runtime/pipeline.py`, GUI Render Scale 配置 | 已按 4K/3K/2K/1K 固定 scale 档位解析；4K级输入按比例缩放并保持输入宽高比；历史 numeric / short alias Render Scale 输入已清理并有测试防回归 |
| 步骤 3：Resize RGB To Render Size | `src/capture/preprocess.py`, `src/stereo_runtime/pipeline.py` | 已在 pipeline 边界把 capture frame 解析到 render_size；跨设备 fallback 仍需继续扩展验证矩阵 |
| 步骤 4：Depth Estimation | `src/stereo_runtime/depth_provider.py`, `src/stereo_runtime/depth_onnx_provider.py`, `src/stereo_runtime/providers/*`, `model_registry.py` | 已支持多 provider / backend；provider 内部尺寸可以不同，但输出必须回到 render_size；实时 provider 主路径不得 CPU numpy 往返，TensorRT/MIGraphX 需区分 GPU event timing 和 CPU enqueue timing |
| 步骤 5：Depth Postprocess | `src/stereo_runtime/runtime.py`, `src/stereo_runtime/synthesis.py`, `src/stereo_runtime/depth_postprocess.py` | 当前使用 normalized / relative depth 处理和轻量 postprocess；`Depth Pop` 是公开的居中深度曲线入口；不能改写为 metric Z 路径 |
| 步骤 6：Resolve Parallax Budget | `src/stereo_runtime/parallax.py`, `adapter.py`, `settings_snapshot.py` | 已有预算表、短边插值和超宽保护；`resolve_parallax_budget().depth_response()` 支持 float 或 tensor `convergence`，tensor 路径保持在 depth device/dtype；legacy multiplier 只应留在显式兼容入口；`Depth Separation Preset` 与 FG/MG/BG Pop 是预算之后的分层视差倍率控制，不替代预算 |
| 步骤 7：Disparity Field | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `openxr_render.py` | 主语义为 `depth_response * max_disparity_px * depth_strength`，再由 depth-pop / FG-MG-BG layer pop 调整不同深度区域的位移倍率；动态会聚可把 measured convergence 作为 CUDA 标量 tensor 继续传递，不能为当前帧调用 `.item()`；CPU-only shader uniform 消费者使用异步 pinned scalar staging 和上一帧 ready 值；near/far、response、depth_strength、convergence、dynamic_convergence_strength 和 layer pop 必须在 metadata / shader uniforms 中可追踪 |
| 步骤 8：Stereo Warp | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `layers.py`, OpenXR shader paths | full synthesis 已有 baseline/layered DIBR；OpenXR D3D11 direct shader 已补齐 OpenGL direct 的核心 DIBR 语义：3-tap depth smoothing、edge falloff、soft disocclusion confidence、directional inpaint；仍需 VDXR 实机视觉/性能验证 |
| 步骤 9：Mask and Hole Fill | `src/stereo_runtime/synthesis.py`, `layers.py` | 已有 edge-aware / directional fill 路径；AI inpainting 只作为未来离线/质量候选 |
| 步骤 10：Temporal Stabilization | `src/stereo_runtime/runtime.py`, `pipeline.py`, `settings_snapshot.py` | 已有 scene/render_size/source/settings reset 机制；temporal 只做稳定，不改变 parallax budget |
| 步骤 11：Output Pack / Viewer Upload | `src/stereo_runtime/output.py`, `output_triton.py`, `src/viewer/*`, `src/xr_viewer/*`, `src/streaming/*` | 本地/网络消费 packed output，OpenXR 支持 full synthesis 与 RGB+depth direct；算法内部保持 GPU tensor，进入显示边界时才上传到 D3D11/GL/Metal 等图形资源；Windows OpenXR 默认 D3D11 native，OpenGL 作为本地 viewer 和显式 forced/auto fallback；OpenGL 路径共享 `GlTensorPboUploader` / `CudaGlTextureUploader`，D3D11 native 不复用 OpenGL uploader；runtime-eye tensor diagnostics 只能记录 shape/dtype/device 等 no-sync 信息；glow 实时采样必须走 GPU source texture，不允许回到 CPU tensor sampling；VDXR 下主画面 3D 以 projection layer 为可靠路径，Quad-layer screen presenter 只保留为实验/诊断/overlay；D3D11 desktop/screen intermediate color texture 使用 UNORM，glTF base/emissive 纹理按 sRGB 语义、normal/MR/occlusion 按 linear 语义处理；RTMP/低延迟 stream 是后续工程 |

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
-> 更新 settings.yaml 中 Stereo Preset / Stereo Quality / Convergence / Dynamic Convergence Strength / Depth Strength / Parallax Budget / Depth Pop / Depth Separation / FG-MG-BG Pop / Temporal / Hole Fill / Edge 等规范字段
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
| Stereo | `Stereo Preset`, `Stereo Quality`, `Synthetic View`, `Depth Strength`, `Convergence`, `Dynamic Convergence Strength`, `Parallax Budget Preset`, `Max Disparity Px`, `Depth Separation Preset`, `Foreground Pop`, `Midground Pop`, `Background Pop` | 当前规范立体参数；Depth Strength 是用户连续强度 gain，Parallax Budget 是档位预算，Convergence 是静态零视差平面，Dynamic Convergence Strength 为 0.00 时关闭动态会聚、大于 0.00 时启用动态会聚 |
| Synthesis postprocess | `Depth Pop`, `Temporal Strength`, `Scene Reset Threshold`, `Edge Dilation`, `Mask Feather Radius`, `Hole Fill Mode`, `Edge Threshold`, `Anti-aliasing` | 深度曲线、稳定、mask、补洞、depth postprocess |
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
-> 权重文件必须存在且大于 0 字节；0KB cache / download 结果一律视为无效
-> 本地权重存在则本地导出 ONNX / 构建后端 artifact
-> 本地 artifact 和权重都不存在时才探测 Hugging Face endpoint 并下载
-> ensure_model_downloaded()
-> huggingface_hub.snapshot_download(...) 或 hf_hub_download(...)
-> 若 Hugging Face hub 返回空权重或下载异常，则用 requests 直连 resolve URL 流式下载到 .tmp，并通过 os.replace 原子替换为正式权重文件
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
convergence (float or tensor scalar)
dynamic_convergence_strength
parallax_budget_preset
max_disparity_px
depth_pop
foreground_pop / midground_pop / background_pop
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
当前：动态会聚、motion sampler、OpenXR shader uniform 等实时标量路径必须避免当前帧 `.item()` 硬同步。
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
optional CUDA/HIP/ROCm PBO upload
shader display modes: SBS/TAB/Depth/Anaglyph/Interleaved/Leia/Mono
```

OpenGL upload 当前规则：

```text
GlTensorPboUploader 负责通用 GPU tensor -> GL PBO -> OpenGL texture，覆盖 CUDA/HIP/ROCm 可共用路径。
CudaGlTextureUploader 负责 CUDA RGBA tensor -> OpenGL texture，优先 CUDA/GL image texture copy，失败后 PBO fallback。
CPU upload 只允许作为显式 fallback，必须红色控制台告警并记录失败原因。
离散 GPU 可使用 pinned host staging / PBO 降低 host-frame 上传开销；集成 GPU 避免不必要 pinned staging overhead。
D3D11 native / Metal / Vulkan / RTMP-MJPEG 不复用 OpenGL uploader。
```

颜色空间规则：

```text
OpenGL 和 D3D11 可以共用材质/纹理语义，但不能共用 API 级格式处理。
D3D11 desktop/runtime screen intermediate color texture 使用 DXGI_FORMAT_R8G8B8A8_UNORM，避免中间纹理被 sRGB 自动解码后又写入 UNORM RTV 导致画面偏暗。
OpenXR swapchain format 仍按 runtime 暴露格式选择；如果 runtime 提供 sRGB swapchain，由最终 projection render 路径负责匹配其输出语义。
glTF baseColor / emissive texture 按 sRGB 语义处理；normal / metallic-roughness / occlusion texture 按 linear data 处理。
GLB 解析、图片解码和首次 texture upload 允许使用 CPU；每帧 controller rendering 必须采样已经创建好的 GPU texture/SRV。
```

规范：

```text
本地 viewer 接收 packed frame，不重新定义立体公式。
本地 viewer 可以做 display mode shader/presentation，但不能改变 synthesis 的 max_disparity_px 语义。
算法内部应保持 CUDA/ROCm/MPS/XPU tensor；只有进入 viewer / OpenXR / stream 显示或编码边界时才转换为对应资源。
CUDA/GL image texture 与 PBO 都是 OpenGL upload 后端，不是立体算法；PBO 是 fallback 时必须标记为 fallback。
glGenerateMipmap 不是 CUDA 功能；实时 viewer/OpenXR 上传路径默认不得每帧生成 mipmap，除非实际使用 mip level 采样并已验证收益。
```

## OpenXR viewer 设计

当前模块：

```text
src/xr_viewer/base.py
src/xr_viewer/openxr_runtime.py
src/xr_viewer/d3d11_native_renderer.py
src/xr_viewer/controller_materials.py
src/xr_viewer/controller_models.py
src/xr_viewer/laser_params.py
src/xr_viewer/core_* modules
src/xr_viewer/environment*.py
src/xr_viewer/controllers/common.json
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
Windows OpenXR 主路径应优先使用 D3D11 native。
render_eye(): RGB+depth shader path。
render_runtime_eye(): runtime eye texture path，depth_srv=None。
CUDART_D3D11 支持 cudaGraphicsD3D11RegisterResource + cudaMemcpy2DToArrayAsync device-to-device copy。
D3D11 native 目标是 D3D11 texture / OpenXR D3D11 swapchain，不得复用 GlTensorPboUploader 或 CudaGlTextureUploader。
```

OpenGL texture upload path：

```text
OpenGL OpenXR 是兼容 fallback，本地 viewer 仍使用 OpenGL。
runtime_direct_opengl_image_texture 优先用于 CUDA tensor -> OpenGL image texture copy。
runtime_direct_opengl_pbo 是 GPU fallback，不代表 image texture 成功。
通用 RGB/depth GPU tensor -> OpenGL texture 使用 GlTensorPboUploader。
CPU GL upload 只允许作为显式 fallback，必须红色控制台告警。
debug/log 必须记录 upload backend、fallback reason、eye texture size。
```

Panda3D OpenXR renderer gate：

```text
Panda3D path 是 renderer ownership 切换：Panda3D 负责 glTF scene graph、animation、PBR/material、skybox、screen NodePath、controller/ray NodePath 和每眼 camera；OpenXR 层仍只负责 session、view locate、swapchain acquire/wait/release、Projection layer submit 和现有交互状态机。
D3D11 主路径必须证明 WGL_NV_DX_interop2 可对当前 acquired OpenXR D3D11 swapchain texture 注册、lock，并作为 GL-writable FBO/texture 被 Panda 渲染。
OpenGL fallback 必须证明 Panda3D GL context 与 OpenXR GL context 是同一个，或处于已验证的 WGL/GLX shared object namespace，使 Panda render target 和 OpenXR swapchain FBO/texture 可互相可见。
普通 D3D11 Texture2D register/lock/FBO complete 只证明 readiness，不等同于真实 OpenXR swapchain + Panda3D GSG/context + acquire/render/release 真机循环通过。
Panda offscreen texture + GPU blit 可以作为 GPU-only 过渡路径单独计时，但不能标记为 strict zero-copy。
CPU readback、PBO readback、PIL/Numpy staging 或任何 CPU 图像搬运在 realtime Panda projection path 中禁止使用。
GLError 1282、FBO incomplete、swapchain lock 失败、无法证明 shared GL context、或无法在 headset 中看到 Panda-rendered frame，都必须记为 Panda zero-copy gate failure。
```

规范：

```text
OpenXR full synthesis 不接受 SBS/TAB 作为 viewer 输入。
OpenXR full synthesis 的 runtime 应优先产出 OpenXR 可直接上传的 uint8/RGBA GPU tensor，viewer 只负责边界上传和 present。
OpenXR debug 可额外导出 SBS 预览，但不作为 swapchain 输入。
viewer 建屏优先使用结构化 `output_display_size`，legacy `debug_info["runtime_output_display_size"]` 只作为兼容 fallback；eye texture 优先使用结构化 `output_eye_size`。
Direct path 的 legacy uniforms 必须由 adapter 从规范参数转换。
OpenXR `xr_wait` / `xr_poll` / `xr_submit` / present 时间属于设备运行时调度，不得混同为 StereoRuntime depth/synthesis/SBS 生成耗时。
VDXR 下 Quad layer stereo overlay 不作为主画面 3D 的可靠路径；主虚拟屏幕应走 projection layer，Quad layer 仅作为实验、诊断或 overlay。
```

### OpenXR shared interaction layer 与 renderer backend 边界

OpenXR 的交互/状态逻辑必须与渲染后端解耦。D3D11 native 和 OpenGL fallback 可以共用操作语义，但不能共用底层绘制实现。

可共用层：

```text
controller pose / input action state
ray hit test / drag / click
virtual screen transform: position / distance / size / curvature-ready state
RuntimeSettingsSnapshot hot update
stereo parameter live adjustment
virtual keyboard layout / key state / text input
OSD state machine / status text
OpenXR frame loop scheduling state
```

后端分离层：

```text
controller mesh rendering
projection screen draw
keyboard / OSD texture rendering and submit
shader programs
texture upload
swapchain render target / blend state / depth state
```

工程规则：

```text
D3D11 native 实现时不得复制手柄操作、屏幕调节、热参数、虚拟键盘等业务逻辑。
应复用 OpenXR shared interaction layer，只新增或补齐 D3D11 renderer backend。
OpenGL backend 保留为本地 viewer / OpenXR fallback；两套 backend 只在绘制、shader、texture 和 swapchain 层分叉。
当前显示分工：主虚拟屏幕、controller model、laser beam、screen hit cursor 属于 main Projection path；virtual keyboard、short OSD/FPS/help panel 属于 Quad overlay。
不再使用 always-on foreground Projection layer 补画手柄/激光；该方案会增加 XR loop 压力且不是当前默认实现。
键盘 cursor 是标准 OpenXR Quad composition layer 的一个小 overlay，用静态 RGBA cursor texture；只有对应手柄仍 hover keyboard key 时提交，离开键盘后必须隐藏。
Keyboard hover/held key highlight 属于 keyboard texture 内容，只在 hover/held key index 变化时刷新；cursor 位置变化不得触发整张 keyboard texture 每帧重建。
```

### OpenXR glTF 2.0 core renderer compliance layer

`docs/38-gltf-2-renderer-compliance-layer-plan.md` 是当前 glTF 渲染层实施计划。该计划的目标是完整 glTF 2.0 core renderer compliance；当前代码已经完成静态 mesh/material/render-pass 子集，但不能把这一子集称为完整 glTF 2.0 支持。

工程边界如下：

```text
glTF 2.0 asset
-> parser / optional mature glTF runtime evaluation
-> glTF core compliance adapter
-> stable scene/node/primitive/material/texture/animation/skin/morph/camera contract
-> runtime state update
-> OpenGL backend / D3D11 backend / preview backend
```

分层规则：

```text
1. parser 层负责读取 glTF/GLB 文档、buffers、bufferViews、accessors、images、samplers、materials、meshes、nodes、scenes、animations、skins、cameras 和 extensions。
2. compliance adapter 负责把 glTF core 语义转换为项目内部稳定 contract，包括 vertex layout、index format、accessor normalized/sparse/stride、material alpha、texture color role、sampler wrap/filter、doubleSided、node hierarchy、animation channels/samplers、skin joints/inverseBindMatrices、morph target weights、camera projection、extension support 和 render pass。
3. runtime update 层按 frame time 更新 animated node matrices、skin joint matrices、morph weights 和 active scene/camera 状态；静态模型可以缓存结果，但动态模型不得把 node transform 永久烘焙进唯一顶点缓冲后丢失动画。
4. backend upload 层只创建 API 对象：OpenGL texture/buffer/VAO、D3D11 buffer/SRV/sampler/state、preview ModernGL resource。动态 node 应通过 per-primitive model matrix、joint matrix buffer 或 morph target buffer/texture 进入 draw，而不是改写 glTF 语义。
5. backend render 层只消费 render plan 和 runtime state，不重新解释 glTF JSON。
6. preview_room_layout.py、OpenXR OpenGL 和 OpenXR D3D11 必须使用同一 scene/primitive/material/runtime contract；不得各自猜 stride、alphaMode、texture role、animation semantics 或 extension 语义。
```

当前必须锁住的 core 合规点：

```text
POSITION / NORMAL / TANGENT / TEXCOORD_0 / TEXCOORD_1 / COLOR_0 / JOINTS_0 / WEIGHTS_0 contract 明确，vertex stride 由 accessor/bufferView contract 给出。
indices 支持 UNSIGNED_BYTE / UNSIGNED_SHORT / UNSIGNED_INT。
accessor byteOffset、bufferView byteOffset/stride、normalized、sparse accessor、GLB BIN、external .bin 和 data URI 均由 parser/adapter 层处理。
node hierarchy 支持 matrix 与 TRS；TRS animation 支持 translation / rotation / scale channel 和 LINEAR / STEP / CUBICSPLINE interpolation。
skins 支持 joints、skeleton、inverseBindMatrices 和 per-frame joint matrices；morph targets 支持 target POSITION/NORMAL/TANGENT 与 mesh/node weights。
baseColor / emissive texture 按 sRGB；normal / metallic-roughness / occlusion 按 linear data。
alphaMode=OPAQUE/MASK/BLEND 进入明确 render pass；BLEND 不得混入 opaque pass。
KHR_materials_unlit 以 extension presence 判定，空对象 `{}` 仍表示启用。
cameras 和 multi-scene 属于 core；没有显式相机时允许使用 profile/viewer fallback，但不能误解析 asset camera。
extensionsRequired 中不支持的必需扩展必须 fail fast，不允许静默显示错误模型。
```

当前实现状态：

```text
src/xr_viewer/gltf/document.py         -> glTF/GLB document、buffer 和 URI/data loading
src/xr_viewer/gltf/accessors.py        -> accessor、bufferView、stride、component/type 解码
src/xr_viewer/gltf/materials.py        -> GltfMaterial、texture binding、sampler/transform/material roles
src/xr_viewer/gltf/primitives.py       -> load_glb_model compatibility primitive stream；当前仍偏静态烘焙，需要为 animated nodes 保留 local mesh + node matrix contract
src/xr_viewer/gltf/scene.py            -> GltfScene、diagnostics、shared scene contract；当前 node matrices 是一次性静态计算
src/xr_viewer/gltf/render_plan.py      -> render pass classification and transparent sorting
src/xr_viewer/gltf/color_management.py -> sRGB/linear/tone-map/output gamma policy
src/xr_viewer/gltf/loader.py           -> public package loader facade
src/xr_viewer/gltf_loader.py           -> legacy compatibility facade only
```

新增/待补模块边界：

```text
src/xr_viewer/gltf/animation.py        -> glTF animation sampler/channel interpolation and per-frame node TRS output
src/xr_viewer/gltf/skinning.py         -> skin joint palette construction and backend joint buffer contract
src/xr_viewer/gltf/morph.py            -> morph target weight evaluation and backend morph input contract
src/xr_viewer/gltf/cameras.py          -> glTF perspective/orthographic camera contract and viewer fallback policy
```

实现规则：

```text
Environment primitives must carry `material_contract`; missing material_contract is a loader/adapter error.
Environment renderer-prepared fields use `render_material`; do not reintroduce the old environment primitive `material` dict fallback.
Controller backend code may still have renderer-prepared per-backend material objects, but glTF semantic fields must originate from GltfMaterial / shared material preparation.
Render planning is explicit: sky, opaque, mask, and transparent are separate pass buckets; transparent primitives are sorted back-to-front with the shared policy before blend drawing.
Color management policy is shared: shader work is linear, base/emissive textures are decoded from sRGB, output is tone-mapped and gamma encoded at the final shader boundary.
Animated GLB support must use standard glTF animation data. Artemis satellite/spaceship motion is the current acceptance fixture: if Don McCurdy's viewer plays the exported GLB animation, our preview/OpenXR backends must eventually play the same node transform animation through the shared runtime state.
Before expanding a homegrown renderer too far, evaluate mature glTF runtimes such as Filament `gltfio`; parser-only libraries are not enough because animation, skinning, morph targets, material variants, and validation behavior are renderer/runtime responsibilities.
```

该层不是 controller 专用。Controller model、environment GLB、preview tool 都必须逐步收敛到同一 contract。Controller profile 只保留品牌覆盖参数、手柄材质例外、按键动画、模型偏移/旋转等 profile 级差异；房间/环境光照和通用 glTF 材质语义不应散落在每个 controller profile 内。
### Environment profile transform and preview layout contract

Environment profile 的 model transform 使用 glTF/local mesh 坐标到 OpenXR/world 米制坐标的 TRS：

```text
world_position = model_position + model_rotation * model_scale * local_position
```

规则：

```text
1. `model_scale` 表示资源单位到世界米的换算；Artemis 当前按源单位厘米处理，因此 scale 为 `[0.1, 0.1, 0.1]`，整体约 `925.89m x 180.0m x 655.48m`。
2. `model_position` 是模型本地原点的世界平移项，不是 camera/view pose，也不是模型几何中心。
3. 对本地原点不在几何中心的模型，`model_position` 可以很大；它通常用于把 transformed bounds 水平居中并让底部落到 world `Y=0` 附近。
4. `view_pose` / `view_poses` 才定义预览和运行时相机/座位位置。地面观察位和几何中心观察位应作为不同 view pose 表达，不应通过误改 `model_position` 解决。
5. `screen` profile 默认锁定 controller grip move/resize，因为多数房间需要虚拟屏幕贴合 GLB 中的电视、投影墙或影院幕布；大型自由导航场景可设置 `screen.allow_controller_move=true`，保留初始屏幕 placement，但允许 OpenXR 手柄继续移动/缩放屏幕。
6. `preview_room_layout.py` 的移动速度和屏幕尺寸调节速度必须根据 transformed world bounds 自适应。普通房间保持基础速度，百米/千米级场景按最大边长放大移动速度；旋转速度保持角速度语义，不随场景大小缩放。
```

### OpenXR controller glTF material contract

Controller glTF/GLB 加载分为共享语义准备和后端 API 绑定两层。共享层负责解释模型和材质字段，OpenGL / D3D11 只负责创建各自 API 的 texture、sampler、constant buffer、shader resource view 和 render state。

共享字段：

```text
baseColorFactor / baseColorTexture
alphaMode / alphaCutoff / material alpha
doubleSided
sampler wrap/filter
texture transform / texcoord index
normalTexture
metallicRoughnessTexture / metallicFactor / roughnessFactor
occlusionTexture
emissiveFactor / emissiveTexture
sRGB texture role: baseColor, emissive
linear texture role: normal, metallic-roughness, occlusion
```

后端规则：

```text
D3D11 controller rasterizer front face 必须匹配 glTF/OpenGL 的 CCW front face，避免 HP/Index 等模型被错误剔除或看似透明。
Controller sampler 默认按 glTF sampler 语义处理；模型 UV 超出 0..1 时需要 WRAP/REPEAT，不能无条件 CLAMP。
透明/alpha/doubleSided 必须来自 glTF material 字段；不得为了掩盖背面问题全局丢弃 doubleSided。
`src/xr_viewer/controllers/common.json` 是 controller 共享配置和临时诊断入口；`diagnostics.materialMode=opaque_unlit` 只能用于定位纹理/alpha/光照问题，不是正常渲染模式。
当前已共享并绑定核心材质字段，但不能把现状描述为完整 glTF 2.0 合规；后续补齐项必须继续落到共享字段层。
```

Laser 规则：

```text
OpenGL 和 D3D11 laser 共享 `src/xr_viewer/laser_params.py` 的尺寸参数。
手柄端总宽度为 6 mm，远端总宽度为 2 mm。
彩色光束使用 renderer-specific crossed tapered quads 近似光柱；动画方向应从手柄向外流动。
```

### 2026-07-07 VDXR Quad stereo validation conclusion

实机日志已经把 OpenXR 主画面问题拆开：

```text
runtime eye diff mean=4.580/255 max=255/255
Quad layer OpenGL swapchains: ... mode=shared-array active=True
Quad submit diag: ... eye0 ... array=0; eye1 ... array=1
```

结论：

```text
1. StereoRuntime 已产出有差异的 left/right 眼图。
2. D2S 已把 Quad headers 按左右眼提交给 OpenXR。
3. VDXR 仍未呈现有效 3D，说明 Quad overlay stereo 在该 runtime 下不可作为主显示保证。
4. Projection layer 是已知可靠路径，因为 OpenXR projection view 天然按每眼相机合成。
```

工程决策：

```text
OpenXR 主画面 3D 回到 projection-layer stereo rendering。
Quad layer 保留 shared-array/per-eye 诊断能力、可选实验开关和 overlay 用途。
后续性能优化应围绕 projection 主画面减少重复渲染/上传，而不是继续把 VDXR 主画面押在 Quad stereo overlay 上。
```

### OpenXR 异步解耦渲染目标

`docs/35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md` 定义目标架构，`docs/36-OpenXR_Asynchronous_Decoupled_Rendering_Implementation_Plan.md` 定义实施计划。2026-07-07 的 VDXR 实机验证修正了主画面呈现边界：硬实时目标仍然成立，但主画面立体不能依赖 Quad stereo overlay，必须以 projection-layer stereo rendering 作为可靠路径。

目标分层：

```text
Capture + StereoRuntime -> latest OpenXRRuntimeResult -> ScreenFrameBridge -> projection-layer screen renderer -> xrEndFrame
ScreenFrameBridge -> AsyncEffectsScheduler -> GPU glow/light/reflection result pool -> background/projection shader consumes latest safe result
Environment profile/assets -> BackgroundBakeService -> panorama/cubemap background layer or sky sphere fallback
```

关键规则：

```text
1. 虚拟显示器是硬实时路径：OpenXR frame loop 不等待 runtime、背景、Glow 或墙面反射。
2. 背景、Glow、墙面反射是软实时路径：只消费已经完成的旧 GPU result。
3. Runtime producer 和 OpenXR presenter 之间只通过 latest result / reuse last frame 语义连接，不引入阻塞 backpressure。
4. Projection layer 是 VDXR 下显示器主体的可靠 3D 路径；Quad layer 不再作为默认主路径，只承载可选实验、诊断或 overlay。
5. 背景目标是 panorama/cubemap 或 OpenXR background layer；复杂 GLB 房间不应成为每帧显示器刷新瓶颈。
6. Glow / 屏幕光必须复用 `docs/20-openxr-gpu-glow-guide.md` 的 GPU downsample/shader 采样技术，禁止实时 CPU 采样。
```

首轮实施顺序：

```text
1. 固定启用 OpenXR async present / projection screen 主路径；仅保留 D2S_OPENXR_ASYNC_EFFECTS / D2S_OPENXR_PANORAMA_BACKGROUND 软路径开关和分段诊断。
2. 增加 ScreenFrameBridge，非阻塞 drain latest runtime result，缺帧时复用 last good screen texture。
3. 保持 controller raycast 使用同一个 logical screen plane；Quad-layer screen presenter 只保留为可选诊断/实验路径。
4. 验证复杂环境下 screen present FPS 与环境渲染成本解耦。
5. 再推进 panorama background、async GPU Glow result pool、GPU-only wall reflection / light probe。
```

GPU Glow / 屏幕光约束：

```text
允许：GL texture、低分辨率 glow texture、textureLod、边缘采样、区域采样、shader / compute pass、未来 D3D/Vulkan GPU pass。
禁止：实时 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 作为屏幕光颜色来源。
Glow Off 必须保持 fast path，不触发 downsample，不触发 CPU 采样。
```

当前落地状态：

```text
代码结构曾以 Quad Layer 作为显示器主路径推进；VDXR 实机验证证明该路径不能保证主画面 3D。screen/background/effect/submit 分段、panorama HDR/SBS、safe light probe、GPU downsample/result pool 与 D3D11/PBO 边界测试仍然有效，但主画面 hard-realtime 路径需要回到 projection-layer stereo rendering。
最终验收未完成：下一轮应证明 projection 主画面在慢 runtime、复杂背景、慢/失败 effect worker 下仍能稳定 present，并且真实头显可见 3D。
```

运行验收信号：

```text
FPSBreakdown.validate_openxr_async() 仍用于汇总 OpenXR async 健康信号，但不得把 Quad layer 可提交等同于 VDXR 3D 可见。
日志字段 openxr_async_ok / openxr_async_missing / openxr_async_failed 是当前快速判断入口。
Windows D3D11 projection-layer 主屏成功渲染两眼后必须记录 openxr_projection_screen_present，并在日志中显示为 screen_proj。
当 async effects 明确关闭时，日志必须显示 fx_enabled=0，OpenXR async 验收不得再要求 effect_submit_or_safe_reuse；当 async effects 开启时，仍必须验证 effect submit 或 safe reuse。
实机稳定运行时应看到：openxr_async_ok=1 openxr_async_missing=none openxr_async_failed=none。
如果 projection 主画面缺失、无 3D 或发生 projection render failure，不能判定 OpenXR 主路径完成。
```

D3D11 native 边界：

```text
D3D11 native 已有 runtime eye / swapchain upload 基础，但不应停留在 Quad Layer 主画面路径。
Windows OpenXR 目标是 D3D11 native projection-layer 主显示路径；完成 D3D11 projection parity 前，OpenGL / NV_DX interop projection 路径仍作为兼容验证路径。
```
### OpenXR 虚拟屏幕、头显预设与 OSD 规范

头显屏幕预设属于 OpenXR presentation 层，不改变 runtime 立体合成语义。GUI 只保存 `XR Headset Model`，运行时通过 `src/utils/xr_headset_presets.py` 解析设备推荐观看距离，并按统一水平视角计算屏幕尺寸。头显焦距/屏幕尺寸不得替代 `Convergence`、`Dynamic Convergence Strength`、`Parallax Budget`、`Depth Separation` 或 FG/MG/BG Pop；这些仍按立体参数表独立调节。

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
9. 屏幕与键盘的激光命中光圈共享同一视觉语义：屏幕光圈在 main Projection path 绘制；键盘光圈作为独立小 Quad overlay 显示在 keyboard Quad 之上，并按 eye-to-hit distance 缩放。
10. 键盘光圈只在对应手柄仍 hover keyboard key 时提交；离开键盘后必须隐藏，不能停留在最后位置。
11. 键盘按键 hover/held 高亮可刷新 keyboard texture，但 cursor 位置移动不得导致整张 keyboard texture 每帧重建或上传。
```

手柄屏幕操作规则：

```text
Left grip: 平移屏幕，保持距离和朝向。
Right grip: 保留 sphere-orbit drag，围绕头部移动屏幕中心；移动后自动更新 yaw/pitch 让屏幕朝向头部。
D2S_OPENXR_RIGHT_GRIP_SCREEN_ROTATION: 默认关闭，只控制右手柄 wrist roll 是否映射到 screen_roll；不得用该开关阻止 right-grip orbit 后的自动 yaw/pitch 朝向头部。
Both grips: 继续作为双手系统移动/调整路径，不由单手规则替代。
```

### OpenXR 手柄光照规范

手柄光照属于 OpenXR viewer 的 presentation 层，不改变 runtime 立体合成语义。屏幕灯光是所有手柄模型共用的统一光源；Pico、Quest、Valve、YVR 等模型只提供 mesh、texture、profile offset 和按钮动画，不各自定义独立灯光模型。环境 profile 的房间 punctual/fill lights 只作用于环境/房间渲染，不作用于 controller surface material。

Controller surface 只接受以下光照来源：

```text
screen light diffuse / specular reflection
head/top light
HDR panorama / environment reflection when enabled
base ambient floor
```

环境 profile 可以通过 `controller_hdr_lighting` 控制是否启用手柄 HDR 环境反射，`controller_hdr_reflection` 仅作为兼容别名；profile 中的 `_scene_lights` / `_env_fill_lights` 不进入 controller shader 常量：

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
| 顶灯亮度 | `CONTROLLER_TOP_LIGHT_INTENSITY = 0.55` | OpenGL/D3D11 共用参数，辅助补光，不能盖过屏幕主光 |
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
5. 新环境 profile 可以显式写 `controller_hdr_lighting`；`.hdr` panorama 可自动开启，但 profile 显式值优先。
6. Controller shader 不消费环境 profile 的 punctual/fill light 列表；如需改变 controller 明暗，必须通过屏幕光、头顶光、HDR 环境或材质参数路径实现。
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

参数分级遵循 `docs/01-Realtime-2d-to-3d-specification.md` 中的 Hot Reload / Pipeline Rebuild / Session Restart 表。

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
4. Viewer upload 在显示边界才发生：Windows OpenXR 优先 D3D11 native；OpenGL 路径共享 `GlTensorPboUploader` / `CudaGlTextureUploader`；CPU upload 必须红色 fallback 告警。
5. GLB controller asset load 允许 CPU parse/decode 和首次 texture upload；每帧 controller draw 必须采样 GPU texture/SRV。
6. 只有网络编码、CPU-only backend、离线保存/报告或显式诊断导出需要时才转 CPU numpy；实时主路径任何 CPU 回传都按 bug 处理。
```

### 实时 GPU 标量 / 诊断 no-sync 规则

```text
1. 实时 CUDA tensor 路径不得用 tensor.item() / tensor.cpu() / tensor.numpy() 做每帧控制流、日志统计或 glow 采样。
2. Dynamic Convergence 的 depth quantile 标量可保留为 CUDA 0-d tensor，并传递给可消费 tensor 的 parallax / synthesis / OpenXR prewarp 路径。
3. 只能接收 CPU/Python 标量的消费者，例如 OpenXR shader uniforms 或 motion sampler 外部标量，必须使用 non_blocking copy 到 pinned scalar buffer + CUDA event，并读取上一帧已经 ready 的值。
4. `core_runtime_eye.py` / `core_frame_upload.py` 的 tensor 诊断只记录 shape / dtype / device / no-sync 元数据；min/max/mean/diff 只允许在 CPU/numpy 路径或显式离线诊断中计算。
5. realtime glow 采样必须使用 GPU source texture；已删除的 CPU tensor sampling 分支不得重新引入。
6. OpenXR async Glow / screen-light 采样必须遵守 `docs/20-openxr-gpu-glow-guide.md`：使用低分辨率 GPU glow texture、shader/compute pass、边缘采样或区域采样；不得用 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 做实时颜色统计。
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
OpenXR 下 runtime preparation / TensorRT build 必须在 capture_loop 启动前完成，不能让 WindowsCaptureCUDA 与 TensorRT/PyTorch CUDA 初始化或 engine build 并发运行。
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
dynamic_convergence_strength
depth_pop
depth_separation_preset
foreground_pop
midground_pop
background_pop
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
FPSBreakdown.validate_openxr_async()
openxr_async_ok / openxr_async_missing / openxr_async_failed 日志字段
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

兼容清理以 docs/01 的最终运行时合同为目标；新代码默认不再增加别名、debug-only 字段或旧参数链路。历史兼容只允许留在明确命名的 import/parsing adapter 或 legacy fallback 中。

优先清理项：

| 清理项 | 目标状态 | 备注 |
|---|---|---|
| RuntimeSettingsSnapshot 字段别名 | 边界字段只保留 `parallax_budget_preset`、`temporal_enabled` 等 docs/01 名称 | `parallax_preset`、`temporal` 等旧名只允许在 legacy parser 中转换 |
| debug-only 兼容键 | host/viewer 消费结构化 result 字段 | `runtime_output_*`、flat OpenXR `openxr_*` shader-uniform debug keys 只作为短期 fallback |
| legacy parallax multiplier | normalized-depth 主路径只使用 `max_disparity_px` / `parallax_budget_preset`，`depth_strength` 只作为规范 gain 保留 | `ipd_mm`、`stereo_scale`、`max_shift_ratio` 不再作为核心强度链；旧 `Foreground Scale` 不再作为公开 GUI/config 字段，统一收敛为 `Depth Pop` |
| Dynamic Convergence checkbox | GUI 不再提供独立启用勾选框，布尔启停由 `Dynamic Convergence Strength` 推导 | `0.00` 表示关闭并使用静态 `Convergence`；大于 `0.00` 表示启用动态会聚强度 |
| render-size/render-scale 旧路径 | 用户侧只保留固定 scale 档位语义，并保持输入宽高比 | 删除历史 numeric scale thresholds、`native/fixed/dynamic` 用户策略语言、固定输出分辨率映射和非 4K 连续缩放行为 |
| D3D11 native direct shader | 已与 OpenGL RGB+depth direct 的核心 DIBR 语义对齐 | 仍需 VDXR 实机验证；完整体验 polish 如 rounded-corner alpha / feather controls 可作为后续视觉项，不得删除 OpenGL fallback 假设 |

估算边界：代码兼容清理加自动测试约 4-8 小时 AI 时间；若同时清理更广的 parallax/debug/render-size 遗留面，约 1 个 AI 工作日。OpenXR/D3D11 真机头显验证不包含在该估算内。

## 当前实现与规范符合状态

本文以 `docs/01-Realtime-2d-to-3d-specification.md` 为正式最终运行时语义来源。若本文和 `docs/01` 在 parallax budget、render_size、OpenXR 输出语义上出现差异，以 `docs/01` 为准，并回写本文保持一致。历史 `docs/25-2d-to-3d-runtime-specification.md` 已作废，不再作为当前规范裁决来源。

| 领域 | 当前状态 | 剩余要求 |
|---|---|---|
| Parallax formula | 已有 `resolve_parallax_budget(render_width, render_height, preset)`，normalized-depth 主路径使用 `max_disparity_px` / `parallax_budget_preset` 解析档位预算，并使用 `depth_strength` 连续缩放实际视差位移；`Depth Pop` 调整居中深度曲线，`Depth Separation Preset` 与 Foreground/Midground/Background Pop 在预算之后做分层位移倍率；`Dynamic Convergence Strength=0.00` 使用静态 `Convergence`，大于 0.00 时启用动态会聚 | 继续通过测试防止旧强度链回流；Depth Strength 保留为用户可调强度 gain，不作为旧物理 IPD 链的一部分；分层 Pop 不得替代 parallax budget |
| RuntimeSettingsSnapshot | 已有 `RuntimeSettingsSnapshot`、`settings_update_q`、帧边界应用、热更新分级、结构化 result 字段 | GUI live hot-save 仍需进一步收敛为直接发送 snapshot；settings.yaml + StereoHotReloader 只保留为兼容路径 |
| Capture metadata | 已有 `CapturedFrame` / `FrameCopyMode`，event/polling runner 会携带 source、device、dtype、copy_mode、capture_size 等 metadata，并进入 runtime debug | 真实硬件 CUDA/ROCm zero-copy 仍需设备验证后才能把路径标成 true zero-copy |
| Capture preprocess device contract | 已显式处理 numpy / CPU tensor / CUDA tensor / ROCm tensor 形态，并记录 origin/output device 与 transfer metadata | 跨设备 fallback 和硬件路径仍需按目标机器补充验证矩阵 |
| OpenXR direct uniforms | 已输出规范 `shader_uniforms`，字段以 `max_disparity_px`、`depth_strength`、`depth_response`、`convergence`、`dynamic_convergence_strength`、layer pop、`render_size`、`screen_roll` 为主；OpenGL 与 D3D11 RGB+D direct 调用层均按 `max_disparity_px / render_width` 派生每眼 shader offset，并使用同一 `depth_strength` 放大实际视差位移，不再消费 IPD / Stereo Scale / Max Shift Ratio 旧强度链；D3D11 native direct shader 已补核心 DIBR 质量语义 | VDXR 实机验证后再决定是否补完整视觉 polish；OpenXR 头显屏幕几何不得反向修改 convergence/parallax 参数 |
| OpenXR headset screen presets / OSD | 已新增 `XR Headset Model` 设置和 `src/utils/xr_headset_presets.py`，按推荐距离 + 60° 水平视角自动计算屏幕尺寸；`_screen_view_distance()` 统一用户可见距离；preset OSD 显示 5 秒且不被 live distance 刷新；Y 恢复同一 preset 会重新显示 OSD；右手柄保留 sphere-orbit 并自动朝向头部；头显屏幕预设只影响 presentation geometry | 后续如增加水平视角 GUI slider，只应调整统一 FOV 参数或显式设置，不应回到每个预设手工维护宽高；屏幕几何仍不得替代 `Convergence`、`Dynamic Convergence Strength`、`Parallax Budget`、`Depth Separation` 或 FG/MG/BG Pop |
| OpenXR main screen presentation | VDXR 实机验证证明 runtime left/right 有差异，D2S 也提交了 Quad `eye0 array=0` / `eye1 array=1`，但 Quad overlay 仍无有效 3D；projection layer 是当前可靠主画面 3D 路径。当前代码已把 D3D11 native runtime-eye / RGB+depth 主虚拟屏幕接回 projection-layer stereo rendering，并把成功的两眼 projection render 计入 `openxr_projection_screen_present` / `screen_proj`；D3D11 projection 主屏的 model matrix 已统一走 shared screen pose helper，OpenXR 公共 frame input 仍在渲染前复用现有 controller/raycast/screen-adjust/hot-parameter 状态；实机已验证 laser/controller 命中与可见屏幕一致，左右手柄拖动、距离/大小/旋转调节、Depth Strength / 2D-3D 热切换均能作用到 D3D11 projection 主屏；virtual keyboard / OSD 使用共享 overlay Quad presenter 和独立 overlay Quad swapchains，在 Projection 主屏之后合成，低频上传小 RGBA UI 纹理；keyboard/OSD 这类功能内容必须走 `overlay_textures.py` 共享 builder，OpenGL 以既有验证逻辑为准写入 OpenXR GL Quad texture，D3D11 写入 D3D11 Quad texture，backend 只负责上传/Quad layer 提交；Quad layer 仍仅保留为可选实验、诊断或 overlay，不作为主画面完成判据 | 下一轮按 `docs/36` phase 2 验证 OpenXR presenter 硬实时：runtime/capture/effects 慢时复用 last-good Projection screen frame，同时保持 controller/head pose refresh 和 `xrEndFrame` 不被阻塞 |
| OpenXR glTF renderer compliance | 已按 `docs/38` 拆出 `src/xr_viewer/gltf/` package，并把 legacy `gltf_loader.py` 收敛为 facade；已有共享 static vertex layout、`GltfMaterial`、texture binding/transform、color policy、render-pass classification、transparent sorting 和 `GltfScene(render_plan, diagnostics)`；preview/OpenGL/D3D11 当前消费同一 10-float primitive contract，Artemis 的 stride、alpha pass、empty unlit extension、UV1、legacy material dict fallback 问题已进入回归测试；`docs/39` 的 Panda3D path 已新增 asset/material/offscreen/NV_DX probes、`panda_runtime` facade、frame-state adapter、node animation sampling、Panda projection bridge diagnostics 和 OpenGL shared-context fallback bridge gate，但仍是 opt-in migration candidate | 当前 self-authored renderer 仍只是静态 renderer 子集，不是完整 glTF 2.0 core。下一步优先按 `docs/39` 证明 Panda3D 可真实接管 scene graph、TRS animation、skybox/material、screen/controller/ray 渲染并通过 D3D11/NV_DX OpenXR swapchain 可见；OpenGL fallback 只有在 Panda3D 与 OpenXR GL context 相同或已验证共享时才算通过；普通 D3D11 texture interop readiness 不能替代真实 OpenXR swapchain gate；若 Panda3D gate 不通过，再回到 `docs/38` 自研路线补 TRS animation sampling、per-frame node matrices、animated primitive model matrices、skinning、morph targets、sparse/interleaved/normalized accessor、camera/multi-scene 与 diagnostics |
| OpenXR controller materials / laser | Controller glTF material 字段已抽到共享准备层，OpenGL/D3D11 分别只做 API 绑定；D3D11 controller 已覆盖 base/normal/MR/occlusion/emissive、alpha/doubleSided、texcoord/transform、CCW front face 和 WRAP sampler，HP/Index 透明纹理问题已由实机视觉确认修复；controller light policy 已收敛为 screen light/reflection、head/top light、HDR environment 和 ambient，不消费环境 profile 的 punctual/fill lights；laser 宽度由 `laser_params.py` 共享，手柄端 6 mm、远端 2 mm，OpenGL/D3D11 均用 crossed tapered quads 近似彩色光柱 | 继续实机比较多个 controller model、HDR 房间和非 HDR 房间；`diagnostics.materialMode` 只作为临时定位手段，不能成为默认材质模式；controller 例外只能落在 profile 覆盖层，通用 glTF 语义必须回到 compliance layer |
| GPU texture upload / color space | 已抽出 OpenGL 共享 uploader：`GlTensorPboUploader` 统一 GPU tensor -> GL PBO -> GL texture，覆盖 CUDA/HIP/ROCm 可共用路径；`CudaGlTextureUploader` 负责 CUDA RGBA image texture copy 与 PBO fallback；CPU fallback 红色告警；算法内部应保持 GPU tensor，只有进入显示/编码边界才上传到 D3D11/GL/Metal 等资源；glow 实时采样只允许走 GPU source texture；Windows OpenXR 默认 D3D11 native，OpenGL 仅作为本地 viewer 和显式 forced/auto fallback；D3D11 desktop/screen intermediate color texture 使用 UNORM，避免 sRGB 自动 decode 后写 UNORM RTV 导致画面偏暗；glTF base/emissive 是 sRGB 语义，normal/MR/occlusion 是 linear data；GLB parse/decode/initial upload 可用 CPU，每帧 draw 必须采样 GPU texture/SRV | 继续 VDXR 真机验证 D3D11 projection/upload 和颜色亮度；继续验证 CUDA/GL fallback 日志 |
| Realtime no-sync scalar policy | 动态会聚、motion sampler、OpenXR shader uniform staging、runtime-eye tensor diagnostics 已按 no-sync 原则处理；实时相关文件检查不再包含 `.item()` | 后续新增 realtime CUDA 标量路径必须优先传 tensor；CPU-only 消费者只能异步 staging 并使用上一帧 ready 值，不能阻塞当前帧 |
| Depth provider GPU timing / zero-copy | TensorRT native 已记录真实 CUDA event timing；MIGraphX 构建已导入 ROCm7 FP8-first/FP16 fallback/force-FP32 skip 规则；TensorRT ORT CPU staging 已作为下一优化目标记录 | TensorRT ORT / ONNX Runtime realtime provider 仍需移除 CPU numpy input/output 往返，保持 iobinding output 在 GPU 并直接返回 CUDA tensor |
| Render Size / 4K scale tier | 已收敛为固定 scale 档位 Render Scale：非 4K 保持 capture_size；4K 级输入按 4K/3K/2K/1K 稳定 scale 档位解析，并保持横屏、竖屏、16:10、DCI 4K、常见 4K 超宽比例；判断排除面积不足的窄高/1440p 超宽；旧 numeric / short alias Render Scale 输入已清理为默认回退 | 继续通过测试防止重新引入 `0.75`、`75%`、`2K` 等用户侧别名；无新的运行时语义待办 |
| Runtime scheduling / backpressure | 已确认 capture 前段可到高刷 cadence，完整 CUDA runtime 后段若无 GPU 完成边界会因异步队列积压反压 WGC / CUDA interop；规范要求 latest-frame、raw overwrite/drop 和 `D2S_RUNTIME_SYNC_AFTER_FRAME=auto` 的 CUDA runtime 同步策略；OpenXR 现在先执行 `RuntimePipelineLoop.prepare()` / `stereo_runtime.load()` 完成 model cache、ONNX、TensorRT artifact preparation，成功后才启动 `CaptureLoop`，避免 capture CUDA 与 TensorRT build 并发 | 继续用真实高刷显示器验证 CUDA/ROCm/非 CUDA 后端矩阵；OpenXR engine 缺失/重建场景需确认日志顺序为 TensorRT build 完成后才出现 capture startup / CaptureGap；不得把该问题误归因到 OpenXR presentation target |
| Debug / result contract | `StereoRuntimeResult` / `OpenXRRuntimeResult` 已暴露 output/timing/provider 结构化字段；每帧 debug_info 已补齐 application_runtime_target、stereo_synthesis_mode、transport、output_transport、capture/render/depth size、active settings metadata、dynamic convergence、Depth Pop 和 FG/MG/BG Pop 诊断字段；OpenXR async 增加 `validate_openxr_async()`、`screen_proj` 和 `openxr_async_ok/missing/failed` 日志验收摘要；OpenXR loop segment 性能日志限制为最多 5 次，Panda bridge 成功诊断日志也限制为最多 5 次，Panda render-path status 和 projection render failure 会对重复错误做节流，并输出 `_panda_render_success_count`、`_panda_render_failure_count`、`_panda_render_last_bridge_mode`、`_panda_render_last_target_size`、`_panda_render_last_image_indices`、`_panda_render_error`、`_panda_render_last_timing` | debug-only 兼容键仍按兼容清理表逐步移除；host/viewer 新消费路径应继续优先读结构化字段；OpenXR 完成判定必须以实机日志显示 `screen_proj>0`、`openxr_async_ok=1` 且 missing/failed 均为 none 为依据；Panda3D migration 完成判定还必须看到 Panda bridge success > 0、failure == 0、bridge mode 为 `nv_dx` 或 `opengl`、错误为空、左右眼尺寸/image index 正常、无 CPU/PBO/PIL/Numpy image staging 且头显中真实可见 |
| Network stream | MJPEG/legacy stream 已消费 packed frame，并引入 `EncoderProfile` 描述 transport 侧 resize、pixel format、quality/FPS 等 | RTMP/更低延迟编码仍是后续工程；不能重新定义立体参数语义 |

## 未来实现目标

以下目标来自 `docs/31-directml-fallback-and-d3d11-d3d12-bridge-survey.md`、`docs/32-macos-zero-copy-capture-inference-survey.md` 和 `docs/33-quality-buffered-output-feasibility-report.md`。它们是后续工程目标，不改变当前默认实时路径。

### Windows DirectML 兜底与 D3D11-D3D12 桥接

短期目标是新增 `DirectMLDepthProvider`，让没有 CUDA / ROCm / XPU 的 Windows GPU 可以通过 ONNX Runtime `DmlExecutionProvider` 运行固定尺寸 ONNX 深度模型。该阶段允许 CPU/Numpy 输入进入 DirectML GPU 推理，但必须明确标注不是捕获零拷贝。

长期目标是原生 D3D11/D3D12/DirectML 零 CPU 回读链路：

```text
DXGI / Windows Graphics Capture D3D11 texture
-> NT shared handle / D3D12 OpenSharedHandle
-> D3D12 compute preprocess: resize / normalize / layout transform
-> ID3D12Resource tensor buffer
-> ONNX Runtime DirectML Device Tensor / IoBinding
```

约束：`onnxruntime-gpu`、`onnxruntime-directml` 和 `onnxruntime` 不应在同一发行包里混装并假设 provider 同时可用；应按 NVIDIA / DirectML / CPU 分发行策略管理。只有完整链路包含 D3D11 texture、D3D12 tensor bridge 和 DirectML inference 时，才允许命名为 Windows DirectML pipeline。

OpenXR / DirectML 边界约束：

```text
D3D11 native OpenXR renderer 不直接绑定 DirectML provider。
DirectML / D3D11-D3D12 bridge 后续作为 depth provider / preprocess bridge 接入。
两者通过 RuntimePipeline 的 GPU tensor / runtime result 合同衔接。
```

### macOS ScreenCaptureKit 到 Metal/CoreML 零 CPU 回读

当前 macOS 路线是 ScreenCaptureKit/CoreGraphics 到 CPU buffer，再上传到 PyTorch MPS；这是 MPS GPU 推理，不是捕获到推理零拷贝。未来目标是原生 bridge：

```text
ScreenCaptureKit
-> CVPixelBuffer / IOSurface
-> CVMetalTextureCacheCreateTextureFromImage
-> MTLTexture
-> Metal preprocess
-> CoreML or MPSGraph inference
-> Python bridge depth output
```

首版验证应选择 Distill Depth / Distill-Any-Depth Base 转 CoreML，先跑通 capture -> Metal preprocess -> CoreML depth 输出闭环；InfiniDepth CoreML 转换风险高，不进入首版范围。完整实现建议封装 `MacOSMetalDepthBridge.start() / next_depth() / stop()`，由 native bridge 管理 Metal/CoreML 对象生命周期、同步和资源池。

约束：不得把当前 PyTorch MPS 路线称为零拷贝；不得为了未来零 CPU 回读重写现有可运行 macOS capture provider；完整验证必须在真实 Apple Silicon Mac 上完成。

### 高画质缓冲输出

高画质缓冲输出是可选模式，不覆盖现有 latest-frame 实时模式。它允许输出端按配置延迟 100-500 ms，用 frame window 和 lookahead 提升补洞、边缘稳定和 temporal 质量，适合视频、电影、本地播放器和非交互式桌面内容，不适合游戏、鼠标精确操作或低延迟 VR 交互。

目标架构：

```text
Realtime mode:
Capture -> runtime latest-frame -> OpenXR / local output

Quality buffered mode:
Capture -> timestamped frame window -> high quality synthesis -> delayed presentation -> OpenXR / local output / 3D monitor
```

缓存规则：实时高画质缓冲严禁把画面帧落盘；优先 GPU resident buffer，其次 system memory ring buffer。GPU 和内存都无法稳定满足目标延迟/帧窗口时，应判定高画质缓冲不可用并提示用户降低延迟、分辨率或补洞质量，不能自动退化为硬盘缓存。

补洞目标应重新评估为 edge-aware fill、layered DIBR、temporal history fill、future-frame lookahead fill、multi-scale push-pull fallback 和 confidence blend。音频同步必须使用独立音频子系统，优先调研 Windows WASAPI loopback；不得复用 RTMP 现有采集作为 OpenXR/本地/3D 显示器的通用音频时钟层。

## 后续实施优先级

1. 完成 GUI live hot-save 到 `RuntimeSettingsSnapshot` 的直接发送路径，把 settings.yaml polling 降为持久化同步路径。
2. 收敛 Windows OpenXR 主路径到 D3D11 native：projection-layer 主画面和 RGB+D 核心 shader parity 已接入，继续做 VDXR 实机验收和必要 polish。
3. 执行 `docs/39` Panda3D glTF OpenXR migration gate：保持 default native/OpenGL 测试路径不变，显式启用 `D2S_GLTF_RENDERER=panda3d` 后先在真实头显/OpenXR runtime 下验证 D3D11/NV_DX 主路径可对 acquired Projection swapchain texture register/lock/render/unlock/release，并显示 Panda-rendered frame；再验证 OpenGL fallback 只在 Panda3D 与 OpenXR GL context 相同或已验证共享时通过。只有 GPU-only gate 通过后才考虑把 Panda3D 设为默认或删除 self-authored renderer 分支。
4. 保留并验证扩展后的 `docs/38` glTF 2.0 core compliance 路线：在 Panda3D gate 未通过前，已实现的 static primitive/material/texture/render-pass contract 仍是 native renderer 的稳定基线；如果 Panda3D gate 失败，再继续自研 TRS animation sampling、per-frame node matrix、animated primitive model matrix、skinning、morph targets、sparse/interleaved/normalized accessors、camera/multi-scene 和 diagnostics。
5. 继续 OpenXR controller parity：在 compliance layer/Panda facade 上保留 controller profile 覆盖、按键动画、模型偏移/旋转和手柄材质例外，实机验证 D3D11 color-space、HDR environment、screen reflection、laser 光柱和多个 controller model。
6. OpenGL 作为本地 viewer、当前测试默认和兼容 fallback：继续验证 `GlTensorPboUploader` / `CudaGlTextureUploader` 的 image/PBO/CPU fallback 日志，不再把 OpenGL uploader 当作 D3D11 方案；Panda3D 自身使用 OpenGL renderer，但主路径通过 NV_DX bridge 交给 D3D11 OpenXR swapchain；OpenGL fallback 必须通过同 context 或 shared context 证明，不能用 CPU/PBO/PIL/Numpy image staging 绕过。
7. 移除 TensorRT ORT / ONNX Runtime depth provider 的 CPU numpy 往返，实现真正 GPU zero-copy input/output。
8. 做 CUDA/ROCm capture zero-copy 硬件验证，只有实测无 CPU 中转后才允许把 metadata 标为 `zero_copy=True`。
9. 做 runtime scheduling/backpressure 回归验证：CUDA 默认同步、latest-frame overwrite/drop、非 CUDA 后端无误触发。
10. 清理兼容冗余：旧 snapshot/API 字段、debug-only 兼容字段；legacy parallax 乘数字段和 render-scale 数值/短写别名已清理，后续只需防回归。
11. 接入 DirectMLDepthProvider 作为 Windows 非 CUDA/ROCm/XPU GPU 的短期 ONNX Runtime 兜底，并明确该阶段不是捕获零拷贝。
12. 验证 Windows D3D11 texture -> D3D12 resource -> DirectML IoBinding 的原生零 CPU 回读链路。
13. 在真实 Apple Silicon Mac 上验证 ScreenCaptureKit -> Metal preprocess -> CoreML Distill Depth 的 macOS 零 CPU 回读原型。
14. 增加可选高画质缓冲输出骨架，默认关闭，先支持固定延迟、本地/OpenXR 共用 presentation scheduler，禁止实时帧落盘。
15. 继续 `docs/36` phase 2：证明 D3D11 native projection-layer 主画面在慢 runtime、慢 capture、慢/失败 effect worker 下复用 last-good frame，不阻塞 controller/head pose refresh、projection screen present 和 `xrEndFrame`。Quad layer 只保留为可选实验/诊断，不再作为完成判据。
16. 继续完善 network_stream 的 encoder transport contract，尤其是 RTMP/低延迟编码路径，但保持其只消费 packed synthesis 输出。

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
