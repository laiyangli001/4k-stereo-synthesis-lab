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
| Local viewer | `src/viewer/viewer.py`, `viewer_runtime.py` | GLFW/ModernGL 本地显示、SBS/TAB/Anaglyph/Interleaved/Leia、CUDA PBO 上传 |
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
```

## docs/28 流程到工程实现映射

| docs/28 阶段 | 当前工程实现 | 当前状态 / 迁移边界 |
|---|---|---|
| 步骤 1：Capture Input | `src/capture/session.py`, `src/capture/runners.py`, `src/capture/types.py`, `src/capture/backends/*` | 已有 `CapturedFrame` metadata、copy mode、source/capture size 记录；真实 CUDA/ROCm zero-copy 仍需硬件验证 |
| 步骤 2：Resolve Render Size | `src/stereo_runtime/render_size.py`, `src/stereo_runtime/pipeline.py`, GUI Render Scale 配置 | 已按 4K/3K/2K/1K 固定 scale 档位解析；4K级输入按比例缩放并保持输入宽高比；历史 numeric / short alias Render Scale 输入已清理并有测试防回归 |
| 步骤 3：Resize RGB To Render Size | `src/capture/preprocess.py`, `src/stereo_runtime/pipeline.py` | 已在 pipeline 边界把 capture frame 解析到 render_size；跨设备 fallback 仍需继续扩展验证矩阵 |
| 步骤 4：Depth Estimation | `src/stereo_runtime/depth_provider.py`, `src/stereo_runtime/depth_onnx_provider.py`, `src/stereo_runtime/providers/*`, `model_registry.py` | 已支持多 provider / backend；provider 内部尺寸可以不同，但输出必须回到 render_size |
| 步骤 5：Depth Postprocess | `src/stereo_runtime/runtime.py`, `src/stereo_runtime/synthesis.py` | 当前使用 normalized / relative depth 处理和轻量 postprocess；不能改写为 metric Z 路径 |
| 步骤 6：Resolve Parallax Budget | `src/stereo_runtime/parallax.py`, `adapter.py`, `settings_snapshot.py` | 已有预算表、短边插值和超宽保护；legacy multiplier 只应留在显式兼容入口 |
| 步骤 7：Disparity Field | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `openxr_render.py` | 主语义为 `depth_response * max_disparity_px`；near/far 和 response 必须在 metadata 中可追踪 |
| 步骤 8：Stereo Warp | `src/stereo_runtime/synthesis.py`, `baseline_shift.py`, `layers.py`, OpenXR shader paths | full synthesis 已有 baseline/layered DIBR；OpenXR D3D11 direct shader 仍需追平 OpenGL direct 的核心 DIBR 质量语义 |
| 步骤 9：Mask and Hole Fill | `src/stereo_runtime/synthesis.py`, `layers.py` | 已有 edge-aware / directional fill 路径；AI inpainting 只作为未来离线/质量候选 |
| 步骤 10：Temporal Stabilization | `src/stereo_runtime/runtime.py`, `pipeline.py`, `settings_snapshot.py` | 已有 scene/render_size/source/settings reset 机制；temporal 只做稳定，不改变 parallax budget |
| 步骤 11：Output Pack / Viewer Upload | `src/stereo_runtime/output.py`, `output_triton.py`, `src/viewer/*`, `src/xr_viewer/*`, `src/streaming/*` | 本地/网络消费 packed output，OpenXR 支持 full synthesis 与 RGB+depth direct；D3D11 direct parity 和 RTMP/低延迟 stream 是后续工程 |

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
-> 更新 settings.yaml 中 Stereo Preset / Stereo Quality / IPD / Stereo Scale / Convergence / Depth Strength / Max Shift Ratio / Temporal / Hole Fill / Edge 等字段
```

后续规范要求：

```text
settings.yaml hot save 只能作为兼容机制。
新热更新路径应收敛到 RuntimeSettingsSnapshot + settings_update_q。
GUI 写 settings.yaml 后，runtime 可以继续由 StereoHotReloader 兼容读取。
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
| Stereo | `Stereo Preset`, `Stereo Quality`, `Synthetic View`, `Depth Strength`, `Convergence`, `Max Shift Ratio`, `Stereo Scale`, `IPD` | 当前 legacy 立体参数 |
| Synthesis postprocess | `Temporal Strength`, `Scene Reset Threshold`, `Edge Dilation`, `Mask Feather Radius`, `Hole Fill Mode`, `Edge Threshold`, `Foreground Scale`, `Anti-aliasing` | 稳定、mask、补洞、depth postprocess |
| Output | `Display Mode`, `Anaglyph Method`, `Cross Eyed`, `Fill 16:9`, `Fix Viewer Aspect`, `VSync` | 本地/推流封装与显示 |
| OpenXR | `XR Preview Window`, `Controller Model`, `Environment Model` | OpenXR viewer 行为 |
| Streaming | `Stream Protocol`, `Streamer Port`, `Stream Quality`, `Stream Key`, `Stereo Mix`, `CRF`, `Audio Delay` | 网络推流配置 |

规范要求：

```text
1. settings.yaml 是持久化配置，不是线程间实时共享对象。
2. runtime 内部使用规范化后的 dataclass，不直接依赖原始 YAML 字段名。
3. legacy 字段 IPD / Stereo Scale / Max Shift Ratio 继续读取，但 normalized-depth 新路径已通过 adapter / 兼容入口映射到 max_disparity_px / parallax_budget；旧字段不作为核心强度语义。
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
ensure_model_downloaded()
-> huggingface_hub.snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=..., force_download=...)
```

artifact 命名：

```text
model_fp16_{height}x{width}.onnx
model_fp32_{height}x{width}.onnx
model_fp16_{height}x{width}.trt
model_fp16_{height}x{width}.mgx
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
4. local_files_only=True 时不得静默联网。
5. force_download/force_rebuild 必须由 GUI 显式选项或命令行显式指定。
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
MIGraphX 失败时是否 fallback PyTorch ROCm 必须在 provider_info.fallback_reason 中记录。
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
-> OpenXRRuntimeResult(source_rgb=rgb_frame, depth=prepared_depth, left_eye=rgb_frame, right_eye=rgb_frame)
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

当前 legacy 位移参数：

```text
depth_strength
convergence
ipd / ipd_mm
max_shift_ratio
stereo_scale
```

规范目标参数：

```text
depth_response(depth, convergence)
max_disparity_px
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

CUDA upload 当前规则：

```text
use_cuda=True 时初始化 CUDART_GL 和 PBO。
离散 GPU: 优先 pinned host staging + PBO，减少 H2D upload 开销。
集成 GPU: 避免 pinned staging overhead，走 texture.write 等更合适路径。
```

规范：

```text
本地 viewer 接收 packed frame，不重新定义立体公式。
本地 viewer 可以做 display mode shader/presentation，但不能改变 synthesis 的 max_disparity_px 语义。
CUDA PBO 是 upload 优化，不是立体算法。
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

OpenGL/PBO path：

```text
runtime_direct_opengl_pbo 可用于 CUDA tensor -> OpenGL texture upload。
当前应在 debug/log 中记录 upload backend 和 eye texture size。
```

规范：

```text
OpenXR full synthesis 不接受 SBS/TAB 作为 viewer 输入。
OpenXR debug 可额外导出 SBS 预览，但不作为 swapchain 输入。
viewer 建屏优先使用结构化 `output_display_size`，legacy `debug_info["runtime_output_display_size"]` 只作为兼容 fallback；eye texture 优先使用结构化 `output_eye_size`。
Direct path 的 legacy uniforms 必须由 adapter 从规范参数转换。
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
OpenXRStateController.update_runtime_config(ipd/depth_ratio/convergence/stereo_scale/max_shift_ratio/screen_roll)
```

现状说明：

```text
当前 OpenXR hot reload 仍是 legacy 参数语义。
当前规范目标是 RuntimeSettingsSnapshot。
短期 adapter 负责 legacy 字段转换。
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
4. Viewer upload 优先 device-to-device 或 PBO/pinned staging。
5. 只有网络编码、CPU-only backend 或兼容 viewer 需要时才转 CPU numpy。
```

### 内存与队列

```text
raw_q maxsize=1，防止 capture 堆积。
runtime_q maxsize=1，防止 render/stream 堆积。
OpenXR hard idle 时清 raw_q/runtime_q 并停止 active capture session。
render_size 变化时必须释放或重建 textures/buffers/temporal state。
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
5. network_stream packed frame resize/encoder metadata tests。
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
| Parallax formula | 已有 `resolve_parallax_budget(render_width, render_height, preset)`，normalized-depth 主路径使用 `max_disparity_px` / `parallax_budget_preset` 语义，并保留 legacy 字段兼容入口 | 清理旧 `IPD/stereo_scale/depth_strength/max_shift_ratio` 作为核心强度乘数的兼容冗余；该项属于 Future Work |
| RuntimeSettingsSnapshot | 已有 `RuntimeSettingsSnapshot`、`settings_update_q`、帧边界应用、热更新分级、结构化 result 字段 | GUI live hot-save 仍需进一步收敛为直接发送 snapshot；settings.yaml + StereoHotReloader 只保留为兼容路径 |
| Capture metadata | 已有 `CapturedFrame` / `FrameCopyMode`，event/polling runner 会携带 source、device、dtype、copy_mode、capture_size 等 metadata，并进入 runtime debug | 真实硬件 CUDA/ROCm zero-copy 仍需设备验证后才能把路径标成 true zero-copy |
| Capture preprocess device contract | 已显式处理 numpy / CPU tensor / CUDA tensor / ROCm tensor 形态，并记录 origin/output device 与 transfer metadata | 跨设备 fallback 和硬件路径仍需按目标机器补充验证矩阵 |
| OpenXR direct uniforms | 已通过 adapter/snapshot 路径生成规范参数到 legacy shader uniforms 的转换，并输出 `legacy_shader_uniforms` / resolved disparity debug | D3D11 native direct shader 仍需追平 OpenGL direct shader 的完整 DIBR 质量语义；这是独立 follow-up |
| Render Size / 4K scale tier | 已收敛为固定 scale 档位 Render Scale：非 4K 保持 capture_size；4K 级输入按 4K/3K/2K/1K 稳定 scale 档位解析，并保持横屏、竖屏、16:10、DCI 4K、常见 4K 超宽比例；判断排除面积不足的窄高/1440p 超宽；旧 numeric / short alias Render Scale 输入已清理为默认回退 | 继续通过测试防止重新引入 `0.75`、`75%`、`2K` 等用户侧别名；无新的运行时语义待办 |
| Network stream | MJPEG/legacy stream 已消费 packed frame，并引入 `EncoderProfile` 描述 transport 侧 resize、pixel format、quality/FPS 等 | RTMP/更低延迟编码仍是后续工程；不能重新定义立体参数语义 |

## 后续实施优先级

1. 完成 GUI live hot-save 到 `RuntimeSettingsSnapshot` 的直接发送路径，把 settings.yaml polling 降为明确兼容机制。
2. 做 D3D11 native OpenXR direct shader parity，使其核心 DIBR 质量语义追平 OpenGL direct shader。
3. 做 CUDA/ROCm capture zero-copy 硬件验证，只有实测无 CPU 中转后才允许把 metadata 标为 `zero_copy=True`。
4. 清理兼容冗余：旧 snapshot/API 字段、debug-only 兼容字段、legacy parallax 乘数字段；render-scale 数值/短写别名已清理，后续只需防回归。
5. 继续完善 network_stream 的 encoder transport contract，尤其是 RTMP/低延迟编码路径，但保持其只消费 packed synthesis 输出。

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
