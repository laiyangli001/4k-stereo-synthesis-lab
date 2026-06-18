# depth.py 功能迁移清单

本文用于记录 Desktop2Stereo `depth.py` 平替前的功能拆分，避免把模型推理、捕捉预处理、立体生成、GUI 设置和多后端兼容混在一起迁移。

结论先行：`depth.py` 不能直接删除。当前主流程可以逐步切到 `stereo_runtime`，但 `depth.py` 仍包含多模型加载、多后端 artifact 构建、旧 depth-shift SBS、temporal stabilizer、overlay 等功能。删除条件是下表所有必要项都有明确替代实现并通过 smoke/benchmark。

## 边界原则

| 模块 | 应负责 | 不应负责 |
|---|---|---|
| `capture.py` / D2S capture | 桌面/窗口捕捉；raw/BGRA/BGR 转 RGB；D2S 队列用 `frame_rgb`；必要时为 runtime 准备严格输入格式 | depth 模型选择；FP16/FP32 推理策略；ONNX/TRT 构建 |
| `stereo_runtime` | 模型 registry；模型目录推导；下载；ONNX dtype auto；ONNX/TRT 构建；depth provider；depth 推理；stereo synthesis | 捕捉；BGR/BGRA 颜色转换；猜测 Host 输入格式 |
| GUI/settings | 用户选择模型、backend、preset、质量参数；持久化配置 | 直接拼 ONNX/TRT 路径；直接管理 provider 内部 dtype |
| viewer/xrviewer | RGB + depth 显示、OpenXR session、swapchain、overlay 显示 | depth 模型推理和 artifact 构建 |

严格接口建议：

- D2S 队列合同保持：`(frame_rgb, depth, capture_start_time)`。
- `frame_rgb` 给 viewer/xrviewer 时保持 D2S 旧语义：RGB 图像帧，通常是 `CHW` tensor 或 `HWC` numpy，数值按 viewer 现有 `0..255` 上传逻辑处理。
- 进入 `stereo_runtime.DepthRuntime` 前，由 capture/main 显式准备 runtime 输入：`CHW` 或 `BCHW`、RGB、float、`0..1`、torch tensor。
- `stereo_runtime` 对 RGB 输入只做合同校验和模型输入预处理，不负责 HWC/CHW、numpy/tensor、uint8/float 的自动猜测。

## depth.py 职责拆分

| depth.py 功能 | 当前状态 | 目标归属 | 删除前要求 |
|---|---|---|---|
| raw/BGRA/BGR -> RGB | 已开始迁出 | `capture.py` | WindowsCapture、DXCamera、MSS 路径都能输出正确 RGB |
| Processing Resolution resize | 已开始迁出 | `capture.py` | 不改变 D2S viewer 队列语义，不影响 depth 推理分辨率 |
| FP16/FP32 推理 dtype 决策 | 部分覆盖 | `stereo_runtime` | GUI 的 FP16 选项转为 runtime config；ONNX dtype auto 生效 |
| 模型列表 `MODEL_MAPPING` | 已迁出基础版 | `stereo_runtime.model_registry` | 已覆盖 D2S 当前完整模型名和 Hugging Face ID；后续补模型能力标签 |
| 模型下载 / `from_pretrained` | 已迁出入口 | `stereo_runtime.model_artifacts` | 已有 `ensure_model_downloaded()`；网络下载待集成测试 |
| PyTorch provider | 已有通用基础版 | `stereo_runtime.depth_provider` | Transformers 标准 AutoModel 路径已支持；DA3/InfiniDepth 等特殊 API 待补 |
| ONNX 导出 | 已迁出库函数 | `stereo_runtime.onnx_export` | dtype auto/probe/export 已迁；真实大模型导出待集成测试 |
| TensorRT engine 构建 | 已迁出入口 | `stereo_runtime.model_artifacts` / `depth_trt_native_provider` | 已接 runtime 开关与 native TRT 构建入口；真实 engine 构建和全模型策略待补 |
| CoreML | 未迁 | 可选 `stereo_runtime` 后端或外置兼容层 | macOS 路径有替代或明确不支持 |
| OpenVINO | 未迁 | 可选 `stereo_runtime` 后端或外置兼容层 | XPU 路径有替代或明确不支持 |
| DirectML / MPS / XPU 特殊处理 | 未迁完 | runtime backend 选择层 | 不再散落在 GUI 或 capture |
| depth 模型输入尺寸 / patch align | 已部分覆盖 | depth provider | 不降低推理分辨率；保持 518/patch 对齐语义 |
| mean/std 归一化 | 部分覆盖 | depth provider | DPT/ZoeDepth/DepthPro 等特殊归一化策略补齐 |
| depth 后处理 normalize / invert / metric | 部分覆盖 | depth provider / postprocess | metric model 语义明确，不误归一化 |
| temporal depth stabilizer | 未完整迁 | `stereo_runtime.temporal` 或 depth-specific smoother | 可开关，场景变化重置，避免拖影 |
| old depth-shift stereo `make_sbs` | 基础版已迁出并通过单测 | `legacy_sbs.py` 或 viewer 兼容层 | legacy MJPEG stream 仍需做视觉等价 smoke |
| FPS overlay/font | 未完整迁 | viewer/overlay 层 | 不放在 depth runtime |
| `torch.compile` 管理 | 未迁完 | runtime backend 策略 | 与 TensorRT/ONNX 后端互斥关系清楚 |
| runtime unload/pause | 部分覆盖 | `DepthRuntime` | OpenXR pause/resume 不加载或占用模型 |
| D2S depth-only 队列 smoke | 已完成 fake provider smoke | `scripts/smoke/d2s_depth_runtime_smoke.py` | 仍需真实 PyTorch/ONNX/TRT provider smoke |

## 已有 stereo_runtime 覆盖能力

| 能力 | 文件 | 备注 |
|---|---|---|
| `DepthRuntime` / `StereoRuntime` 常驻对象 | `src/stereo_runtime/runtime.py` | 已有 rolling timing/report 基础 |
| Distill base PyTorch provider | `src/stereo_runtime/depth_provider.py` | 主要验证过 Distill 系列 |
| ONNX CUDA / IO Binding | `src/stereo_runtime/depth_onnx_provider.py` | 高性能路径之一 |
| Native TensorRT data_ptr 路径 | `src/stereo_runtime/depth_trt_native_provider.py` | 当前重点路径 |
| depth upsample | `src/stereo_runtime/depth_upsample.py` | bilinear/guided |
| 2-layer / occlusion-aware stereo synthesis | `src/stereo_runtime/synthesis.py` 等 | 超越 D2S depth-shift 的主线 |
| SBS/TAB/depth map 输出 | `src/stereo_runtime/output.py` | 不等价于 D2S 旧 `make_sbs(rgb, depth)` |

## 建议迁移顺序

1. 固化接口合同：capture 输出 D2S 队列 RGB；进入 `DepthRuntime` 前显式准备 `float 0..1 CHW/BCHW tensor`。
2. 完成 `ModelRegistry`：迁入 D2S `MODEL_MAPPING` 全列表。已完成基础版。
3. 完成通用模型下载与目录推导：`model_id + cache_dir -> model_dir`。已完成目录推导、下载入口和 artifact 准备框架。
4. 泛化 PyTorch provider：先覆盖 D2S 常用模型的 forward/normalize/postprocess 特例。已完成 Transformers 标准 AutoModel 基础版。
5. 泛化 ONNX export：保留智能 dtype auto 和 artifact 命名。已迁出库函数，脚本为薄 CLI。
6. 泛化 TensorRT engine 构建：Distill 先稳定，其它模型失败时明确报错，不静默降级。已接 runtime 开关、workspace 参数和 native TRT 构建入口。
7. 迁出 legacy `make_sbs`：只服务旧 MJPEG stream 或兼容 viewer，不再放在 depth runtime。基础版已完成并通过 `tests/test_legacy_sbs.py`。
8. 迁出/废弃 overlay/font：归 viewer/overlay 层。
9. 决定 CoreML/OpenVINO/DirectML 是否纳入第一阶段；不纳入则文档明确。
10. 全部调用点不再 import `depth.py` 后，把 `depth.py` 替换为迁移说明或删除。

## 删除 depth.py 的验收条件

- `Select-String` 或 CodeGraph 确认无 `from depth import` / `import depth` 外部调用。
- `tests/test_no_legacy_depth_imports.py` 持续防止新增外部 `depth.py` import。
- `DepthRuntime` 能完成 D2S 第一阶段 depth-only 队列合同。
- viewer/xrviewer 显示不变暗、不改变 RGB 数值语义。
- Distill base/large 的 PyTorch、ONNX CUDA、Native TensorRT 路径可用。
- 至少一个非 Distill 模型 PyTorch provider 可用，用于验证通用模型列表不是空壳。
- 旧 MJPEG stream 如仍保留，改用 `legacy_sbs.py` 或新 viewer 输出路径。
- 文档说明第一阶段暂不支持或已支持 CoreML/OpenVINO/DirectML。

## 已完成 smoke

```powershell
.\src\python3\python.exe -B scripts\smoke\d2s_depth_runtime_smoke.py --device cpu --width 64 --height 40 --target-height 32 --out -
.\src\python3\python.exe -m pytest tests\test_d2s_depth_runtime_smoke.py -q
```

该 smoke 使用 fake depth provider，不加载真实模型，用来固定 D2S 第一阶段队列合同：

```python
(frame_rgb, depth, capture_start_time)
```

验证点：

- raw BGRA 输入由 capture 侧转换成 RGB；
- viewer 队列里的 `frame_rgb` 保持 RGB 图像帧；
- runtime 输入由 capture/main 显式准备为 float `0..1` RGB tensor；
- `DepthRuntime` 常驻 provider 只加载一次；
- 输出 depth 与 runtime RGB 尺寸一致；
- `depth_result.timing` 包含 preprocess/model/postprocess/total。

真实 provider smoke 手动命令：

```powershell
.\src\python3\python.exe -B scripts\smoke\d2s_depth_runtime_smoke.py --real-provider --device cuda --backend pytorch_cuda --width 3840 --height 2160 --target-height 2160 --out -
.\src\python3\python.exe -B scripts\smoke\d2s_depth_runtime_smoke.py --real-provider --device cuda --backend onnx_cuda --width 3840 --height 2160 --target-height 2160 --out -
.\src\python3\python.exe -B scripts\smoke\d2s_depth_runtime_smoke.py --real-provider --device cuda --backend tensorrt_native --width 3840 --height 2160 --target-height 2160 --out -
```

默认不加 `--real-provider` 时不会加载模型，适合常规回归；加 `--real-provider` 才用于本机 PyTorch/ONNX/TensorRT 真实链路验证。
