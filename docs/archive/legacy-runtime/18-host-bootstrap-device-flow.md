# Host 设备检测与 Runtime 参数传递流程

本文定义 D2S / Host 在 capture 前完成设备检测、厂商识别和 backend 选择的流程。`stereo_runtime` 只消费 Host 已经决定好的参数，不做设备枚举。

## 总体流程

```text
GUI / Host 启动
-> 枚举设备
-> 识别 vendor
-> 选择 capture backend
-> 选择 depth backend
-> 写入 settings
-> main.py 读取 settings
-> runtime_config_from_d2s_settings()
-> DepthRuntime / StereoRuntime
```

## HostDeviceInfo

建议 Host 内部使用一个统一结构承载检测结果：

```python
HostDeviceInfo(
    vendor="nvidia",                  # nvidia/amd_rocm/directml/intel/apple/cpu
    device="cuda:0",                  # cuda:0/mps/xpu:0/cpu/directml:0
    device_label="CUDA 0: NVIDIA GeForce RTX 2060",
    capture_backend="WindowsCaptureCUDA",
    depth_backend="tensorrt_native",  # tensorrt_native/onnx_cuda/pytorch_cuda/openvino/coreml/directml
    supports_tensorrt=True,
    supports_openvino=False,
    supports_coreml=False,
    supports_directml=False,
)
```

这是 Host / GUI / capture bootstrap 层的数据结构，不属于 `stereo_runtime` 主动检测职责。

## D2S 当前识别方式

| 平台 | 当前检测方式 | 建议 vendor |
|---|---|---|
| NVIDIA CUDA | device label 含 `CUDA` 且 `torch.version.hip is None` | `nvidia` |
| AMD ROCm | device label 含 `CUDA` 且 `torch.version.hip is not None` | `amd_rocm` |
| DirectML | `torch_directml.is_available()` | `directml` |
| Apple MPS | `torch.backends.mps.is_available()` | `apple` |
| Intel XPU | `torch.xpu.is_available()` | `intel` |
| CPU | fallback | `cpu` |

D2S 当前 GUI 还会用 device label 控制高级选项显隐：

| 条件 | GUI 选项 |
|---|---|
| `CUDA` 且非 ROCm | TensorRT |
| `MPS` | CoreML |
| `XPU` | OpenVINO |
| `DirectML` | DirectML / 通用路径 |

这部分仍属于 Host/GUI。

## Capture Backend 选择

Capture backend 必须在进入 `stereo_runtime` 前确定。

| vendor | 建议 capture backend |
|---|---|
| `nvidia` | `WindowsCaptureCUDA` |
| `amd_rocm` | `WindowsCaptureROCm` |
| `directml` | `DXCamera` 或 `DesktopDuplication` |
| `intel` | `DXCamera` 或 `DesktopDuplication` |
| `apple` | `ScreenCaptureKit` |
| `cpu` | `DXCamera` / 平台默认 |

`stereo_runtime` 不选择 capture backend。

## Depth Backend 选择

Host 根据 vendor、用户设置和可用 artifact 选择 depth backend，然后写入 settings。

| vendor | backend=auto 推荐 |
|---|---|
| `nvidia` | `tensorrt_native` -> `onnx_cuda` -> `pytorch_cuda` |
| `amd_rocm` | `pytorch_cuda` 语义上的 ROCm PyTorch，后续可命名为 `pytorch_rocm` |
| `directml` | `directml` |
| `intel` | `openvino` |
| `apple` | `coreml` 或 `mps_pytorch` |
| `cpu` | `onnx_cpu` 或 `pytorch_cpu` |

第一阶段已验证的真实 provider：

- `pytorch_cuda`
- `onnx_cuda`
- `tensorrt_native`

其它 backend 进入 provider factory 后应给出明确未实现错误，不应静默降级。

## Settings 字段建议

当前已有字段：

```yaml
Depth Model: Distill-Any-Depth-Base
Computing Device: 0
TensorRT: true
Recompile TensorRT: false
CoreML: false
OpenVINO: false
```

建议新增或由 Host 内部生成后传入 runtime：

```yaml
Device Vendor: nvidia
Device Label: "CUDA 0: NVIDIA GeForce RTX 2060"
Runtime Device: "cuda:0"
Depth Backend: "tensorrt_native"
Capture Backend: "WindowsCaptureCUDA"
```

`runtime_config_from_d2s_settings()` 只读取这些字段并生成 runtime config，不做硬件探测。

## 边界规则

- Host/GUI 负责设备枚举、厂商识别、capture backend 选择、GUI 控件显隐。
- capture 负责 raw/BGRA/BGR 到 RGB 图像帧。
- `stereo_runtime` 负责模型准备、depth provider、depth 推理、stereo synthesis。
- viewer/OpenXR 负责显示、swapchain、overlay 和 frame timing。
- 不能把设备检测逻辑塞进 `stereo_runtime`。
- 不能让 GUI 拼 ONNX/TRT/CoreML/OpenVINO artifact 路径。
