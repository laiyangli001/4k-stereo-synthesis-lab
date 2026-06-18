# 多平台 Depth Provider 目录设计

目标是在 `stereo_runtime` 内支持 NVIDIA / AMD / Intel / Apple / CPU 多平台，但避免重新变成旧 `depth.py` 式的大杂烩。

## 分层原则

`runtime.py` 只负责常驻对象、timing、report、pause/resume 和调用 `provider.predict_profile()`。

`adapter.py` 只负责把 GUI/settings/host 参数转换成 runtime config。

`providers/factory.py` 负责根据 Host 已经传入的 `vendor/backend/device` 选择 provider。

各平台 provider 只处理自己的 runtime、artifact 和后端细节。

设备枚举、厂商识别和 capture backend 选择发生在 capture 之前，属于 Host / GUI / capture bootstrap 层，不属于 `stereo_runtime`。

## 推荐目录

```text
src/stereo_runtime/
  runtime.py
  adapter.py
  model_registry.py
  model_artifacts.py
  onnx_export.py

  providers/
    __init__.py
    base.py
    factory.py

    nvidia/
      __init__.py
      pytorch_cuda.py
      onnx_cuda.py
      tensorrt_native.py
      tensorrt_ort.py
      cuda_utils.py

    amd/
      __init__.py
      pytorch_rocm.py
      directml.py
      rocm_utils.py

    intel/
      __init__.py
      openvino.py
      onnx_dml.py
      xpu_utils.py

    apple/
      __init__.py
      coreml.py
      mps_pytorch.py

    cpu/
      __init__.py
      pytorch_cpu.py
      onnx_cpu.py
```

## 平台默认策略

`vendor` 必须由 Host 传入。`stereo_runtime` 不主动枚举 GPU，不调用 `torch_directml.is_available()`，不根据系统环境猜测当前厂商。`backend=auto` 只表示“在 Host 已确认的 vendor 内选择默认 provider”。

| vendor | backend=auto 优先级 |
|---|---|
| `nvidia` | Native TensorRT -> ONNX CUDA -> PyTorch CUDA |
| `amd` | PyTorch ROCm -> DirectML -> ONNX CPU |
| `intel` | OpenVINO -> XPU PyTorch -> ONNX CPU |
| `apple` | CoreML -> MPS PyTorch -> CPU |
| `cpu` | ONNX CPU -> PyTorch CPU |

## Artifact 目录

当前兼容路径仍保留：

```text
models/models--owner--repo/
  model_fp16_294x518.onnx
  model_fp32_294x518.onnx
  model_fp16_294x518.trt
```

未来多平台推荐路径：

```text
models/models--owner--repo/
  onnx/
    model_fp16_294x518.onnx
    model_fp32_294x518.onnx

  tensorrt/
    nvidia_sm86/
      model_fp16_294x518.trt

  openvino/
    intel_gpu/
      model_fp16_294x518.xml
      model_fp16_294x518.bin

  coreml/
    apple_neural_engine/
      model_fp16_294x518.mlpackage

  directml/
    model_fp16_294x518.onnx
```

迁移期规则：

- 优先读新路径；
- 找不到再读旧路径；
- 不移动用户已有模型文件；
- 不降低推理分辨率或改变 `294x518` patch-aligned 语义。

## 当前阶段落地

第一阶段只建立 `providers/` 包骨架和兼容导出，不搬动已验证的 NVIDIA provider 实现。

现有文件仍保留：

- `depth_provider.py`
- `depth_onnx_provider.py`
- `depth_trt_native_provider.py`
- `depth_trt_provider.py`

新增 `providers/*` 作为未来平台分层入口，逐步迁移时必须保持旧 import 兼容。

当前 D2S 的设备识别仍由 GUI / Host 层完成：

- DirectML：`torch_directml.is_available()`；
- NVIDIA CUDA：设备 label 含 `CUDA` 且非 ROCm；
- AMD ROCm：`torch.version.hip is not None`；
- Apple：`torch.backends.mps.is_available()`；
- Intel：`torch.xpu.is_available()`；
- CPU：Host fallback。

这些检测结果后续应由 Host 转成 `vendor` / `device` / `backend` 参数传给 `stereo_runtime`。

## 非目标

- 不把 CoreML / OpenVINO / DirectML 塞进 `depth_provider.py` 主文件；
- 不在 `runtime.py` 内写平台判断；
- 不在 `stereo_runtime` 内做设备枚举或 capture backend 选择；
- 不让 GUI 拼后端 artifact 路径；
- 不为了平台兼容降低 depth inference 分辨率。
