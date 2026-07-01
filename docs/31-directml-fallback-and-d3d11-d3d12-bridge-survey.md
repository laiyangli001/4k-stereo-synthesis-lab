# ONNX Runtime DirectML 兜底与 D3D11-D3D12 零拷贝桥接调查报告

## 结论

本项目可以新增 `onnxruntime-directml` 作为 Intel、国产显卡、以及其它非 NVIDIA CUDA / AMD ROCm / Intel XPU 环境的 Windows GPU 推理兜底路线。它的价值是覆盖更多 Windows D3D12/DirectML 设备，不依赖 CUDA、ROCm 或厂商 Python GPU tensor 生态。

但 `onnxruntime-directml` 的 Python 成熟方案主要是 CPU/Numpy 输入进入 DirectML GPU 推理，不等于 DXGI 捕获画面能自动 GPU 零拷贝进入模型。真正的未来目标应分成两阶段：

```text
阶段 1：DXGI/Windows 捕获 -> CPU/Numpy 输入 -> ONNX Runtime DirectML -> GPU 推理
阶段 2：D3D11 捕获 texture -> D3D12 tensor buffer -> ONNX Runtime DirectML I/O binding
```

阶段 1 可以作为短期可落地方案。阶段 2 需要 C++/Rust 原生桥接，不能只靠 Python `onnxruntime-directml` 完成。

## 背景

Windows 捕获 API 的现实是：

```text
捕获侧仍然以 D3D11 surface / texture 为中心
计算/推理侧的新路线是 D3D12 / DirectML
中间需要互操作桥
```

典型长期目标链路是：

```text
D3D11 texture
-> shared handle / NT handle
-> D3D12 OpenSharedHandle 或 D3D11On12 / 共享资源桥
-> D3D12 resource
-> DirectML / ONNX Runtime DirectML
```

当前已验证的事实：

- DXGI Desktop Duplication 原生获取的是 D3D11 texture，而不是 D3D12 resource。
- Windows Graphics Capture 也是以 `Direct3D11CaptureFrame` / `IDirect3DSurface` 为核心。
- DirectML 是 D3D12 体系的机器学习 API。
- ONNX Runtime DirectML Python 包成熟可用，但默认接口以常规 ONNX tensor 输入为主。

## 官方资料要点

ONNX Runtime DirectML EP 是官方 Windows GPU 推理后端。官方文档说明 DirectML EP 使用 DirectML 加速 ONNX 模型，并强调它不需要安装厂商专用扩展，覆盖广泛 Windows GPU 硬件。文档同时说明 DirectML 现在处于 sustained engineering，新 Windows 项目建议关注 WinML。

安装命令：

```bash
pip install onnxruntime-directml
```

DirectML EP 需要 DirectX 12 capable device；官方示例硬件包括 NVIDIA Kepler 及以上、AMD GCN 1st Gen 及以上、Intel Haswell 及以上、Qualcomm Adreno 600 及以上。DirectML 随 Windows 10 1903 引入。

DirectML EP 的 C API 有两种创建方式：

```text
OrtSessionOptionsAppendExecutionProvider_DML(options, device_id)
SessionOptionsAppendExecutionProvider_DML1(options, IDMLDevice*, ID3D12CommandQueue*)
```

后者可以传入自建 DirectML device 和 D3D12 command queue，是未来原生桥接可利用的入口。

ONNX Runtime 的 Device Tensor 文档提供了 DirectML resource 输入路线：

```text
ID3D12Resource* d3d_buffer
-> CreateGPUAllocationFromD3DResource(d3d_buffer, &dml_resource)
-> Ort::Value::CreateTensor(memory_info_dml, dml_resource, ...)
```

这说明 D3D12 resource 作为 ORT DirectML 输入是官方支持的，但这属于原生 API 级别，不是 Python `InferenceSession` 默认路径。

DirectML 原生 binding 使用 `IDMLBindingTable` 和 `DML_BUFFER_BINDING`。`DML_BUFFER_BINDING` 绑定的是 `ID3D12Resource` 的字节范围，符合“模型输入 tensor 是 D3D12 buffer”这一设计。

D3D12 `OpenSharedHandle` 可打开 shared resource、heap、fence 等对象。D3D11/D3D12 互操作还可能涉及 `CreateSharedHandle`、NT handle、shared fence、D3D11On12 等机制。

## 本项目现状

本地项目当前环境检测：

```text
onnxruntime: installed, version 1.26.0
onnxruntime_directml: not installed
available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

现有 ONNX Runtime depth provider 位置：

```text
src/stereo_runtime/depth_onnx_provider.py
```

当前 provider 选择逻辑是：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
```

也就是说，现在没有 `DmlExecutionProvider` 路径。

GUI 设备枚举已尝试通过 `torch_directml` 暴露 DirectML 设备：

```text
src/gui/devices.py
```

但这只是设备枚举 / PyTorch DirectML 视角，不等于 ONNX Runtime DirectML provider 已接入。

## 短期实现方案：DirectMLDepthProvider

目标：让没有 CUDA / ROCm / XPU 的 Windows GPU 也能跑 ONNX 深度模型。

新增 provider 建议：

```text
src/stereo_runtime/providers/windows/directml.py
```

或如果保持平台目录较少，也可放在：

```text
src/stereo_runtime/providers/directml.py
```

推荐行为：

1. 检测 `onnxruntime` 可用 providers：

```python
import onnxruntime as ort
available = ort.get_available_providers()
```

2. 如果 `DmlExecutionProvider` 不在列表中，报明确错误：

```text
ONNX Runtime DirectML provider unavailable; install onnxruntime-directml or use a DirectML-enabled package.
```

3. 创建 session：

```python
providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(str(onnx_path), providers=providers)
```

4. 验证 active provider：

```python
active = session.get_providers()
if "DmlExecutionProvider" not in active:
    raise RuntimeError(...)
```

5. 输入仍按当前 ONNX provider 的 CPU/Numpy 路径处理。短期不要宣称零拷贝。

6. 注意 DirectML EP 限制：

- 同一个 session 不应多线程并发 `Run`。
- DirectML EP 不支持 ORT memory pattern optimizations 和 parallel execution，原生 API 需要禁用这些选项；Python 创建 session 时也应避免引入不兼容配置。
- 模型输入形状固定更有利于 DirectML 性能。建议优先使用已固定 input size 的 ONNX artifact。
- 算子支持受 DirectML EP 和 ONNX opset 约束，需要用实际模型验证。

## 包管理风险

`onnxruntime-gpu`、`onnxruntime-directml`、`onnxruntime` 在 Python 中通常都导入为同一个模块名：

```python
import onnxruntime as ort
```

本项目当前内置的是 `onnxruntime-gpu`，DirectML 包未安装。直接混装可能导致覆盖 DLL/provider 或 provider 列表异常。

建议打包策略：

```text
NVIDIA build: onnxruntime-gpu + TensorRT/CUDA provider
DirectML build: onnxruntime-directml + DmlExecutionProvider
CPU build: onnxruntime
```

如果必须做单包，需要在部署阶段明确选择一种 ORT 发行包，不应同时假设 GPU/CUDA 和 DirectML provider 都可用。

## 长期目标：D3D11 捕获纹理到 D3D12 DirectML Tensor

长期目标链路：

```text
DXGI Desktop Duplication / Windows Graphics Capture
-> ID3D11Texture2D
-> shared handle / NT handle
-> ID3D12Resource
-> D3D12 compute preprocess
-> DirectML tensor buffer
-> ONNX Runtime DirectML IoBinding / Device Tensor
```

关键任务：

1. 捕获层输出 GPU 资源

已经在本地 `windows-capture` fork 中验证可以从 Rust/PyO3 层暴露：

```text
texture_ptr
shared_texture_ptr
shared_handle
```

这一步只解决“Python/Rust 能拿到 D3D11 texture 或 shared handle”，不等于能直接喂给 ONNX Runtime。

2. D3D11 到 D3D12 互操作

可选路线：

```text
A. D3D11 shared NT handle -> ID3D12Device::OpenSharedHandle
B. D3D11On12 互操作
C. D3D11 texture GPU copy -> D3D12 owned resource
```

需要重点验证：

- shared handle 必须是 D3D12 可打开的 NT handle，优先使用 `IDXGIResource1::CreateSharedHandle`。
- 老式 `IDXGIResource::GetSharedHandle` 可能不适合 D3D12/跨 API 生命周期管理。
- 同 adapter / LUID 必须匹配，否则 shared resource 打开或 GPU copy 会失败。
- 需要 shared fence 或其它同步机制，避免 D3D12 读取 D3D11 尚未完成写入的纹理。

3. Texture 到 tensor buffer 的 GPU 预处理

模型输入通常不是 BGRA texture，而是：

```text
float16/float32
NCHW 或 NHWC
固定尺寸
归一化后的 tensor buffer
```

所以需要 D3D12 compute shader 做：

```text
BGRA/RGBA texture sample
resize
normalize
layout transform
write ID3D12Resource buffer
```

这一步不是 CPU copy，但通常是一次 GPU compute/copy，不应称为“完全无拷贝”。更准确说是“零 CPU 回读”。

4. ORT DirectML Device Tensor / IoBinding

最终输入应是：

```text
ID3D12Resource tensor buffer
-> OrtDmlApi::CreateGPUAllocationFromD3DResource
-> Ort::Value::CreateTensor
-> IoBinding 绑定输入
```

这需要 C++/Rust 原生模块，因为 Python `onnxruntime-directml` 默认不暴露这个 D3D12 resource binding 工作流。

## 推荐实施路线

### 里程碑 1：DirectML CPU 输入兜底

目标：尽快让非 CUDA/ROCm/XPU Windows GPU 可运行。

工作项：

- 新增 `DirectMLDepthProvider`。
- 调整 provider factory，在 DirectML 设备标签或 Windows 非 CUDA/ROCm 场景下选择它。
- 增加启动诊断输出：`available_providers`、`active_providers`。
- 增加测试：provider 缺失时报明确错误；provider 激活后写入 `DepthProviderInfo.execution_provider`。
- 文档明确：该阶段是 DirectML GPU 推理，不是捕获零拷贝。

### 里程碑 2：DirectML artifact 与模型验证

目标：确认 InfiniDepth ONNX 在 DirectML EP 上可运行且性能可接受。

工作项：

- 固定输入尺寸，避免动态 shape 影响 DirectML 编译优化。
- 确认 ONNX opset / 算子是否被 DirectML EP 支持。
- 比较 CPU、DirectML、CUDA/TensorRT 的首帧编译时间、稳定帧耗时、显存占用。
- 明确 fallback 策略：DirectML 不支持模型时回 CPU，并在控制台显示原因。

### 里程碑 3：D3D11 shared handle 原生桥验证

目标：证明 D3D11 捕获 texture 可以在同 adapter 上被 D3D12 打开/消费。

工作项：

- 在 `windows-capture` fork 中优先使用 NT shared handle。
- 编写最小 Rust/C++ demo：D3D11 texture -> shared handle -> D3D12 OpenSharedHandle。
- 加入 fence 同步验证。
- 输出 D3D12 resource 尺寸、格式、adapter LUID 验证信息。

### 里程碑 4：D3D12 GPU 预处理

目标：不回 CPU，把捕获 texture 转成模型输入 tensor buffer。

工作项：

- D3D12 compute shader 实现 resize / normalize / BGRA to tensor layout。
- 输出 `ID3D12Resource` buffer。
- 验证 tensor buffer 内容与 CPU 预处理近似一致。

### 里程碑 5：ORT DirectML IoBinding

目标：把 D3D12 tensor buffer 绑定到 ONNX Runtime DirectML。

工作项：

- 使用 ORT C/C++ API 创建 DirectML session。
- 使用 `SessionOptionsAppendExecutionProvider_DML1` 传入自建 DML device 和 D3D12 command queue。
- 用 `CreateGPUAllocationFromD3DResource` 创建 ORT 输入 tensor。
- 使用 IoBinding 执行推理。
- 输出与 Python DirectML provider 一致性校验。

## 命名建议

短期 provider 不建议叫 `WindowsCaptureDirectML`，因为它并不改变捕获路径，只改变推理后端。

建议命名：

```text
推理 provider: DirectMLDepthProvider
捕获 source: DXGIDesktopDuplication
未来完整链路模式: DXGIDesktopDuplicationDirectML 或 WindowsDirectMLPipeline
```

只有当完整链路包含：

```text
捕获 D3D11 texture
-> D3D12 tensor bridge
-> DirectML inference
```

才适合用 `DirectML` 命名完整 runtime 模式。

## 风险与注意事项

- DirectML 覆盖广，但不是所有 Windows 显卡都能稳定高性能运行目标模型。
- 国产显卡是否可用，取决于 WDDM、D3D12、DirectML、驱动质量和模型算子支持。
- Python `onnxruntime-directml` 是最短路径，但不能解决 texture 零拷贝输入。
- 原生零 CPU 回读链路开发成本较高，需要 D3D11/D3D12、DirectML、ORT C API、同步和 shader 预处理经验。
- `onnxruntime-gpu` 与 `onnxruntime-directml` 包管理需要分发行策略，避免互相覆盖。

## 参考资料

- ONNX Runtime DirectML Execution Provider: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX Runtime Install: https://onnxruntime.ai/docs/install/
- ONNX Runtime Device Tensors: https://onnxruntime.ai/docs/performance/device-tensor.html
- DirectML `DML_BUFFER_BINDING`: https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_buffer_binding
- D3D11On12 `D3D11On12CreateDevice`: https://learn.microsoft.com/en-us/windows/win32/api/d3d11on12/nf-d3d11on12-d3d11on12createdevice
- D3D12 `ID3D12Device::OpenSharedHandle`: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-opensharedhandle
- Desktop Duplication API: https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api