# Host API 合同

本文定义 GUI、Desktop2Stereo runtime、OpenXR host 与本仓库核心库之间的职责边界。目标是让外部 host 能稳定调用算法能力，同时避免把捕捉、GUI、模型管理和立体合成参数混在一起。

## 职责边界

本仓库负责：

- Desktop2Stereo 兼容模型 registry；
- 模型下载、本地模型目录推导和 artifact 准备；
- ONNX dtype auto 探测、ONNX 导出和 TensorRT engine 构建；
- depth provider 创建、常驻和推理封装；
- stereo synthesis：`fast`、`quality_4k`、`hq_4k`；
- OpenXR per-eye render core；
- 输出格式打包；
- preset 默认参数；
- benchmark、smoke 和视觉回归工具。

外部 host 负责：

- 用户选择模型、backend 和 preset；
- settings 持久化；
- 桌面 / 窗口 / 播放器 / 游戏画面捕捉；
- 捕捉侧颜色前处理，例如 BGR/BGRA 转 RGB；
- GUI 和用户交互；
- 完整 OpenXR session / swapchain / frame timing；
- 系统指标异步采集；
- 产品级错误提示和恢复策略。

## 数据流 1：Host 已有 RGB + Depth

```python
from stereo_runtime import stereo_config_for_preset, synthesize_stereo
from stereo_runtime.temporal import TemporalState

config = stereo_config_for_preset("cinema", output_format="half_sbs")
temporal_state = TemporalState()

result = synthesize_stereo(rgb, depth, config, temporal_state=temporal_state)
```

输入约定：

| 参数 | 形状 | 数值范围 | 责任方 |
|---|---|---|---|
| `rgb` | `CHW` 或 `BCHW` | 推荐 `0..1` float tensor | Host |
| `depth` | `HW`、`BHW` 或 `B1HW` | 推荐 `0..1` float tensor | Host 或 depth provider |
| `config` | `StereoConfig` | 由 preset helper 或显式配置生成 | 核心库 / Host |
| `temporal_state` | `TemporalState` | 每条视频流一个常驻状态 | Host 创建并复用 |

输出约定：

| 字段 | 含义 |
|---|---|
| `left_eye` | 原始分辨率左眼 tensor |
| `right_eye` | 原始分辨率右眼 tensor |
| `sbs` | 按 `output_format` 打包后的输出 |
| `debug_info` | 后端、mask、耗时或调试 tensor，取决于 `debug_output` |

## 数据流 2：Host 只有 RGB，且 D2S 仍使用 RGB + Depth 渲染

Desktop2Stereo 第一阶段接入建议使用 depth-only runtime，只替换 `depth.py` 的模型下载、artifact 准备和 depth inference 职责。`main.py` 到 viewer / xrviewer 的队列合同保持不变：

```python
(frame_rgb, depth, capture_start_time)
```

```python
from stereo_runtime import DepthRuntime, DepthRuntimeConfig, ModelRegistry

registry = ModelRegistry.default()
spec = registry.get("Distill-Any-Depth-Base")

runtime_config = DepthRuntimeConfig(
    model_id=spec.model_id,
    cache_dir="./models",
    depth_backend="auto",
    onnx_dtype="auto",
    depth_upsample="bilinear",
    depth_upsample_edge_strength=0.35,
)

runtime = DepthRuntime(runtime_config)
runtime.load()

for frame_rgb, capture_start_time in frames:
    result = runtime.predict_depth_frame(frame_rgb)
    depth_q.put((frame_rgb, result.depth, capture_start_time))
```

`result.timing` 返回 depth preprocess/model/postprocess/total 耗时；`runtime.to_report()` 返回 provider、artifact、rolling stats 和显存信息。

## 数据流 3：Host 只有 RGB，且直接需要 Stereo/SBS

推荐外部 host 使用 `StereoRuntimeConfig` 作为 RGB -> depth -> stereo/SBS 的完整边界对象。Host 只提供模型选择和 runtime 参数；本仓库根据 `model_id/cache_dir/export_height/export_width/onnx_dtype` 推导模型目录与 artifact 路径，并负责下载、转换、构建、预处理、推理和立体合成。

```python
from stereo_runtime import (
    StereoRuntime,
    StereoRuntimeConfig,
)

runtime_config = StereoRuntimeConfig(
    model_id="lc700x/Distill-Any-Depth-Base-hf",
    cache_dir="./models",
    mode="movie",
    stereo_quality="quality_4k",
    output_format="half_sbs",
    depth_backend="auto",
    depth_upsample="bilinear",
    depth_upsample_edge_strength=0.35,
    depth_strength=2.0,
    convergence=0.0,
    ipd=0.064,
    max_shift_ratio=0.05,
    layers=2,
    occlusion=True,
    symmetric=True,
    hole_fill="edge_aware",
    temporal=True,
    temporal_strength=0.75,
    auto_reset_temporal=True,
    edge_threshold=0.04,
    edge_dilation=2,
    fused=True,
)

runtime = StereoRuntime(runtime_config)
runtime.load()

for rgb_frame in frames:
    result = runtime.process_rgb_frame(rgb_frame)
```

`result.timing` 每帧返回：

| 字段 | 说明 |
|---|---|
| `depth_preprocess_ms` | RGB frame 到 depth 模型输入 |
| `depth_model_ms` | depth 模型推理 |
| `depth_postprocess_ms` | depth normalize / upsample 等后处理 |
| `depth_total_ms` | depth provider 总耗时 |
| `synthesis_ms` | stereo synthesis + output packing |
| `total_ms` | 单帧总耗时 |

`runtime.to_report()["rolling_stats"]` 返回滚动窗口统计：

- 每个 timing stage 的 `latest/min/max/mean/median/p90/p99`；
- 由 `total_ms` 换算的 latest / mean / p90 / p99 FPS；
- CUDA memory allocated/reserved/peak 的滚动统计。

### RGB frame 输入合同

Desktop2Stereo 的 `capture.py` / `main.py` 继续负责桌面、窗口、游戏画面捕捉，以及捕捉侧颜色前处理，例如 BGR/BGRA 转 RGB。对本仓库来说，Host 每帧只传一个语义对象：已经完成颜色前处理的当前 RGB 图像帧 `rgb_frame`。

Host 不需要、也不应该向本仓库传递 backend 细节：

- TensorRT / Triton / ONNX backend 输入 tensor 名称或绑定细节。

这些由本仓库内部的 depth provider/preprocess 层负责。这样后续接 Triton 时，外部仍然只传 `rgb_frame`，而不是让 D2S GUI 或 capture 层暴露 backend 细节。

Host 只需要保证：

- `rgb_frame` 是当前捕捉到的完整图像帧；
- D2S 已完成捕捉侧颜色前处理，语义上必须是 RGB 画面；
- 保持源分辨率，例如 4K 输入就传 4K frame；
- 不要提前缩到 depth 模型尺寸；
- 不要为了性能改动 depth inference 分辨率语义。

本仓库内部负责：

- 根据 `ModelRegistry` 将 GUI 模型名解析为模型 spec / Hugging Face ID；
- 根据 `model_id/cache_dir` 推导本地模型目录；
- 下载或确认本地模型文件；
- 复用已有 ONNX dtype auto 探测逻辑导出 ONNX；
- 复用 native TensorRT build 逻辑生成 engine；
- 从 RGB frame 开始准备 depth provider 输入；
- 按 `294x518` 路径执行 depth 模型预处理；
- 运行 depth provider；
- 将 depth 上采回 RGB 源分辨率，默认 `bilinear`，可选 `guided`；
- 为 TensorRT native、ONNX CUDA、PyTorch CUDA、未来 Triton backend 打包各自需要的输入；
- 执行立体合成和输出打包；
- 在 capture source、分辨率或场景变化时配合 Host 重置 `TemporalState`。

本仓库不负责：

- 桌面/窗口/显示器捕捉；
- BGR/BGRA 转 RGB；
- 窗口裁剪、monitor 选择、DPI 处理；
- capture fallback 策略。
- 用户配置文件写入策略；
- GUI 控件布局和产品级错误弹窗。

常驻对象要求：

| 对象 | 创建频率 | 说明 |
|---|---|---|
| `ModelRegistry` | 进程启动时创建或使用默认单例 | 覆盖 D2S 当前完整模型列表，供 GUI 和 runtime 解析模型 |
| `DepthRuntime` | 进程启动或模型/backend/分辨率切换时创建一次 | D2S 第一阶段首选调用对象，只输出 depth，保持 RGB + depth 渲染合同 |
| `StereoRuntime` | 进程启动或模型/模式切换时创建一次 | 完整 RGB -> depth -> stereo/SBS 调用对象，内部持有 provider、stereo config、temporal state |
| `DepthProvider` | 进程启动或模型切换时创建一次 | 内部持有模型、ONNX session 或 TensorRT engine |
| `DepthRuntimeConfig` / `StereoRuntimeConfig` | 模型、backend、模式或参数改变时创建 | Host 传模型选择和 cache/runtime 参数，artifact 路径由本仓库推导 |
| `StereoConfig` | 由 runtime config 转换生成 | 不需要每帧创建 |
| `TemporalState` | 每条输入流一个 | 源切换或场景重置时可 reset |
| OpenXR session/swapchain | Host runtime 管理 | 本仓库不创建完整 runtime |

禁止行为：

- 不要每帧创建 `StereoRuntime`。
- 不要每帧调用 `create_depth_provider()`。
- 不要每帧重新 load ONNX session 或 TensorRT engine。
- 不要为了提速降低 `depth_resolution=518` 或修改 `294x518` 输入路径。
- 不要让 GUI 直接拼 Hugging Face cache 目录、ONNX 路径或 TensorRT 文件名；GUI 只传模型选择和 runtime 参数。

Artifact 推导规则：

| 输入 | 规则 |
|---|---|
| GUI 模型名 | 由 `ModelRegistry` 解析为 `DepthModelSpec` |
| `model_id` | `DepthModelSpec` 给出的 Hugging Face ID |
| `cache_dir` | 默认 `./models`，可由 host 覆盖 |
| `model_dir` | `stereo_runtime` 按 D2S 规则由 `model_id/cache_dir` 推导 |
| ONNX fp16 | `{model_dir}/model_fp16_294x518.onnx` |
| ONNX fp32 | `{model_dir}/model_fp32_294x518.onnx` |
| TensorRT engine | `{model_dir}/model_fp16_294x518.trt` |

ONNX dtype 默认使用 `onnx_dtype="auto"`：优先 fp16 dummy forward 检查，失败则回退 fp32；fp32 也失败时停止并报告错误。

`prepare_model_artifacts()` 应复用并泛化已有 ONNX dtype auto、artifact 命名和 native TensorRT build 能力；它不是让 GUI 或 host 自行拼路径。

### Desktop2Stereo settings 桥接

D2S GUI 可以继续保存旧字段，但进入本仓库时应统一转换为 `DepthRuntimeConfig` / `StereoRuntimeConfig`。推荐入口：

```python
from stereo_runtime import runtime_config_from_d2s_settings

runtime_config = runtime_config_from_d2s_settings(
    settings,
    cache_dir="./models",
    device="cuda",
)
```

桥接规则：

| D2S settings 字段 | runtime 字段 | 说明 |
|---|---|---|
| `Depth Model` | `model_id` | 可以传 GUI 模型名或 HF ID，由 `ModelRegistry` 解析 |
| `TensorRT` | `depth_backend` / `build_trt_engine` | `true` 时走 `tensorrt_native` 并允许构建 engine |
| `Recompile TensorRT` | `force_rebuild_trt` | 只影响 engine 重建 |
| `FP16` | `onnx_dtype` | 只作为导出请求；真实可用性仍由 dtype auto / probe 判断 |
| `Display Mode` | `output_format` | `Half-SBS` / `Full-SBS` 等映射为 runtime 输出格式 |
| `Run Mode` | `mode` | 映射到 movie/game/image/auto |
| `Depth Strength` | `depth_strength` | stereo 合成参数 |
| `Convergence` | `convergence` | stereo 合成参数 |
| `IPD` | `ipd` | stereo 合成参数 |

GUI 不应直接控制 provider 内部 dtype、ONNX session 输入绑定、TensorRT tensor address、artifact 文件名或模型 cache 子目录。

## 输出格式

| `output_format` | 输出尺寸 | 说明 |
|---|---|---|
| `half_sbs` | `W x H` | 左右眼各压到半宽后拼接 |
| `full_sbs` | `2W x H` | 左右眼保持原宽后拼接 |
| `half_tab` | `W x H` | 左右眼各压到半高后上下拼接 |
| `full_tab` | `W x 2H` | 左右眼保持原高后上下拼接 |
| `mono` | `W x H` | 返回左眼 |
| `depth_map` | `W x H` | 返回匹配输出尺寸的深度图 |
| `anaglyph` | `W x H` | 红青等双色输出 |
| `interleaved` | `W x H` | 行交错输出 |
| `leia` | `W x H` | 列交错输出 |

Half-SBS 和 Full-SBS 必须来自同一组左右眼，只允许最后打包尺寸不同。

Depth 上采与 SBS 下采：

- depth 模型推理保持 `294x518` 路径，不允许为了性能降低推理分辨率语义；
- depth 输出会先上采回 RGB 源分辨率，再进入 stereo synthesis；
- `depth_upsample="bilinear"` 是默认兼容模式；
- `depth_upsample="guided"` 是可选 edge-aware/guided 上采，用 RGB 边缘轻量约束 depth 边界；
- `half_sbs` / `half_tab` 的 torch fallback 使用 `area` 下采，更接近 D2S 的 Half 输出语义；
- `full_sbs` / `full_tab` 不做下采，只拼接原始左右眼。

## Preset 边界

Preset 可以控制 stereo / OpenXR core 参数：

- `backend`
- `layers`
- `occlusion`
- `symmetric`
- `hole_fill`
- `temporal`
- `output_format`
- `debug_output`
- `depth_strength`
- `convergence`
- `ipd`
- `max_shift_ratio`
- `temporal_strength`
- `auto_reset_temporal`
- `scene_reset_threshold`
- `reset_cooldown_frames`
- `foreground_scale`
- `depth_antialias_strength`
- `edge_dilation`
- `edge_threshold`
- `cross_eyed`
- `anaglyph_method`
- `refine`
- `fused`
- OpenXR core 的 `screen_roll` 和 `padding_mode`

Preset 不控制：

- depth 模型名称或模型 ID；
- depth 推理分辨率；
- ONNX 路径；
- TensorRT engine 路径；
- 模型下载策略；
- capture source；
- GUI 状态；
- OpenXR session / swapchain 生命周期。

模型选择、下载和 artifact 准备属于 runtime config / model registry 层，不属于 preset 层。

## Auto 模式

`auto` 只在用户选择 auto preset 时启动检测。手动选择 `cinema`、`game_low_latency`、`still_image_hq` 或 `debug_export` 时，host 不应启动场景检测线程。

```python
from stereo_runtime import AutoModeRuntime, AutoModeSignals, auto_detection_required

if auto_detection_required(selected_preset):
    runtime = AutoModeRuntime()
    decision = runtime.update(signals)
```

系统指标必须由 host 异步采集并聚合，不允许阻塞捕捉、depth inference 或 stereo synthesis 热路径。

推荐信号：

- GPU 3D 使用率；
- GPU Video Decode 使用率；
- 键鼠输入活跃度；
- idle seconds；
- audio active；
- fullscreen / maximized；
- frame motion score；
- latency pressure；
- target FPS。

进程名只能作为低权重 hint，不应维护大规模游戏白名单。

## OpenXR 合同

固定 SBS/TAB 不能代替真正 OpenXR 输出。OpenXR host 应调用 per-eye render core，并由 host 管理 session、swapchain、projection layer 和 frame timing。

```python
from stereo_runtime import openxr_config_for_preset
from stereo_runtime.openxr_render import render_openxr_stereo

config = openxr_config_for_preset("cinema", screen_roll=screen_roll)
result = render_openxr_stereo(rgb, depth, config)
```

## Smoke 验证

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --openxr --preset cinema --screen-roll 0.25 --out -
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset auto --out -
```
