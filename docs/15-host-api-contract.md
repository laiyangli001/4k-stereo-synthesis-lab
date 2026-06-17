# Host API 合同

本文定义 GUI、Desktop2Stereo runtime、OpenXR host 与本仓库核心库之间的职责边界。目标是让外部 host 能稳定调用算法能力，同时避免把捕捉、GUI、模型管理和立体合成参数混在一起。

## 职责边界

本仓库负责：

- depth provider 创建、常驻和推理封装；
- stereo synthesis：`fast`、`quality_4k`、`hq_4k`；
- OpenXR per-eye render core；
- 输出格式打包；
- preset 默认参数；
- benchmark、smoke 和视觉回归工具。

外部 host 负责：

- 桌面 / 窗口 / 播放器 / 游戏画面捕捉；
- GUI、配置持久化、用户交互；
- 完整 OpenXR session / swapchain / frame timing；
- 系统指标异步采集；
- 产品级错误提示和恢复策略。

## 数据流 1：Host 已有 RGB + Depth

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo
from stereo_lab.temporal import TemporalState

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

## 数据流 2：Host 只有 RGB

推荐外部 host 使用 `StereoLabRuntimeConfig` 作为边界对象。Desktop2Stereo 负责下载模型并确定 `model_id` / `model_dir`，本仓库从 `model_dir` 推导 ONNX 与 TensorRT artifact 路径，并负责后续转换、构建、预处理、推理和立体合成。

```python
from stereo_lab import (
    StereoLabRuntimeConfig,
    depth_provider_config_from_runtime,
    stereo_config_from_runtime,
    synthesize_stereo,
)
from stereo_lab.depth_provider import create_depth_provider
from stereo_lab.temporal import TemporalState

runtime_config = StereoLabRuntimeConfig(
    model_id="lc700x/Distill-Any-Depth-Base-hf",
    model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
    mode="movie",
    stereo_quality="quality_4k",
    output_format="half_sbs",
    depth_backend="auto",
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

depth_provider = create_depth_provider(depth_provider_config_from_runtime(runtime_config))
depth_provider.load()

config = stereo_config_from_runtime(runtime_config)
temporal_state = TemporalState()

for rgb in frames:
    depth = depth_provider.predict(rgb)
    result = synthesize_stereo(rgb, depth, config, temporal_state=temporal_state)
```

常驻对象要求：

| 对象 | 创建频率 | 说明 |
|---|---|---|
| `DepthProvider` | 进程启动或模型切换时创建一次 | 内部持有模型、ONNX session 或 TensorRT engine |
| `StereoLabRuntimeConfig` | 模型、模式或参数改变时创建 | Host 传 `model_id` 和 `model_dir`，artifact 路径由本仓库推导 |
| `StereoConfig` | 由 runtime config 转换生成 | 不需要每帧创建 |
| `TemporalState` | 每条输入流一个 | 源切换或场景重置时可 reset |
| OpenXR session/swapchain | Host runtime 管理 | 本仓库不创建完整 runtime |

禁止行为：

- 不要每帧调用 `create_depth_provider()`。
- 不要每帧重新 load ONNX session 或 TensorRT engine。
- 不要为了提速降低 `depth_resolution=518` 或修改 `294x518` 输入路径。
- 不要让 GUI 直接拼 ONNX / TensorRT 文件名；GUI 只传 `model_id` 和 `model_dir`。

Artifact 推导规则：

| 输入 | 规则 |
|---|---|
| `model_id` | Desktop2Stereo 模型列表/下载逻辑给出的 Hugging Face ID |
| `model_dir` | Desktop2Stereo 下载完成后的本地模型目录 |
| ONNX fp16 | `{model_dir}/model_fp16_294x518.onnx` |
| ONNX fp32 | `{model_dir}/model_fp32_294x518.onnx` |
| TensorRT engine | `{model_dir}/model_fp16_294x518.trt` |

ONNX dtype 默认使用 `onnx_dtype="auto"`：优先 fp16 dummy forward 检查，失败则回退 fp32；fp32 也失败时停止并报告错误。

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

## Auto 模式

`auto` 只在用户选择 auto preset 时启动检测。手动选择 `cinema`、`game_low_latency`、`still_image_hq` 或 `debug_export` 时，host 不应启动场景检测线程。

```python
from stereo_lab import AutoModeRuntime, AutoModeSignals, auto_detection_required

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
from stereo_lab import openxr_config_for_preset
from stereo_lab.openxr_render import render_openxr_stereo

config = openxr_config_for_preset("cinema", screen_roll=screen_roll)
result = render_openxr_stereo(rgb, depth, config)
```

## Smoke 验证

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --openxr --preset cinema --screen-roll 0.25 --out -
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset auto --out -
```
