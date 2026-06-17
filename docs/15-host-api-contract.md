# 宿主 API 契约

本文定义 GUI、Desktop2Stereo runtime、OpenXR host 与本仓库核心库之间的职责边界。目标是让外部宿主可以稳定调用算法能力，同时避免把 GUI、捕获、模型管理和立体合成参数混在一起。

## 核心原则

- 本仓库是算法 / 推理 / 立体合成核心库，不是完整产品 runtime。
- 宿主负责捕获、窗口管理、GUI、OpenXR session/swapchain、配置持久化和产品级错误提示。
- 核心库负责 depth provider、stereo synthesis、OpenXR per-eye render core、输出格式打包、benchmark 和视觉回归工具。
- Preset 只映射立体生成和 OpenXR core 参数，不控制 depth 模型、depth 推理分辨率、模型路径或下载路径。
- Depth provider、ONNX session、TensorRT engine、OpenXR session 都必须常驻复用，不允许每帧重建。

## 数据流 1：宿主已有 RGB + Depth

适用于宿主已经从其他地方得到深度图的场景。

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo

config = stereo_config_for_preset("cinema", output_format="half_sbs")
result = synthesize_stereo(rgb, depth, config, temporal_state=temporal_state)
```

输入契约：

| 名称 | 形状 | 数值范围 | 责任方 |
|---|---|---|---|
| `rgb` | `CHW` 或 `BCHW` | 推荐 `0..1` float tensor | 宿主 |
| `depth` | `HW`、`BHW` 或 `B1HW` | 推荐 `0..1` float tensor | 宿主或 depth provider |
| `config` | `StereoConfig` | 由 preset helper 或显式配置生成 | 核心库 / 宿主 |
| `temporal_state` | `TemporalState` | 每条视频流一个常驻状态 | 宿主创建并复用 |

输出契约：

| 字段 | 说明 |
|---|---|
| `left_eye` | 原始分辨率左眼 tensor |
| `right_eye` | 原始分辨率右眼 tensor |
| `sbs` | 按 `output_format` 打包后的输出 |
| `debug_info` | 后端、mask、耗时或调试 tensor，取决于 `debug_output` |

尺寸规则：

| 输出格式 | 输出尺寸 |
|---|---|
| `half_sbs` | `W x H`，左右眼各压到半宽后拼接 |
| `full_sbs` | `2W x H`，左右眼保持原始宽度后拼接 |
| `half_tab` | `W x H`，左右眼各压到半高后上下拼接 |
| `full_tab` | `W x 2H`，左右眼保持原始高度后上下拼接 |
| `mono` | `W x H`，返回左眼 |
| `depth_map` | `W x H`，返回匹配到输出尺寸的深度图 |
| `anaglyph` | `W x H`，红青等双色输出 |
| `interleaved` | `W x H`，行交错输出 |
| `leia` | `W x H`，列交错输出 |

## 数据流 2：宿主只有 RGB

适用于 GUI 或实时 runtime 希望核心库负责 depth 推理的场景。

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo
from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_lab.temporal import TemporalState

depth_provider = create_depth_provider(
    DepthProviderConfig(backend="tensorrt_native", device="cuda")
)
depth_provider.load()

config = stereo_config_for_preset("cinema", output_format="half_sbs")
temporal_state = TemporalState()

for rgb in frames:
    depth = depth_provider.predict(rgb)
    result = synthesize_stereo(rgb, depth, config, temporal_state=temporal_state)
```

常驻对象要求：

| 对象 | 创建频率 | 说明 |
|---|---|---|
| `DepthProvider` | 进程或模型切换时创建一次 | 内部持有模型、ONNX session 或 TensorRT engine |
| `StereoConfig` | 模式或参数改变时创建 | 不需要每帧重新创建 |
| `TemporalState` | 每条输入流创建一个 | 场景切换或源切换时可 reset |
| OpenXR session/swapchain | 宿主 runtime 管理 | 本仓库不创建完整 OpenXR runtime |

禁止行为：

- 不要每帧调用 `create_depth_provider()`。
- 不要每帧重新 load ONNX session 或 TensorRT engine。
- 不要为了提速降低 `depth_resolution=518` 或修改 `294x518` 输入路径。
- 不要把模型产物写入 Desktop2Stereo 的模型目录。

## Preset 边界

Preset API：

```python
from stereo_lab import (
    stereo_config_for_preset,
    openxr_config_for_preset,
    stereo_config_for_auto_mode,
    openxr_config_for_auto_mode,
)
```

Preset 可以控制：

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

Preset 不允许控制：

- depth 模型名称或模型 ID
- depth 推理分辨率
- ONNX 路径
- TensorRT engine 路径
- cache/download 目录
- TensorRT / ONNX / PyTorch backend 选择
- CUDA DLL PATH
- Desktop capture 或 OpenXR session/swapchain 生命周期

如果 GUI 需要切换 depth backend，应该通过单独的 `DepthProviderConfig` 和模型管理 UI 完成，不应该把它塞进 `Cinema / Game / Still Image / Debug / Auto` preset。

## Auto 模式边界

核心库提供低风险分类器和轻量状态机。场景信号采集必须由宿主异步完成，不能放在桌面捕获、depth 推理或 stereo synthesis 热路径里同步执行。

重要限制：只有用户选择 `auto` 模式时，宿主才应该启动异步场景检测。用户手动选择 `cinema`、`game_low_latency`、`still_image_hq` 或 `debug_export` 时，宿主应直接使用对应 preset，不启动检测线程，也不调用 `AutoModeRuntime.update()`。

```python
from stereo_lab import AutoModeRuntime, AutoModeSignals, auto_detection_required, stereo_config_for_preset

if auto_detection_required(user_selected_preset):
    auto_runtime = AutoModeRuntime()
    signals = AutoModeSignals(
        frame_motion_score=motion,
        scene_cut_score=scene_cut,
        still_duration_s=still_seconds,
        gpu_3d_util=gpu_3d_2s_avg,
        gpu_video_decode_util=video_decode_2s_avg,
        input_activity=input_activity_2s_avg,
        idle_seconds=idle_seconds,
        audio_active=audio_active,
        foreground_process=process_name,
        fullscreen=is_fullscreen,
        maximized=is_maximized,
        openxr_active=is_openxr,
        user_export_action=is_export,
        latency_pressure=latency_pressure,
        target_fps=target_fps,
    )
    decision = auto_runtime.update(signals, dt_s=sample_dt)
    config = stereo_config_for_preset(decision.preset, output_format="half_sbs")
else:
    config = stereo_config_for_preset(user_selected_preset, output_format="half_sbs")
```

异步采集要求：

- GPU 3D、Video Decode、键鼠输入、音频、窗口状态等系统指标由宿主后台线程采集。
- 采集线程只在 `auto` 模式启动；离开 `auto` 模式时应停止或休眠。
- 建议采样周期 `100-250 ms`，用 `2-3` 秒滑动平均或指数平均后生成 `AutoModeSignals`。
- 渲染/推理线程只读取最新信号快照，调用 `AutoModeRuntime.update()`，不得同步查询系统 API。
- 前台进程名只作为低权重 hint，不能依赖大白名单。
- 信号冲突时保持上一次模式或回到 `cinema`，不要频繁跳变。

状态机行为：

- `AutoModeRuntime` 内部执行连续样本确认、hold 时间和快速升级到 `game_low_latency/debug_export`。
- 从 `game_low_latency` 回到 `cinema` 或 `still_image_hq` 会受 hold 限制，避免来回抖动。
- 参数渐变仍由宿主执行；核心库返回 `blend_seconds` 作为建议。

## OpenXR 边界

本仓库只提供 roll-adaptive per-eye synthesis 和投影辅助函数，不提供完整 OpenXR runtime。

```python
from stereo_lab import openxr_config_for_preset, render_openxr_stereo

config = openxr_config_for_preset(
    "cinema",
    screen_roll=current_screen_roll_radians,
)
result = render_openxr_stereo(rgb, depth, config)
```

宿主负责：

- OpenXR instance/session/swapchain。
- frame timing。
- pose/FOV 获取。
- 左右眼 texture 提交。
- runtime 错误恢复和 UI 提示。

核心库负责：

- 基于 `rgb + depth + screen_roll` 生成左右眼 tensor。
- 提供 `build_openxr_eye_mvp()` 等矩阵工具。
- 不把固定 SBS 输出伪装成 VR 输出。

## 版本兼容策略

- 新增 preset 字段时，必须先更新本文档和 `tests/test_presets.py`。
- 修改 preset 默认值时，最后一步必须跑视觉回归基准。
- 性能优化不允许改变 depth 推理分辨率、RGB resize、antialias 或 normalize 语义。
- 如果必须改变这些质量相关语义，需要单独出 depth 质量评估和同场景视觉对比。
