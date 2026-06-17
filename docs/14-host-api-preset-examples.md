# 宿主 API 与预设调用示例

本文说明 GUI、Desktop2Stereo runtime 或 OpenXR host 应该如何调用本核心库。

本仓库定位是核心算法库。宿主项目负责桌面/窗口捕获、GUI、OpenXR session/swapchain、配置持久化、安装包，以及产品级错误提示 UI。

## 预设 API

GUI 不建议手动填满每一个 `StereoConfig` 字段，而是优先使用预设。

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo

config = stereo_config_for_preset(
    "cinema",
    output_format="half_sbs",
    overrides={"depth_strength": 2.2},
)

result = synthesize_stereo(rgb, depth, config)
```

支持的预设名称：

```text
auto
cinema
game_low_latency
still_image_hq
debug_export
```

常用别名也可以接受：

```text
game -> game_low_latency
still image / hq -> still_image_hq
debug -> debug_export
```

## 预设默认值

| 预设 | Backend | Temporal | 适用输出场景 |
|---|---|---|---|
| `cinema` | `quality_4k` | 开启 + 自动重置 | 电影、播放器、稳定视频 |
| `game_low_latency` | `quality_4k` | 轻量开启 + 自动重置 | 游戏、桌面交互、快速运动 |
| `still_image_hq` | `hq_4k` | 关闭 | 截图、暂停帧、静态 2D 转 3D |
| `debug_export` | `quality_4k` | 开启 + debug 输出 | 视觉回归、导出、算法检查 |

如果直接使用 `auto`，它会映射到 `cinema`。真正的自动模式应该调用下面的分类器，并由宿主 runtime 状态机负责防抖、保持时间和参数渐变。

## Auto 模式

核心库提供一个低风险分类器和轻量状态机。GPU、键鼠、音频、窗口状态等系统指标必须由宿主后台线程异步采集并做 `2-3` 秒滑动平均；捕获/推理线程只读取最新快照，不能同步查询系统 API。

只有用户选择 `auto` 模式时才启动检测。手动选择 `cinema`、`game_low_latency`、`still_image_hq` 或 `debug_export` 时，直接使用固定 preset，不启动检测线程。

```python
from stereo_lab import AutoModeRuntime, AutoModeSignals, auto_detection_required, stereo_config_for_preset

user_selected_preset = "auto"

if auto_detection_required(user_selected_preset):
    auto_runtime = AutoModeRuntime()
    signals = AutoModeSignals(
        frame_motion_score=0.42,
        scene_cut_score=0.1,
        still_duration_s=0.0,
        gpu_3d_util=0.72,
        gpu_video_decode_util=0.03,
        input_activity=0.85,
        idle_seconds=0.2,
        audio_active=True,
        foreground_process="Unknown3DApp.exe",
        fullscreen=True,
        maximized=True,
        openxr_active=False,
        user_export_action=False,
        latency_pressure=0.8,
        target_fps=120.0,
    )
    decision = auto_runtime.update(signals, dt_s=0.25)
    config = stereo_config_for_preset(decision.preset, output_format="half_sbs")
else:
    config = stereo_config_for_preset(user_selected_preset, output_format="half_sbs")

print(decision.preset)
print(decision.reason)
print(decision.hold_seconds)
print(decision.scores)
```

分类器输出示例：

```python
AutoModeDecision(
    preset="game_low_latency",
    reason="high 3d/input/latency behavior",
    hold_seconds=2.0,
    blend_seconds=0.2,
    require_consecutive_frames=4,
    scores={"game": 6.7, "video": 0.7, "still": 1.0},
)
```

宿主 runtime 规则：

```text
只有 Auto 模式启动异步场景检测；手动 preset 不启动检测。
系统指标异步低频采样，不要阻塞捕获、depth 推理或 stereo synthesis。
不要因为单次采样立刻切换模式。需要连续多次满足条件，切换后保持当前模式 2-5 秒，并对参数做渐变。
检测到高 GPU 3D / 高频输入 / 高延迟压力时，可以快速切到 Game / Low Latency。
从 Game / Low Latency 回到 Cinema 或 Still Image / HQ 时应更慢，避免来回抖动。
```

## RGB + Depth 转 Stereo

宿主已经有 RGB 和 depth 时：

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo

config = stereo_config_for_preset("cinema", output_format="half_sbs")
result = synthesize_stereo(rgb, depth, config)

left_eye = result.left_eye
right_eye = result.right_eye
packed = result.sbs
debug = result.debug_info
```

## RGB 转 Depth 再转 Stereo

宿主希望核心库负责估计 depth 时：

```python
from stereo_lab.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_lab import stereo_config_for_preset, synthesize_stereo

depth_provider = create_depth_provider(
    DepthProviderConfig(backend="tensorrt_native", device="cuda")
)
depth_provider.load()

config = stereo_config_for_preset("cinema", output_format="half_sbs")

for rgb in frames:
    depth = depth_provider.predict(rgb)
    result = synthesize_stereo(rgb, depth, config)
```

重要要求：

```text
Depth provider 必须创建并 load 一次后常驻复用。
不要每帧重新构造 provider/session/engine。
```

## OpenXR Core

对 OpenXR 来说，不要把固定 SBS 假装成 VR 输出。宿主 runtime 应该把 runtime pose/FOV/roll 传给自己的 OpenXR session/swapchain 层。

当前核心库提供 roll-adaptive per-eye synthesis：

```python
from stereo_lab import openxr_config_for_preset, render_openxr_stereo

config = openxr_config_for_preset(
    "cinema",
    screen_roll=current_screen_roll_radians,
)

result = render_openxr_stereo(rgb, depth, config)
left_eye = result.left_eye
right_eye = result.right_eye
```

结合 Auto 模式：

```python
from stereo_lab import AutoModeSignals, openxr_config_for_auto_mode, render_openxr_stereo

decision, config = openxr_config_for_auto_mode(
    AutoModeSignals(openxr_active=True, frame_motion_score=0.05),
    screen_roll=current_screen_roll_radians,
)

result = render_openxr_stereo(rgb, depth, config)
```

本仓库不提供完整 OpenXR session/swapchain runtime。

## 预设视觉回归

在生成最终视觉回归集之前，可以先跑宿主 API smoke，确认 preset、常驻状态和 stereo 输出链路能被外部宿主按契约调用：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out outputs\host_api_smoke_cinema.json
```

在只想验证调用链、不写文件的受限环境中，可以使用：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
```

验证 OpenXR per-eye core 调用链：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --openxr --preset cinema --screen-roll 0.25 --out -
```

演示 Auto 模式宿主状态机接入，不采集真实系统指标，只消费模拟的后台采样快照：

```powershell
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset auto --out -
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset game_low_latency --out -
```

如需验证真实 depth provider 链路，再显式加 `--rgb` 和 `--auto-depth`：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset cinema --output-format half_sbs --out outputs\host_api_smoke_4k_native.json
```

修改预设默认值前，先生成固定视觉回归集：

```powershell
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --out-dir outputs\visual_regression\preset_cinema
```

生成指定预设的视觉回归集：

```powershell
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset cinema --out-dir outputs\visual_regression\preset_cinema
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset game_low_latency --out-dir outputs\visual_regression\preset_game
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset still_image_hq --out-dir outputs\visual_regression\preset_still_hq
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset debug_export --out-dir outputs\visual_regression\preset_debug
```

重点检查：

```text
contact_sheet_labeled.png
visual_regression_report.json
```

调 still image 参数时，建议比较：

```text
cinema
still_image_hq
debug_export
```

重点观察：

- 边缘是否撕裂
- 是否有空洞
- 是否出现重复纹理
- Half-SBS 中线是否异常
- UI / 文字是否变形
- temporal 是否有拖影
- OpenXR 中是否出现错误深度导致的不适

## 安全规则

- 不要把降低 depth 推理分辨率当成性能优化捷径。
- 不要在没有单独 depth 质量评估的情况下修改 resize / antialias / normalize 语义。
- 不要把换模型当成当前模型路径的优化。
- 不要把模型产物写入 Desktop2Stereo 的模型目录。
- 修改预设时必须同时使用视觉回归和正式 benchmark。
