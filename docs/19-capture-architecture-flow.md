# Capture Architecture Flow

本文记录当前 `src/capture` 拆分后的工作架构和数据流。它描述 Desktop2Stereo host/capture 层如何选择后端、运行捕获循环、把 raw frame 转成 RGB，并交给 `stereo_runtime` depth runtime。

## 模块分层

```text
src/capture/
  __init__.py                 公共 re-export 入口
  types.py                    CaptureConfig / Protocol / callback 类型
  factory.py                  按 OS + Capture Tool 选择后端和 runner
  runners.py                  polling 型捕获循环
  preprocess.py               raw BGR/BGRA -> RGB + resize + depth runtime 输入准备
  geometry.py                 monitor/window 几何 helper，当前预留/逐步抽共用
  backends/
    windows_capture_event.py  WindowsCapture / CUDA / ROCm 事件式捕获
    windows_dxcamera.py       DXCamera / wincam polling 捕获
    windows_desktop_duplication.py  DXGI Desktop Duplication polling 捕获
    macos_screencapturekit.py ScreenCaptureKit 捕获
    macos_coregraphics.py     CoreGraphics fallback + cursor overlay
    linux_mss.py              Linux Xorg + MSS 捕获
```

兼容旧入口：

```python
from capture import DesktopGrabber
from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime
```

推荐新入口：

```python
from capture import CaptureConfig, create_capture_runner
```

## 启动配置

`main.py` 构建 `CaptureConfig`：

```python
capture_config = CaptureConfig(
    output_resolution=OUTPUT_RESOLUTION,
    fps=FPS,
    window_title=WINDOW_TITLE,
    capture_mode=CAPTURE_MODE,
    monitor_index=MONITOR_INDEX,
    capture_tool=CAPTURE_TOOL,
    os_name=OS_NAME,
)
```

关键字段：

| 字段 | 作用 |
|---|---|
| `os_name` | `Windows` / `Darwin` / `Linux...`，决定平台后端 |
| `capture_tool` | `WindowsCaptureCUDA`、`DXCamera`、`ScreenCaptureKit` 等，决定具体实现 |
| `capture_mode` | `Monitor` 或 `Window` |
| `window_title` | Window 模式下定位窗口 |
| `monitor_index` | Monitor 模式下选择显示器 |
| `output_resolution` | capture 侧 resize 目标高度 |
| `fps` | capture session 或 polling sleep/timeout 参考 |

## 后端选择

`capture.factory.create_capture_runner()` 是分发入口：

```text
if Windows 且 capture_tool in:
  - WindowsCapture
  - WindowsCaptureCUDA
  - WindowsCaptureROCm

=> WindowsCaptureEventRunner

else
=> PollingCaptureRunner + create_capture_source(config)
```

`create_capture_source(config)` 再选择具体 `DesktopGrabber`：

```text
Windows:
  DesktopDuplication -> backends/windows_desktop_duplication.py
  DXCamera 或其它默认 -> backends/windows_dxcamera.py

Darwin:
  ScreenCaptureKit -> backends/macos_screencapturekit.py
  其它 -> backends/macos_coregraphics.py

Linux:
  -> backends/linux_mss.py
```

当前有两类捕获模型：

```text
事件式 runner:
  WindowsCapture / CUDA / ROCm

轮询式 runner:
  DXCamera
  DesktopDuplication
  ScreenCaptureKit
  CoreGraphics
  Linux MSS
```

## Runner 回调合同

`main.py` 通过 callback 把应用状态交给 runner：

```python
runner.run(
    shutdown_event=shutdown_event,
    on_frame=_capture_frame_arrived,
    on_error=_capture_error,
    on_closed=_capture_closed,
    is_paused=_openxr_source_paused,
    is_hard_idle=_openxr_hard_idle_active,
    on_paused=_capture_paused,
    on_session_update=_capture_session_update,
    on_tick=_log_source_health,
)
```

| callback | 调用方 | 作用 |
|---|---|---|
| `on_tick` | runner 每轮循环 | 记录 source health |
| `is_hard_idle` | runner | OpenXR hard idle 时暂停 capture |
| `is_paused` | runner | OpenXR source paused 时暂停 capture |
| `on_paused` | runner | 清空 `raw_q`，记录 paused drop |
| `on_frame` | runner 拿到帧后 | 把 raw frame 放进 `raw_q` |
| `on_error` | runner 异常时 | 记录 capture error |
| `on_closed` | capture session 关闭时 | 打日志 |
| `on_session_update` | runner 创建/更新 session/control 时 | 让 `main.py` cleanup 能停止当前 capture |

capture 包不直接依赖 `raw_q`、`_source_stat_inc()` 或 OpenXR 状态变量；这些仍归 `main.py` 管。

## WindowsCapture 事件流

适用后端：

```text
WindowsCapture
WindowsCaptureCUDA
WindowsCaptureROCm
```

流程：

```text
main.py
  -> create_capture_runner(config)
  -> WindowsCaptureEventRunner(config)
  -> runner.run(...)

WindowsCaptureEventRunner.run:
  1. 设置 Windows DPI awareness
  2. 根据 capture_tool import:
       WindowsCaptureROCm -> wc_rocm
       WindowsCaptureCUDA -> wc_cuda
       WindowsCapture -> windows_capture
  3. 启动 Alt+Tab keyboard worker
  4. 循环直到 shutdown:
       - on_tick()
       - hard idle 则 on_paused("hard_idle") + sleep
       - 创建 WindowsCapture session:
           Window 模式 -> WindowsCapture(window_name=...)
           Monitor 模式 -> WindowsCapture(monitor_index=...)
       - on_session_update(session, control=None)
       - 注册 cap.event on_frame_arrived
       - 注册 cap.event closed
       - cap.start()
       - 异常则 on_error()
       - finally 清空 session/control
```

帧到达时：

```text
on_frame_arrived(frame, internal_capture_control):
  1. 保存 control
  2. on_session_update(session, control)
  3. 记录 capture_start_time
  4. 如果 shutdown / hard idle / paused，直接丢帧
  5. frame.frame_buffer.copy() 或 clone()
  6. on_frame(raw, output_resolution, capture_start_time)
```

`main.py` 的 `_capture_frame_arrived()`：

```text
1. capture_frames +1
2. breakdown capture +1
3. raw_q 放入 (frame_raw, size, capture_start_time)
4. raw_put +1
```

## Polling 捕获流

适用于 DXCamera、DesktopDuplication、macOS、Linux。

```text
main.py
  -> create_capture_runner(config)
  -> PollingCaptureRunner(config, source_factory)
  -> runner.run(...)

PollingCaptureRunner.run:
  1. self._source = source_factory()
  2. on_session_update(source, None)
  3. while not shutdown:
       - on_tick()
       - hard idle: on_paused("hard_idle"), sleep 0.1
       - paused: on_paused("paused"), sleep 0.05
       - capture_start_time = now
       - frame_raw, size = source.grab()
       - on_frame(frame_raw, size, capture_start_time)
  4. finally:
       - on_session_update(None, None)
       - on_closed()
```

各 backend 的 `DesktopGrabber.grab()` 统一返回：

```python
(frame_raw, scaled_height)
```

`frame_raw` 通常是 BGR 或 BGRA numpy array。WindowsCapture CUDA/ROCm 路径可能返回 torch/cuda buffer。

## 后端职责

| 后端 | 职责 |
|---|---|
| `windows_dxcamera.py` | 使用 `wincam.DXCamera`；支持 Monitor / Window；Window 模式通过 `win32gui` 获取 client bounds；窗口移动或尺寸变化时重建 camera；失败时尝试返回 last frame |
| `windows_desktop_duplication.py` | 使用 `windows_capture.DxgiDuplicationSession`；支持 monitor/window；窗口跨屏时 `switch_monitor` 或 `recreate`；无新帧时返回 last frame 或黑帧；device/access loss 时重建 |
| `windows_capture_event.py` | 使用 `windows_capture` / `wc_cuda` / `wc_rocm`；事件式 `on_frame_arrived`；保留 Alt+Tab worker 和 DPI awareness 行为 |
| `macos_screencapturekit.py` | 使用 ScreenCaptureKit；支持 Display / Window filter；`_SCKFrameReceiver` 接收 sample buffer；输出 BGRA 或 BGR |
| `macos_coregraphics.py` | 使用 CoreGraphics 截图；支持 Window / Monitor；实现 cursor 获取、resize cache、cursor overlay；可返回 BGRA 或 BGR |
| `linux_mss.py` | Linux Xorg 下用 Xlib 查窗口；用 MSS 抓 monitor/window 区域；返回 raw frame 和 scaled height |

## Raw Frame 到 Depth Runtime

`capture_loop` 只负责把 raw frame 放进 `raw_q`：

```text
capture_loop
  -> raw_q.put((frame_raw, size, capture_start_time))
```

`process_depth_loop` 负责 RGB 转换和 depth 推理：

```text
process_depth_loop
  -> raw_q.get()
  -> capture_frame_to_rgb(frame_raw, size, device=DEVICE, use_torch=USE_CUDART, output="tensor")
  -> prepare_rgb_for_depth_runtime(frame_rgb, device=DEVICE)
  -> depth_runtime.predict_depth_frame(runtime_rgb)
  -> depth_q.put((frame_rgb, depth, capture_start_time))
```

`capture_frame_to_rgb()`：

```text
raw BGR/BGRA -> RGB
resize 到 target_height
保持偶数宽高
```

torch 路径：

```text
frame_raw[..., [2,1,0]]
HWC -> CHW
F.interpolate bilinear
align_corners=False
downscale 时 antialias=True
```

numpy 路径：

```text
cv2.COLOR_BGRA2RGB 或 cv2.COLOR_BGR2RGB
resize:
  downscale -> INTER_AREA
  upscale -> INTER_CUBIC
```

`prepare_rgb_for_depth_runtime()`：

```text
numpy HWC -> torch CHW
接受 CHW 或 BCHW
to(device=device, dtype=torch.float32)
如果 max > 1.5，则 /255.0
clamp 到 0..1
```

进入 `DepthRuntime.predict_depth_frame()` 的合同：

```text
RGB
torch.Tensor
CHW 或 BCHW
float32
0..1
在目标 device 上
```

## 队列合同

当前 `main.py` 保持两级队列：

```text
raw_q:
  (frame_raw, size, capture_start_time)

depth_q:
  (frame_rgb, depth, capture_start_time)
```

`raw_q` 是 capture 输出；`depth_q` 是 viewer / xrviewer 消费的稳定合同。

## 清理流程

`main.py` 保留：

```python
capture_control = None
capture_session = None
```

runner 通过 `on_session_update(session, control)` 更新它们。

cleanup 时 `_stop_active_capture_session()`：

```text
优先:
  capture_control.stop()

否则:
  capture_session.stop()
```

这保证 WindowsCapture event control 和 polling source 都能被统一停止。

## 边界

capture 层负责：

```text
OS / capture_tool 后端选择
Monitor / Window 捕获
捕获 session 生命周期
pause / hard idle 配合
raw frame 输出
capture 侧 RGB 转换和 resize
把 RGB 转成 depth runtime 输入 tensor
```

capture 层不负责：

```text
depth 模型选择
artifact 准备
ONNX/TensorRT backend 细节
depth inference
stereo synthesis
OpenXR session/swapchain
GUI settings 持久化
```

## 一句话流程

```text
settings/utils 决定 OS + Capture Tool
  -> CaptureConfig
  -> factory 选择 event runner 或 polling runner
  -> backend 抓 raw frame
  -> main.py 放入 raw_q
  -> preprocess 转 RGB + resize + float32 tensor
  -> DepthRuntime 推 depth
  -> depth_q 输出给 viewer/xrviewer
```
