# Desktop2Stereo 工程设计规范重构 — Codex 提示词

## 任务概述

根据 `docs/26-desktop2stereo-engineering-design-specification.md` 工程设计规范，对 4k-stereo-synthesis-lab 项目进行重构。目标是将当前散落的参数和实现收敛到统一的工程契约。

## 重构优先级

按规范后续实施优先级排序，分为 8 项任务，建议分阶段实施。

---

## 任务 1：新增 `RuntimeSettingsSnapshot` dataclass + `settings_update_q`

### 规范依据

> GUI hot save -> settings.yaml (legacy) → 新路径应收敛到 RuntimeSettingsSnapshot + settings_update_q

### 当前状态

`src/stereo_runtime/hot_reload.py` 中的 `StereoHotReloader` 直接读取 `settings.yaml`，用 `hot_reload_value_snapshot()` 提取 YAML 字段，然后调用 `replace(runtime.config, **values)` 修改 `StereoRuntimeConfig`。这是通过文件修改时间（mtime）轮询的机制，不是正规的队列驱动热更新。

### 实现要求

1. **新建 `src/stereo_runtime/settings_snapshot.py`**，定义：

```python
@dataclass(frozen=True)
class RuntimeSettingsSnapshot:
    version: int
    timestamp: float
    # Stereo synthesis params (normalized)
    stereo_quality: StereoQuality | None = None
    output_format: OutputFormat | None = None
    depth_strength: float | None = None
    convergence: float | None = None
    ipd_mm: float | None = None
    stereo_scale: float | None = None
    max_shift_ratio: float | None = None
    temporal: bool | None = None
    temporal_strength: float | None = None
    foreground_scale: float | None = None
    depth_antialias_strength: float | None = None
    edge_dilation: int | None = None
    edge_threshold: float | None = None
    mask_feather_radius: int | None = None
    hole_fill_mode: str | None = None
    hole_fill_radius: int | None = None
    hole_fill_strength: float | None = None
    screen_edge_mask_suppression: int | None = None
    cross_eyed: bool | None = None
    anaglyph_method: str | None = None
    fused: bool | None = None
    # Depth/device params (pipeline rebuild required)
    depth_backend: str | None = None
    model_id: str | None = None
    export_height: int | None = None
    export_width: int | None = None
    # Session restart required
    device: str | None = None

    def classify(self) -> SnapshotChangeClass:
        """Classify which params changed (hot_reload / pipeline_rebuild / session_restart)."""
        ...
```

2. **新建 `settings_update_q`**（`queue.Queue` maxsize=1），和 `raw_q`/`runtime_q` 同级。

3. **修改 `RuntimePipelineLoop.run()`**：在帧循环中先检查 `settings_update_q`，按变更分级处理：
   - `hot_reload`：调用 `runtime.configure_stereo()`
   - `pipeline_rebuild`：重建 depth provider 后重新配置
   - `session_restart`：抛出信号由外层处理

4. **保持向后兼容**：`StereoHotReloader` 继续工作，但新的热更新应通过 `settings_update_q.put(snapshot)` 发送。

5. **`StereoRuntimeResult.debug_info`** 增加 `active_settings_version: int` 追踪当前活跃的 snapshot 版本号。

### 涉及文件

- **新建** `src/stereo_runtime/settings_snapshot.py`
- **修改** `src/stereo_runtime/hot_reload.py` — 可选兼容
- **修改** `src/stereo_runtime/pipeline.py` — `RuntimePipelineLoop.run()` 添加 `settings_update_q` 检查
- **修改** `src/stereo_runtime/runtime.py` — `StereoRuntimeResult` 添加 `active_settings_version`
- **修改** `src/stereo_runtime/adapter.py` — 如果需要 snapshot 到 config 的转换
- **修改** `src/app_runtime/runtime_context.py` — 创建 `settings_update_q`
- **修改** `src/app_runtime/runtime_callbacks.py` — 添加 snapshot send 方法
- **修改** `src/gui/config_mgr.py` — 写 settings.yaml 的同时发送 snapshot
- **修改** `tests/test_hot_reload.py` — 添加 snapshot 测试

---

## 任务 2：新增 `resolve_parallax_budget()` 视差解析器

### 规范依据

> normalized-depth 路径使用 `max_disparity_px` + `depth_response`，替代 legacy `IPD/stereo_scale/depth_strength/max_shift_ratio`

### 当前状态

`StereoConfig`（`src/stereo_runtime/synthesis.py:30`）使用 legacy 参数：
```python
depth_strength: float = 2.0
ipd: float = 0.064
stereo_scale: float = 0.4
max_shift_ratio: float = 0.05
```

`baseline_shift.py` 中的 `compute_shift_px()` 使用这些参数计算最终 shift_px。

### 实现要求

1. **新建 `src/stereo_runtime/parallax.py`**：

```python
@dataclass(frozen=True)
class ParallaxBudget:
    max_disparity_px: float
    depth_response: DepthResponseFn  # how depth 0..1 maps to disparity

def resolve_parallax_budget(
    render_width: int,
    render_height: int,
    preset: str,
    depth_strength: float,
    stereo_scale: float,
    convergence: float,
    ipd_mm: float,
    max_shift_ratio: float,
) -> ParallaxBudget:
    """Convert legacy + preset params to normalized parallax budget.
    
    Rule:
    1. Start from preset baseline (e.g., cinema=2%, game=1.5%, image=3% of width)
    2. Apply depth_strength as multiplier
    3. Apply stereo_scale as final multiplier
    4. Clamp by max_shift_ratio * render_width
    5. Return max_disparity_px + depth_response mapping
    """
```

2. **`compute_shift_px()`升级**：保留旧签名（legacy 参数），内部调用 `resolve_parallax_budget()` 并跟踪 debug_info 中的实际使用参数。

3. **`StereoConfig` 增加新字段**（可选默认 None）：
   ```python
   max_disparity_px: float | None = None  # override legacy formula when set
   ```

4. **Debug info 增加**：
   ```python
   debug["resolved_max_disparity_px"] = ...
   debug["parallax_budget_preset"] = ...
   debug["parallax_resolver_version"] = 1
   ```

5. **presets 更新**：`src/stereo_runtime/presets.py` 中的 preset 应导出预设的 max_disparity_px baseline。

### 涉及文件

- **新建** `src/stereo_runtime/parallax.py`
- **修改** `src/stereo_runtime/baseline_shift.py` — `compute_shift_px()` 调用 resolve
- **修改** `src/stereo_runtime/synthesis.py` — `StereoConfig` 增加新字段
- **修改** `src/stereo_runtime/presets.py` — 导出 parallax baseline
- **修改** `src/stereo_runtime/openxr_render.py` — `_shift_params()` 通过 parallax budget
- **修改** `tests/test_presets.py` — 添加 parallax budget 测试
- **新建** `tests/test_parallax.py`

---

## 任务 3：新增 `CaptureFrame` metadata contract

### 规范依据

> Capture metadata 从 `tuple(frame_raw, size, timestamp)` 升级为 `CapturedFrame` 包含 device/dtype/format/copy_mode/source metadata

### 当前状态

当前 `CapturedFrame`（`src/capture/types.py:21`）：
```python
@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    target_height: OutputResolution
    timestamp: float
```

没有 device、dtype、format、copy_mode 等信息。`runners.py` 中的 `on_frame` callback 也只传 `(frame_raw, size, capture_start_time)`。

### 实现要求

1. **升级 `src/capture/types.py` 中的 `CapturedFrame`**：

```python
from enum import Enum, auto

class FrameCopyMode(Enum):
    NONE = auto()       # zero-copy, aliased buffer
    CLONE = auto()      # GPU-to-GPU clone
    COPY = auto()       # CPU-to-CPU copy
    CPU_NUMPY = auto()  # converted to CPU numpy
    GPU_TENSOR = auto() # converted to GPU tensor

@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    target_height: OutputResolution
    timestamp: float
    # New metadata fields:
    capture_tool: str = ""
    capture_mode: str = ""
    monitor_index: int = 0
    window_title: str = ""
    capture_size: tuple[int, int] | None = None
    frame_raw_type: str = ""          # "numpy", "torch.Tensor", "cuda", etc.
    frame_raw_device: str = ""        # "cpu", "cuda:0", "rocm:0", etc.
    frame_raw_dtype: str = ""         # "uint8", "float32", etc.
    copy_mode: FrameCopyMode = FrameCopyMode.COPY
    original_format: str = ""         # "RGB", "BGR", "BGRA", "NV12", etc.
    metadata: dict[str, Any] = field(default_factory=dict)
```

2. **修改 `src/capture/runners.py`**：各 runner 的 `on_frame` callback 参数改为 `CapturedFrame`，填充 metadata。

3. **修改 pipeline `RuntimePipelineLoop.run()`**：从 `raw_q` 取出的不再是 `(frame_raw, size, timestamp)` 三元组，而是 `CapturedFrame` 对象。

4. **修改所有消费 raw_q 的地方**（`pipeline.py` 第 89 行拆包 `frame_raw, size, capture_start_time`）。

5. **Windows 后端升级**：`windows_capture_event.py` 中 copy/clone 操作填充 `copy_mode` 字段。

### 涉及文件

- **修改** `src/capture/types.py` — `CapturedFrame` 升级
- **修改** `src/capture/runners.py` — `on_frame` 使用 `CapturedFrame`
- **修改** `src/capture/session.py` — `CaptureSessionLoop` 传递 metadata
- **修改** `src/capture/backends/windows_capture_event.py` — 填充 copy_mode
- **修改** `src/capture/backends/windows_desktop_duplication.py` — 填充 metadata
- **修改** `src/capture/backends/windows_dxcamera.py` — 填充 metadata
- **修改** `src/stereo_runtime/pipeline.py` — 拆包 `CapturedFrame`
- **修改** `src/stereo_runtime/runtime.py` — 处理 `CapturedFrame`
- **修改** `src/app_runtime/runtime_context.py` — 构建时适配
- **修改** `tests/test_capture_session.py` — 更新测试
- **新建** `tests/test_capture_metadata.py`

---

## 任务 4：Runtime preprocess 显式处理多 device tensor 路径

### 规范依据

> 输入 frame_raw 可以是 numpy、CPU torch tensor、CUDA torch tensor、ROCm torch tensor。输出 render_rgb 必须 B/C/H/W RGB tensor，device 与 depth provider 对齐。

### 当前状态

`capture/preprocess.py` 中的 `capture_frame_to_rgb()` 和 `prepare_rgb_for_stereo_runtime()` 没有显式的 device dispatch。当前 pipeline 靠 `device=ctx.device` 参数传给 `capture_frame_to_rgb()`，但 device 转换逻辑散落在多个位置。

### 实现要求

1. **修改 `src/capture/preprocess.py`**，增加 device dispatch：

```python
def capture_frame_to_rgb(
    frame,
    size,
    *,
    device="cuda",
    use_torch=False,
    output="tensor",
):
    # Step 1: Determine input device/type
    # Step 2: Color conversion (BGR/BGRA/NV12 -> RGB)
    # Step 3: Resize if needed
    # Step 4: Device transfer (cpu -> cuda, cuda -> rocm, etc.)
    # Step 5: Tensor shape normalization -> BCHW
    # Step 6: Mark _d2s_preprocess_backend
```

2. **所有 device transfer 显式记录**：在 `_d2s_preprocess_backend` 或在 debug_info 中输出 `preprocess_device_transfer: cpu->cuda`。

3. **明确标注当前实现是否真正零拷贝**（规范第 5 条）。

### 涉及文件

- **修改** `src/capture/preprocess.py`
- **修改** `src/capture/preprocess_triton.py`
- **修改** `src/stereo_runtime/runtime.py`
- **修改** `tests/test_capture_preprocess.py`

---

## 任务 5：OpenXR legacy uniforms 移入 adapter

### 规范依据

> OpenXR direct uniforms 从 legacy ipd/depth_ratio/stereo_scale/max_shift_ratio 改为 adapter 从规范参数转换

### 当前状态

`OpenXRStateController`（`src/stereo_runtime/openxr_state.py:8`）和 `OpenXRRenderConfig`（`src/stereo_runtime/openxr_render.py:17`）直接存储 legacy 参数。`current_render_config()` 方法从 `runtime_config_state` dict 构造 `OpenXRRenderConfig`，两端都是 legacy 字段。

`RuntimeCallbacks.update_openxr_runtime_config()`（`src/app_runtime/runtime_callbacks.py:81`）直接传递 legacy 字段给 `OpenXRStateController`。

`StereoHotReloader.apply_if_needed()` 第 193 行直接传 legacy 字段：
```python
on_openxr_config_update(
    ipd=values["ipd"],
    depth_strength=values["depth_strength"],
    convergence=values["convergence"],
    stereo_scale=values["stereo_scale"],
    max_shift_ratio=values["max_shift_ratio"],
)
```

### 实现要求

1. **在 `src/stereo_runtime/adapter.py` 中新增适配函数**：

```python
def openxr_render_config_from_snapshot(
    snapshot: RuntimeSettingsSnapshot,
    render_size: tuple[int, int],
    preset: str,
) -> OpenXRRenderConfig:
    """Convert normalized settings to OpenXRRenderConfig with legacy uniforms."""
    budget = resolve_parallax_budget(
        render_width=render_size[0],
        render_height=render_size[1],
        preset=preset,
        depth_strength=snapshot.depth_strength or 2.0,
        stereo_scale=snapshot.stereo_scale or 0.4,
        convergence=snapshot.convergence or 0.0,
        ipd_mm=snapshot.ipd_mm or 32.0,
        max_shift_ratio=snapshot.max_shift_ratio or 0.05,
    )
    return OpenXRRenderConfig(
        depth_strength=snapshot.depth_strength or 2.0,
        convergence=snapshot.convergence or 0.0,
        ipd=(snapshot.ipd_mm or 32.0) / 1000.0,
        max_shift_ratio=snapshot.max_shift_ratio or 0.05,
        ipd_mm=snapshot.ipd_mm or 32.0,
        stereo_scale=snapshot.stereo_scale or 0.4,
        screen_roll=0.0,
        # Add normalized fields:
        # resolved_max_disparity_px=budget.max_disparity_px,
    )
```

2. **修改 `OpenXRStateController`**：内部存储改为 `RuntimeSettingsSnapshot` 风格，不再直接暴露 legacy 字段。`current_render_config()` 通过 adapter 转换。

3. **修改 `RuntimeCallbacks`**：`update_openxr_runtime_config()` 接收 `RuntimeSettingsSnapshot` 而非零散 legacy 参数。

4. **OpenXR debug_info 增加规范字段**：不再只记录 `openxr_ipd/openxr_depth_strength/openxr_stereo_scale/openxr_max_shift_ratio`，改为记录 `resolved_max_disparity_px`。

### 涉及文件

- **修改** `src/stereo_runtime/adapter.py` — 新增 `openxr_render_config_from_snapshot()`
- **修改** `src/stereo_runtime/openxr_state.py` — `OpenXRStateController` 内部存储升级
- **修改** `src/stereo_runtime/openxr_render.py` — `OpenXRRenderConfig` 可选增加新字段
- **修改** `src/stereo_runtime/runtime.py` — `process_openxr_frame()` debug_info 更新
- **修改** `src/app_runtime/runtime_callbacks.py` — 接口升级
- **修改** `src/stereo_runtime/hot_reload.py` — 适配新接口
- **修改** `tests/test_openxr_state.py`
- **修改** `tests/test_runtime_openxr.py`

---

## 任务 6：RenderSizePolicy 独立为 runtime policy

### 规范依据

> render_size policy 还未完整独立成 runtime policy，需要 native/scaled/fixed/dynamic 统一解析

### 当前状态

render_size 逻辑分散在多个位置：
- `StereoRuntimeConfig.export_height/export_width` 只是模型输入尺寸
- `capture/preprocess.py` 中的 resize 逻辑
- OpenXR viewer 中 `frame_size_from_runtime_result()` 自行解析
- 各地通过 `_runtime_frame_size()` 辅助函数获取

### 实现要求

1. **新建 `src/stereo_runtime/render_size.py`**：

```python
from enum import Enum
from dataclasses import dataclass

class RenderSizePolicy(Enum):
    NATIVE = "native"      # use capture full resolution
    SCALED = "scaled"      # scale by factor
    FIXED = "fixed"        # fixed resolution
    DYNAMIC = "dynamic"    # adaptive

@dataclass(frozen=True)
class RenderSizeConfig:
    policy: RenderSizePolicy = RenderSizePolicy.NATIVE
    scale_factor: float = 1.0
    fixed_width: int = 1920
    fixed_height: int = 1080
    max_pixels: int = 3840 * 2160
    min_dimension: int = 480
    align: int = 16  # dimension alignment

def resolve_render_size(
    capture_size: tuple[int, int],
    config: RenderSizeConfig,
) -> tuple[int, int]:
    """Resolve the actual render size based on policy and capture size.
    
    NATIVE: (capture_width, capture_height) aligned to align
    SCALED: (capture_w * factor, capture_h * factor) aligned
    FIXED: (fixed_width, fixed_height)
    DYNAMIC: pick best within max_pixels and min_dimension
    """
```

2. **修改 `RuntimePipelineContext`**：携带 `RenderSizeConfig`，在 pipeline 中解析。

3. **各消费端统一使用** `debug_info["runtime_output_eye_size"]` 和 `debug_info["runtime_output_display_size"]`。

### 涉及文件

- **新建** `src/stereo_runtime/render_size.py`
- **修改** `src/stereo_runtime/runtime.py` — 消费 `RenderSizeConfig`
- **修改** `src/stereo_runtime/pipeline.py` — `RuntimePipelineContext` 增加 `RenderSizeConfig`
- **修改** `src/xr_viewer/openxr_runtime.py` — 使用 `runtime_output_display_size`
- **修改** `src/viewer/viewer_runtime.py` — 使用统一 size 解析

---

## 任务 7：CUDA/ROCm capture 零拷贝路径标注

### 规范依据

> WindowsCaptureCUDA / WindowsCaptureROCm 是零拷贝候选，但必须显式标注当前实现是否真正零拷贝，对 GPU tensor 定义 contract 避免 CPU numpy 中转

### 当前状态

`windows_capture_event.py` 中，事件 runner 对 `frame.frame_buffer` 调 `copy()` 或 `clone()`。当前无法确定是否零拷贝，因为：
- 没有 `copy_mode` 标注
- 没有 device 追踪
- 没有 tensor 类型声明

### 实现要求

1. **在 `windows_capture_event.py` 各 runner 中增加 copy_mode 标注**：

```python
# WindowsCaptureCUDA:
frame_buffer = frame.frame_buffer.clone()  # GPU-to-GPU
copy_mode = FrameCopyMode.CLONE
frame_raw_device = "cuda"

# WindowsCaptureROCm:
frame_buffer = frame.frame_buffer.clone()
copy_mode = FrameCopyMode.CLONE  
frame_raw_device = "rocm"

# WindowsCapture (CPU):
frame_raw = frame.frame_buffer.copy()
copy_mode = FrameCopyMode.COPY
frame_raw_device = "cpu"
```

2. **在 preprocess 中根据 `frame_raw_device` 选择路径**：
   - `cuda` → 直接 CUDA tensor，避免 CPU 转换
   - `rocm` → 直接 ROCm tensor
   - `cpu` → 正常 numpy -> tensor -> device

3. **debug_info 输出** `capture_copy_mode` 和 `preprocess_device_origin`。

### 涉及文件

- **修改** `src/capture/backends/windows_capture_event.py`
- **修改** `src/capture/preprocess.py`
- **修改** `tests/test_windows_capture_event.py`

---

## 任务 8：network_stream encoder profile 与 packed frame contract

### 规范依据

> network_stream 从 MJPEG legacy（消费 sbs numpy）升级为统一 packed_synthesis + encoder transport contract

### 当前状态

MJPEG streamer 直接消费 `runtime_result.sbs` 转 numpy，通过 `runtime_output_to_numpy()` 处理，JPEG quality 固定，没有 encoder profile 概念。

### 实现要求

1. **新建 `src/streaming/encoder_profile.py`**：

```python
@dataclass(frozen=True)
class EncoderProfile:
    codec: str = "mjpeg"        # mjpeg, h264, h265
    quality: int = 85           # 0-100
    target_fps: int = 30
    target_bitrate: int | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    pixel_format: str = "rgb"   # rgb, nv12, bgra
```

2. **修改 `src/streaming/mjpeg_streamer.py`**：接受 `EncoderProfile`，根据 profile 调整 quality/resize。

3. **packed frame contract 文档**：在代码注释中明确 network stream 使用 `packed_sbs` 作为默认输入格式，转换发生在 transport 层。

### 涉及文件

- **新建** `src/streaming/encoder_profile.py`
- **修改** `src/streaming/mjpeg_streamer.py` — 接受 EncoderProfile
- **修改** `src/streaming/legacy_runtime.py` — 适配
- **修改** `src/gui/config_mgr.py` — 推流配置映射到 EncoderProfile
- **新建** `tests/test_encoder_profile.py`

---

## 通用注意事项

### Codex 风格

- 每项任务独立提交，不要混在一个 commit 中
- 每个 commit 信息格式：`refactor: 任务名称 — 简短说明`
- 所有新 dataclass 使用 `@dataclass(frozen=True)`
- 所有 debug_info 字段名使用 snake_case
- 向后兼容优先：legacy 字段继续读取，但标注 deprecated

### 测试要求

每项任务必须附带：

1. **单元测试**：新 dataclass/snapshot/resolver 的基础功能测试
2. **集成测试**：新组件在 pipeline 中的行为测试
3. **向后兼容测试**：旧配置文件和旧参数仍能工作
4. **debug_info 验证测试**：确认新字段被正确写入

### 文件读写规则

- 使用 file_system MCP 读取/写入文件
- 使用 codegraph MCP 理解代码关系和依赖

### 当前项目结构参考

```
src/
├── app_runtime/       # 应用运行时上下文
├── capture/           # 图像捕捉子系统
│   └── backends/      # 各平台捕捉后端
├── gui/               # Flet GUI
├── main.py            # 主入口
├── stereo_runtime/    # 立体合成运行时
│   ├── adapter.py     # 配置适配
│   ├── baseline_shift.py
│   ├── depth_provider.py
│   ├── hot_reload.py  # 热更新 (将被替换)
│   ├── model_artifacts.py
│   ├── model_registry.py
│   ├── occlusion.py
│   ├── hole_fill.py
│   ├── openxr_render.py
│   ├── openxr_state.py
│   ├── output.py
│   ├── pipeline.py    # 主流水线循环
│   ├── presets.py
│   ├── runtime.py     # StereoRuntime 核心
│   └── synthesis.py   # 立体合成
├── streaming/         # 网络推流
├── viewer/            # 本地 GLFW 查看器
└── xr_viewer/         # OpenXR 查看器
tests/
└── *.py               # 测试文件
```
