# Cinema IPD64 生产级参数扫描指南

## 测试目的

本文档记录如何运行电影模式立体参数的生产级视觉回归扫描脚本。

该脚本会使用真实样本图，走生产深度路径和立体合成路径，然后按照伪影风险和立体分离度对参数组合排序。它主要用于确认：在显示 IPD 固定为 64mm 时，不同深度强度、立体缩放和最大位移比例是否还能生成可接受的 half-SBS 输出。

## 脚本路径

```text
scripts/tools/cinema_ipd64_quality_sweep.py
```

默认真实样本输入：

```text
samples/4K.jpg
```

默认生产深度模型文件：

```text
src/models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.onnx
src/models/models--lc700x--Distill-Any-Depth-Base-hf/model_fp16_294x518.trt
```

## 测试覆盖范围

脚本覆盖会影响电影模式输出观感的生产级立体参数：

- 显示 IPD 映射：`display_ipd_mm -> runtime_ipd_mm = display_ipd_mm * 30 / 60`
- 深度选项：`soft=2.0`、`standard=2.5`、`enhanced=3.0`
- 立体缩放：通常测试 `0.3,0.4,0.5`
- 最大位移比例：通常测试 `0.03,0.04,0.05`
- convergence
- 前景比例
- 深度抗锯齿强度
- 边缘扩张
- 边缘阈值
- 遮罩羽化半径
- 补洞模式、半径和强度
- 时序强度
- 自动场景重置
- 场景重置阈值
- 重置冷却帧数

默认情况下，脚本会基于真实样本图生成一个短平移序列。因此时序平滑和场景重置逻辑会真正进入执行路径，而不是只作为静态配置字段被记录。

## 默认生产行为

默认模式使用生产 TensorRT 深度 provider：

```text
depth_source=production
backend=tensorrt_native
engine=model_fp16_294x518.trt
onnx=model_fp16_294x518.onnx
```

默认立体路径：

```text
stereo_backend=quality_4k
output_format=half_sbs
layers=2
temporal=true
auto_reset_temporal=true
sequence_frames=3
sequence_shift_px=2
```

## 推荐的重点扫描命令

在仓库根目录运行：

```powershell
src\python3\python.exe scripts\tools\cinema_ipd64_quality_sweep.py `
  --rgb samples\4K.jpg `
  --out outputs\visual_regression\cinema_ipd64_production_focused_sweep `
  --max-width 1280 `
  --ipd-mm 64 `
  --stereo-scales 0.3,0.4,0.5 `
  --max-shift-ratios 0.03,0.04,0.05 `
  --depth-options soft,standard,enhanced `
  --device cuda `
  --depth-source production
```

这是当前 IPD 校准工作中主要使用的生产级扫描命令。

## 快速 Smoke 测试

修改脚本或立体 runtime 后，先用这个命令做快速验证：

```powershell
src\python3\python.exe scripts\tools\cinema_ipd64_quality_sweep.py `
  --rgb samples\4K.jpg `
  --out outputs\visual_regression\cinema_ipd64_production_temporal_smoke `
  --max-width 640 `
  --stereo-scales 0.3 `
  --max-shift-ratios 0.03 `
  --depth-options standard `
  --device cuda `
  --depth-source production `
  --sequence-frames 3 `
  --sequence-shift-px 2
```

有效的 smoke 报告应包含：

- `depth_source: production_tensorrt_native`
- `depth_provider.depth_backend: tensorrt_native`
- `depth_provider.engine_path` 以 `model_fp16_294x518.trt` 结尾
- `cinema_base.temporal: true`
- `cinema_base.auto_reset_temporal: true`
- 每一行结果里包含 `scene_delta`、`temporal_reset`、`temporal_reset_count`

## 重要参数

### 深度来源

```powershell
--depth-source production
```

使用生产 TensorRT engine。这是默认值，真实质量结论应使用这个模式。

```powershell
--depth-source proxy
```

使用由样本 RGB 亮度生成的代理深度。它只适合在 CUDA 或 TensorRT 不可用时调试脚本流程，不应作为最终质量判断依据。

### 生产模型路径

默认路径已内置，也可以显式覆盖：

```powershell
--depth-onnx src\models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.onnx `
--depth-engine src\models\models--lc700x--Distill-Any-Depth-Base-hf\model_fp16_294x518.trt
```

### 时序序列

```powershell
--sequence-frames 3
--sequence-shift-px 2
```

脚本会从真实样本图生成一个短平移序列。每一帧都会走生产深度。每个参数组合会在整个序列中复用同一个 `TemporalState`，因此时序平滑、场景阈值和冷却行为都会被测试到。

如果只想做单帧静态对比：

```powershell
--no-temporal --sequence-frames 1
```

### 电影模式 Runtime 参数

脚本直接暴露以下生产参数：

```powershell
--convergence 0.25
--foreground-scale 0.5
--depth-antialias-strength 1.0
--edge-dilation 2
--edge-threshold 0.04
--mask-feather-radius 3
--hole-fill-mode soft_low_ghost
--hole-fill-radius 3
--hole-fill-strength 1.0
--temporal-strength 0.85
--scene-reset-threshold 0.22
--reset-cooldown-frames 3
```

## 输出文件

输出目录包含：

```text
cinema_ipd64_quality_sweep_report.json
source_rgb.png
sequence_last_rgb.png
production_depth.png
depth_edge_mask.png
low_artifact_contact_sheet.png
top_realism_contact_sheet.png
*_half_sbs.png
*_occlusion_mask.png
```

主要报告字段：

- `depth_source`：确认使用生产深度还是代理深度
- `depth_provider`：provider、backend、ONNX 路径、TRT engine 路径
- `depth_timing_sequence`：序列中每一帧的深度耗时
- `sequence`：帧数和生成运动参数
- `cinema_base`：本次运行使用的完整生产立体配置
- `ipd_mapping`：显示 IPD 到 runtime IPD 的映射
- `ranked_low_artifact`：按伪影风险排序的最稳组合
- `ranked_realism`：按立体感/真实感排序的强立体组合
- `by_depth_option`：每个深度选项下的最佳组合
- `rows`：所有测试组合的完整指标

每一行结果包含：

- `max_shift_px`、`p95_shift_px`
- `occlusion_mask_ratio`、`occlusion_edge_overlap`
- `edge_source_mae`、`edge_left_right_mae`
- `edge_gradient_delta`
- `ghost_risk_score`、`hole_risk_score`、`stereo_score`、`realism_score`
- `hole_fill_backend`、`warp_composite_backend`
- `temporal_enabled`、`temporal_strength`
- `scene_delta`、`temporal_reset`、`temporal_reset_count`

## 如何阅读结果

优先查看 `low_artifact_contact_sheet.png`，用于检查保守参数组合。这些组合最适合观察是否存在明显虚影、边缘撕裂或补洞伪影。

再查看 `top_realism_contact_sheet.png`，用于检查立体感最强的组合。这些组合通常会提高分离度，但也会增加边缘压力和虚影风险。

对于当前 IPD 64mm 校准，重点扫描应方便比较：

```text
stereo_scale: 0.3, 0.4, 0.5
max_shift_ratio: 0.03, 0.04, 0.05
depth option: soft, standard, enhanced
```

根据之前的生产深度测试结果，当前预期是：`standard + stereo_scale 0.3 + max_shift_ratio 0.03` 是最稳的平衡默认值；更强的组合会提升 3D 分离感，同时增加边缘伪影风险。

## 信任测试结果前的检查项

在使用报告做参数决策前，确认：

1. 输入图片来自 `samples/`，当前基线优先使用 `samples/4K.jpg`。
2. `depth_source` 是 `production_tensorrt_native`。
3. `depth_provider.engine_path` 指向 `model_fp16_294x518.trt`。
4. 做生产级时序测试时，`cinema_base.temporal` 是 `true`。
5. 测试时序行为时，`sequence.frames` 大于 `1`。
6. 需要肉眼检查 contact sheet，不能只看数值排序。

## 局限性

当前序列是由一张真实图片合成出的平移运动序列。它可以覆盖 temporal 代码路径、scene delta 和 reset cooldown 行为，但不能完全替代真实视频帧序列测试。

数值分数只适合作为相对排序参考。最终参数决策应结合 half-SBS 输出和 contact sheet 的肉眼检查。 
