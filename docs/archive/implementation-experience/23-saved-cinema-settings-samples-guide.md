# 保存参数样本集视觉评估指南

## 测试目的

本文档记录如何运行当前保存参数的样本集视觉评估脚本。

该脚本不会扫描多组参数，而是读取 `src/settings.yaml` 中当前保存的电影模式立体参数，然后对 `samples/` 下所有图片逐张生成 half-SBS 输出、遮挡 mask、深度图和质量指标。它主要用于回答：当前 GUI/配置保存下来的参数组合，在一组真实样本图片上是否足够稳，若不稳应优先调哪个参数。

## 脚本路径

```text
scripts/tools/evaluate_saved_cinema_settings_samples.py
```

默认配置输入：

```text
src/settings.yaml
```

默认样本目录：

```text
samples/
```

默认输出目录：

```text
outputs/visual_regression/saved_cinema_settings_samples
```

## 和参数扫描脚本的区别

`cinema_ipd64_quality_sweep.py` 用于扫描多组参数组合，找出更稳或立体感更强的候选值。

`evaluate_saved_cinema_settings_samples.py` 用于验证当前保存参数，只跑 `settings.yaml` 中的一组参数，并覆盖 `samples/` 下多张样本图。它更适合在 GUI 调参后做回归检查，确认当前组合是否能作为默认值或阶段性候选。

## 读取的保存参数

脚本会从 `settings.yaml` 中读取并映射以下字段：

- `Depth Strength`
- `IPD`
- `Convergence`
- `Stereo Scale`
- `Max Shift Ratio`
- `Foreground Scale`
- `Depth Antialias Strength`
- `Edge Dilation`
- `Edge Threshold`
- `Mask Feather Radius`
- `Hole Fill Mode`
- `Hole Fill Radius`
- `Hole Fill Strength`
- `Temporal`
- `Temporal Strength`
- `Auto Scene Reset`
- `Scene Reset Threshold`
- `Reset Cooldown Frames`
- `Stereo Quality` / `Synthetic View`
- `TensorRT`

`IPD` 如果小于等于 `1.0`，会按 runtime meters 解释，例如 `0.032` 表示 `32mm` runtime IPD。若保存的是大于 `1.0` 的值，则按毫米解释。

## 推荐生产命令

在仓库根目录运行：

```powershell
src\python3\python.exe scripts\tools\evaluate_saved_cinema_settings_samples.py `
  --depth-source production `
  --out outputs\visual_regression\saved_cinema_settings_samples
```

生产级结论应使用 `--depth-source production`，这会走 TensorRT 深度路径。默认不限制输入宽度，会按每张样本图的原始分辨率评估；`samples/4K.jpg` 就按 4K 输入进入流程。

## 快速 Smoke 测试

只验证脚本流程时，可以使用代理深度和较小宽度：

```powershell
src\python3\python.exe scripts\tools\evaluate_saved_cinema_settings_samples.py `
  --depth-source proxy `
  --max-width 320 `
  --out outputs\visual_regression\saved_cinema_settings_samples_proxy_smoke
```

proxy 模式只用于确认脚本能读取配置、遍历样本、生成报告。不要用 proxy 报告做最终参数质量判断。

## 常用参数

### 指定配置文件

```powershell
--settings src\settings.yaml
```

### 指定样本目录

```powershell
--samples samples
```

默认只处理当前目录及非 `private` 子目录中的图片。若需要包含 `samples/private/`：

```powershell
--include-private
```

### 控制输入尺寸

默认值：

```powershell
--max-width 0
```

`0` 表示不降采样，按图片原始分辨率评估。生产评估应保持默认值；例如 4K 样本就按 4K 处理。

只有做快速回归或流程 smoke 时才建议显式限制宽度：

```powershell
--max-width 320
```

### 时序序列

```powershell
--sequence-frames 3
--sequence-shift-px 2
```

脚本会从每张样本图生成短平移序列，使 temporal、scene reset、reset cooldown 进入执行路径。

## 输出文件

输出目录包含：

```text
saved_cinema_settings_samples_report.json
highest_risk_contact_sheet.png
<sample_name>/<sample_name>_half_sbs.png
<sample_name>/<sample_name>_occlusion_mask.png
<sample_name>/<sample_name>_depth.png
```

报告主要字段：

- `saved_settings_subset`：本次读取的关键保存参数
- `runtime_config`：映射到立体 runtime 后实际使用的配置
- `assessment.verdict`：`pass` 或 `needs_tuning`
- `assessment.failures`：超过阈值的指标
- `assessment.recommendations`：推荐调参方向
- `assessment.config_notes`：仅基于当前参数值给出的提示
- `assessment.aggregate`：所有样本的聚合指标
- `rows`：每张样本图的完整指标与输出路径

## 判断标准

脚本默认使用以下阈值判断当前参数是否合理：

```text
max_occlusion_mask_ratio: 0.085
max_occlusion_edge_overlap: 0.20
max_edge_source_mae: 0.070
max_edge_gradient_delta: 0.065
max_ghost_risk_score: 8.0
max_hole_risk_score: 10.0
min_stereo_score: 2.0
max_temporal_reset_rate: 0.15
```

这些阈值是视觉回归用的保守门槛。数值通过不代表肉眼一定最佳，但数值失败通常表示需要检查 half-SBS 和 contact sheet。

## 如何阅读结果

先看：

```text
highest_risk_contact_sheet.png
```

它会展示综合风险最高的样本输出，适合快速检查边缘撕裂、虚影、补洞痕迹和过强视差。

再看报告中的：

```text
assessment.verdict
assessment.failures
assessment.recommendations
assessment.aggregate.worst_ghost_sample
assessment.aggregate.worst_hole_sample
```

如果 `verdict` 是 `pass`，说明当前保存参数在样本集上没有触发默认风险阈值。如果是 `needs_tuning`，优先按 `recommendations` 中列出的参数顺序调试。

## 调参方向

### 遮挡面积过大

触发指标：

```text
occlusion_mask_ratio
```

优先调：

```text
Stereo Scale / Max Shift Ratio / Depth Strength
```

建议先降低 `Max Shift Ratio` 或 `Stereo Scale`。如果只在高深度强度下出现，再降低 `Depth Strength`。

### 遮挡集中在边缘

触发指标：

```text
occlusion_edge_overlap
```

优先调：

```text
Edge Dilation / Mask Feather Radius / Edge Threshold
```

建议先增加 `Edge Dilation` 到 `3` 或 `Mask Feather Radius` 到 `4`。如果遮罩变得过宽，再提高 `Edge Threshold`。

### 边缘撕裂或轮廓异常

触发指标：

```text
edge_source_mae
edge_gradient_delta
```

优先调：

```text
Depth Antialias Strength / Edge Dilation / Foreground Scale
```

建议提高 `Depth Antialias Strength`。如果前景轮廓虚影明显，降低 `Foreground Scale` 或增加 `Edge Dilation`。

### 虚影风险高

触发指标：

```text
ghost_risk_score
```

优先调：

```text
Hole Fill Mode / Hole Fill Strength / Stereo Scale
```

建议优先把 `Hole Fill Mode` 调成 `soft_low_ghost`。如果仍偏高，再降低 `Hole Fill Strength` 或 `Stereo Scale`。

### 补洞风险高

触发指标：

```text
hole_risk_score
```

优先调：

```text
Hole Fill Radius / Mask Feather Radius / Max Shift Ratio
```

建议适度提高 `Hole Fill Radius` 或 `Mask Feather Radius`。如果画面变糊，回退半径并降低 `Max Shift Ratio`。

### 立体感不足但伪影可控

触发条件：

```text
stereo_score 低于阈值，且其它风险指标未失败
```

优先调：

```text
Stereo Scale / Max Shift Ratio / Convergence
```

建议小幅提高 `Stereo Scale` 或 `Max Shift Ratio`。如果舒适度不足，再把 `Convergence` 调到 `0.15-0.25` 区间试验。

### 时序重置过频繁

触发指标：

```text
temporal_reset_rate
```

优先调：

```text
Scene Reset Threshold / Temporal Strength / Reset Cooldown Frames
```

如果真实视频中频繁重置，提高 `Scene Reset Threshold` 或 `Reset Cooldown Frames`。如果拖影明显，降低 `Temporal Strength`。

## 信任测试结果前的检查项

在使用报告做参数决策前，确认：

1. `depth_source_request` 是 `production`。
2. 每个样本的 `depth_source` 是 `production_tensorrt_native`。
3. `runtime_config` 中的参数和 `settings.yaml` 当前保存值一致。
4. `samples/` 中没有混入不应该参与基线判断的私有截图或临时图。
5. 已检查 `highest_risk_contact_sheet.png` 和失败样本的 half-SBS 输出。

## 局限性

脚本使用单张样本图生成短平移序列，可以覆盖 temporal 代码路径，但不能完全替代真实视频帧序列。

报告分数适合作为回归和调参方向参考。最终是否采用当前参数，仍需要结合 half-SBS 输出和 contact sheet 做肉眼确认。
