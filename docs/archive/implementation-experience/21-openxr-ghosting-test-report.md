# OpenXR 边缘虚影测试报告

日期：2026-06-24

## 1. 背景

今天主要排查 OpenXR 模式下人物、物体边缘出现虚影、重影、拖影的问题。现象主要出现在两类链路中：

1. OpenXR lowest / fast 模式：用户反馈同样参数对齐 Beta 后仍能看到边缘虚影，降低 IPD 后虚影明显减小。
2. 较高质量模式：`quality_4k`、`hq_4k` 在物体边缘仍存在不同程度虚影，尤其 Stereo Scale 增大时更明显。

本报告记录今天确认的发生原因、已做优化、测试方法、当前结论和后续优化方向。

## 2. 现象归类

### 2.1 OpenXR lowest / fast 模式虚影

OpenXR lowest 的目标是对齐 Beta 旧架构：

- Runtime 输出 `rgb + depth`。
- OpenXRViewer 在 viewer shader 中按每眼做 DIBR。
- shader 使用 `u_eye_offset`、`u_depth_strength`、`u_convergence` 计算左右眼采样偏移。

今天定位到一个关键问题：GUI 的 `Stereo Scale` 曾经乘进 OpenXR 的 `u_eye_offset`，导致实际视差倍率和预期不一致。用户降低 IPD 时虚影减小，也反向证明虚影主要由实际眼间偏移过大触发。

### 2.2 较高质量模式边缘虚影

`quality_4k` / `hq_4k` 的边缘虚影属于 DIBR 类算法的结构性问题：

- 左右眼视角需要从单张 RGB + depth 合成。
- 物体边缘发生视差偏移后，会暴露原图中不存在的遮挡区域。
- 这些区域只能由 hole fill、edge mask、temporal 或后续 inpaint 猜测补齐。
- `Stereo Scale` 越大，采样偏移越大，遮挡空洞越宽，边缘虚影越明显。

因此，较高质量模式下的边缘虚影不是单个参数错误，而是单目转双目 DIBR 的固有难点。

## 3. 已确认原因

### 3.1 Stereo Scale 对边缘误差影响很大

已确认 `Stereo Scale` 会影响最终视差强度。它不是画质增强参数，而是最终立体分离倍率。

测试中，随着 `Stereo Scale` 从 0.5 增大到 1.0，边缘误差明显升高。例如此前 4K 样本中：

```text
quality_4k scale=0.5: edge 0.050767 -> 0.048349，提升 4.76%
quality_4k scale=0.7: edge 0.056570 -> 0.054628，提升 3.43%
quality_4k scale=1.0: edge 0.065796 -> 0.064352，提升 2.20%

hq_4k scale=0.5: edge 0.050185 -> 0.047744，提升 4.86%
hq_4k scale=0.7: edge 0.055905 -> 0.053879，提升 3.62%
hq_4k scale=1.0: edge 0.064687 -> 0.063209，提升 2.28%
```

这说明：

- Stereo Scale 越高，边缘区域越难补。
- mask feather 能缓解硬边，但不能抵消大视差带来的遮挡缺失。
- 如果追求少虚影，默认 Stereo Scale 不宜过高。

### 3.2 OpenXR lowest 的关键风险点是 u_eye_offset 映射

OpenXR lowest 与本地 fast 的差异不应该来自 temporal 或高级参数，因为 fast 后端不使用这些高级项。真正需要严格对齐的是：

- IPD / Stereo Scale 到 shader `u_eye_offset` 的映射。
- depth strength / convergence 的符号和倍率。
- shader resolution 采用 source texture 还是 swapchain size。
- RGB + depth 纹理上传后是否和 Beta 旧路径一致。

今天通过视觉回归确认：`Stereo Scale` 乘进 `u_eye_offset` 时，会明显改变边缘位移。后续 OpenXR lowest 必须继续保持和 Beta 的有效位移公式一致。

### 3.3 mask_feather_radius 的作用是减轻硬边，不是恢复真实细节

新增 `mask_feather_radius` 后，hole fill 前会对遮挡 mask 做羽化，让补洞区域从硬边变成软过渡。

它可以降低边缘硬切和局部重影感，但副作用是：

- 边缘会更软。
- 半透明过渡区域可能看起来像轻微 halo。
- 半径越大，边缘越不锐。

因此它是折中参数，不是根治方案。

## 4. 已做优化

### 4.1 OpenXR lowest 对齐 Beta 架构

已将 OpenXR lowest 重点回到 Beta 旧模式：

- Runtime 输出 `openxr_rgb_depth`。
- OpenXRViewer shader 执行 DIBR。
- 避免 lowest 走不必要的高质量合成路径。
- 视觉回归中检查 `scaled`、`beta_direct`、`source resolution`、`swapchain resolution` 等变体。

### 4.2 fast 模式禁用无效高级参数

针对 fast 模式，已避免 temporal、foreground、antialias 等高级项造成误解或副作用：

- fast 下 temporal 置为 false。
- temporal strength / scene reset / reset cooldown 归零。
- foreground scale 和 antialias 对 fast 不参与实际后端。

这解决了本地模式下用户看到的拖影问题。

### 4.3 引入 mask_feather_radius

在 `edge_aware_fill()` 中增加 `mask_feather_radius`：

- 默认值：3。
- GUI 已暴露为高级立体参数 `Mask Feather`。
- 设置项：`Mask Feather Radius`。
- hot reload 已接入，运行时日志输出 `mask_feather=...`。

当前 GUI 范围为 `0-5`。

### 4.4 多图视觉回归验证

使用 `scripts/tools/stereo_scale_sweep.py` 扫描多张图片：

```text
samples/4K.jpg
samples/car.jpg
samples/bird.jpg
samples/gui.png
samples/desktop.png
samples/fast_plus/tree.jpg
```

测试组合：

```text
backend: quality_4k, hq_4k
Stereo Scale: 0.5, 0.7, 1.0
mask_feather_radius: 0, 1, 2, 3, 4, 5
max_width: 960
```

输出目录：

```text
outputs/visual_regression/mask_feather_multi_*
```

补充测试又选 3 张图扫到 10：

```text
samples/4K.jpg
samples/gui.png
samples/fast_plus/tree.jpg

mask_feather_radius: 0-10
```

输出目录：

```text
outputs/visual_regression/mask_feather_wide_*
```

## 5. 测试方法

### 5.1 视觉回归脚本

使用命令示例：

```powershell
src\python3\python.exe scripts\tools\stereo_scale_sweep.py \
  --rgb samples\4K.jpg \
  --out outputs\visual_regression\mask_feather_multi_4k \
  --backends quality_4k,hq_4k \
  --scales 0.5,0.7,1.0 \
  --mask-feathers 0,1,2,3,4,5 \
  --max-width 960
```

脚本执行内容：

1. 读取 RGB。
2. 根据 RGB 生成 proxy depth。
3. 使用 Sobel + max pool 生成边缘 mask。
4. 对不同 backend、Stereo Scale、mask feather 组合合成左右眼。
5. 统计：
   - `edge_mean_to_source_mae`
   - `mean_left_right_mae`
   - `mean_left_to_source_mae`
   - `mean_right_to_source_mae`
   - `occlusion_mask_ratio`
6. 保存 `metrics.json` 和对应 SBS 输出图。

### 5.2 指标解释

本次主要看 `edge_mean_to_source_mae`：

- 数值越低，表示边缘区域相对源图变化越小。
- 对硬边虚影有参考价值。
- 但它会偏向更强羽化，因为更软的边缘通常会降低 MAE。

因此，该指标不能单独决定默认值。默认值还需要考虑主观清晰度和边缘锐度。

## 6. 测试结果

### 6.1 多图 0-5 扫描结果

在 6 张图、2 个 backend、3 个 Stereo Scale 的 36 个组合中：

```text
total = 36
mask_feather_radius=5 赢了 36/36
mask_feather_radius=3 相比 0 平均改善约 4.8%
mask_feather_radius=5 相比 3 额外改善约 1.2%
```

按当前边缘 MAE 指标，`5` 比 `3` 更好。

### 6.2 0-10 补扫结果

补扫发现：很多组合从 5 继续增大到 10，边缘 MAE 仍继续下降，少数组合在 6 附近达到最低。

这说明当前单一指标有明显倾向：

- feather 越大，边缘越软。
- 边缘越软，MAE 可能越低。
- 但视觉上可能更糊，不能直接把最大 feather 设为默认。

### 6.3 当前默认值判断

`mask_feather_radius=3` 不是边缘 MAE 的数学最优值，但仍是更合理的默认折中：

- 相比 0 已经明显降低硬边和虚影。
- 相比 5 保留更多边缘锐度。
- GUI 可手动调到 4 或 5，给重影敏感场景使用。

如果产品策略更偏向“宁愿边缘软，也不要重影”，可以将默认改为 5。否则建议继续默认 3。

## 7. 当前结论

1. OpenXR lowest 的虚影主要来自有效眼间位移公式、shader 参数和 Beta 对齐问题，而不是 temporal 这类高级参数。
2. 较高质量模式的边缘虚影是 DIBR 从单张图合成双眼时的结构性副作用。
3. Stereo Scale 是必要参数，但应该作为高级舒适度/立体强度控制项，不应该频繁作为画质旋钮。
4. `mask_feather_radius=3` 是可接受默认值，不是绝对最优值。
5. `mask_feather_radius=5` 在当前边缘 MAE 指标上更低，但可能带来边缘变软。
6. 未来出现物体边缘虚影时，不能简单认为是 bug，需要判断是参数过强、深度错误、遮挡补洞失败，还是 OpenXR shader 映射偏差。

## 8. 副作用和可接受范围

### 8.1 可以接受的副作用

在当前 DIBR 架构下，下列现象属于可接受范围：

- 高视差边缘出现轻微软边。
- 物体边缘有少量补洞痕迹。
- Stereo Scale 增大后边缘误差变大。
- 深度不连续区域出现局部虚影。

### 8.2 不应接受的异常

下列情况应继续当作 bug 排查：

- lowest 模式明显比 Beta 旧版更重影。
- 降低 IPD / Stereo Scale 后虚影仍完全不变。
- 本地 fast 和 OpenXR fast 同参数下位移方向或倍率不一致。
- temporal 关闭后仍出现跨帧拖影。
- 左右眼某一只眼明显偏移错误。
- GUI 显示值和 runtime log 中实际参数不一致。

## 9. 后续优化方向

### 9.1 edge mask 优化

当前 edge mask 主要依赖深度梯度。后续可以改进：

- RGB 梯度 + depth 梯度联合判断。
- 对人物、UI 字体、小物体边缘使用不同阈值。
- 将 edge mask 分成 foreground edge 和 background edge。
- 对大面积平滑区域减少不必要的 feather。

目标是让羽化只作用在真正需要补洞的遮挡边缘，而不是全局抹软。

### 9.2 hole fill 优化

当前 hole fill 仍偏传统局部填充。后续方向：

- 按视差方向做定向填补，而不是纯邻域扩散。
- 前景边缘优先从背景侧采样，减少前景拖影。
- UI/文字区域使用更保守的补洞策略。
- 引入低成本 content-aware fill 或轻量 inpaint。

### 9.3 temporal 优化

之前本地模式拖影说明 temporal 必须更谨慎：

- fast 模式默认关闭 temporal。
- temporal 应优先稳定 depth，而不是直接混合最终 RGB。
- scene reset 需要对 UI、视频切镜、快速运动更敏感。
- 对边缘区域使用更低 temporal 权重。

### 9.4 Stereo Scale 策略

建议保留 Stereo Scale，但定位为高级参数：

- 默认继续使用 0.5。
- 普通用户主要调 Depth Strength。
- 高质量视觉回归固定测试 0.5、0.7、1.0 三档。
- 出现虚影时优先检查 Stereo Scale 是否过高。

### 9.5 更完整的视觉回归指标

当前 `edge_mean_to_source_mae` 不足以判断主观最优。后续需要新增：

- 边缘锐度损失指标。
- halo / oversoften 检测。
- 左右眼一致性指标。
- UI 文字区域专项指标。
- 人物边缘专项样本集。
- 与 Beta 输出逐像素对齐的 OpenXR lowest 专项回归。

## 10. 推荐默认配置

当前建议：

```yaml
Stereo Scale: 0.5
Mask Feather Radius: 3
Edge Dilation: 2
Edge Threshold: 0.04
Temporal Strength: fast 模式为 0，高质量模式按预设
```

可选策略：

- 如果更重视边缘锐度：`Mask Feather Radius = 2 或 3`。
- 如果更重视减少虚影：`Mask Feather Radius = 4 或 5`。
- 如果 Stereo Scale 调到 0.7 或 1.0，建议同步提高 mask feather，或降低 depth strength。

## 11. 最终判断

今天的虚影问题可以分成两类处理：

1. 参数映射错误或链路不一致导致的虚影：必须修复，例如 OpenXR lowest 的 `u_eye_offset` 对齐问题。
2. DIBR 遮挡区域无法真实恢复导致的虚影：只能通过 mask、hole fill、temporal 和视差策略缓解。

目前 `mask_feather_radius=3` 是偏锐度和稳定性的折中默认，不是指标上的最小值。如果后续用户继续反馈边缘重影，可先让用户在 GUI 中将 `Mask Feather` 调到 4 或 5 做快速验证；若仍明显，则进入 edge mask / hole fill / temporal 的专项优化。