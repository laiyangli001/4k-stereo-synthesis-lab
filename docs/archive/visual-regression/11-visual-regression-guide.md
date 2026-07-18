# Visual Regression Guide

本文档说明 `outputs/visual_regression/...` 目录中各类图片的含义、如何判断图片是否有效，以及 `Base Native TensorRT + quality_4k + 2 layers` 的具体输出处理流程。

## 视觉回归目录的作用

视觉回归目录本质上是一个固定输入、固定深度、固定参数的对照包，用来回答三件事：

1. 深度有没有变。
2. 左右眼生成有没有变。
3. `fast baseline` 和 `quality_4k` 的差异在哪里。

## 图片含义

### `input_rgb.png`

原始 4K 输入图。后面所有结果都从这张图和同一份 depth 生成。

### `used_depth.png`

实际用于 stereo synthesis 的深度图。白/亮通常代表一个深度方向，黑/暗代表另一个方向。

重点不是绝对物理距离，而是相对层次是否稳定。如果这张图变了，后面左右眼变化就不能只归因于合成算法。

### `baseline_left.png` / `baseline_right.png`

`fast` baseline 生成的左右眼。它类似 Desktop2Stereo 的基础 depth-shift：根据深度把像素水平偏移，得到左右视差。

### `baseline_half_sbs.png`

baseline 的 Half-SBS 输出。左右眼各压到半宽，再拼成一张 3840x2160 图。适合常见 3D 播放链路。

### `baseline_full_sbs.png`

baseline 的 Full-SBS 输出。左右眼保持完整 3840x2160 宽度，拼成 7680x2160。保留更多横向细节，但带宽更高。

### `quality_4k_left.png` / `quality_4k_right.png`

`quality_4k` 生成的左右眼。它不是单次 depth-shift，而是 2-layer occlusion-aware synthesis：按深度分成两层分别 warp，再 composite，并做遮挡区域 hole fill。

### `quality_4k_half_sbs.png` / `quality_4k_full_sbs.png`

`quality_4k` 的 Half-SBS / Full-SBS 最终输出。

### `quality_4k_occlusion_mask.png`

遮挡/边缘区域 mask。白色区域表示算法认为这里更可能出现前后景错位、拉伸、空洞，需要 hole fill 或特殊处理。

这张图非常重要，因为它告诉你 quality 路线在哪里介入修补。

### `quality_4k_shift_px.png` / `baseline_shift_px.png`

视差偏移量图。它展示每个像素大概会被水平移动多少。

如果 shift 图异常，比如大片纯黑/纯白、断裂、噪声很多，说明 depth 或 shift 参数有问题。

### `baseline_vs_quality_4k_*_absdiff.png`

baseline 和 `quality_4k` 的绝对差异图。越亮表示差异越大。

这不是错误图，而是改进/变化发生在哪里的图。通常希望差异集中在深度边缘、遮挡边界、前后景交界处，而不是整张图大面积随机噪声。

### `contact_sheet.png`

把多张关键图拼在一起，便于快速肉眼检查。

### `contact_sheet_labeled.png`

带标签的拼图版，最适合快速审阅或发给别人看。

### `visual_regression_report.json`

机器可读报告，包含输入形状、depth backend、参数、耗时、PSNR/MAE/MSE 等指标。

## 如何判断图片是否有效

### 1. 检查 `input_rgb.png`

必须是正确的 4K 图，没有旋转、裁切、黑屏、颜色异常。

### 2. 检查 `used_depth.png`

应该能看出大体空间层次。不能是全黑、全白、棋盘噪声或明显破碎。

物体边界附近可以有深度变化，但不应该整张图乱闪。

### 3. 检查 `quality_4k_left.png` / `quality_4k_right.png`

左右眼应该看起来都像原图，只是有轻微水平视差。不能有大面积撕裂、空洞、重复纹理、彩色噪声、黑边。

### 4. 检查 `quality_4k_occlusion_mask.png`

白色区域应该主要出现在前后景边缘、深度突变边界。

如果整张图大面积发白，说明 mask 过度；如果几乎全黑，可能遮挡检测没有起作用。

### 5. 检查 `baseline_vs_quality_4k_*_absdiff.png`

合理情况是差异集中在边缘、遮挡、前后景交界处。

如果整张图均匀发亮，说明 `quality_4k` 改动过大，可能不是局部优化，而是整体图像发生了偏移或亮度变化。

### 6. 检查 `half_sbs` / `full_sbs`

Half-SBS 应该还是 3840x2160，左右眼各占半宽。

Full-SBS 应该是 7680x2160，左右眼完整并排。

拼接中线不能错位或留黑条。

## `Base Native TensorRT + quality_4k + 2 layers` 做了什么

该配置分为两部分：深度估计和 4K stereo synthesis。

## 深度估计

- 使用 `Distill-Any-Depth-Base @ 518`。
- 输入按固定规则缩放到 `294x518`。
- 使用 Native TensorRT fp16 engine 在 GPU 上推理。
- 输出 depth 后归一化，并匹配回 4K 分辨率。

## 4K Stereo Synthesis

- 输入：原始 4K RGB + 4K depth。
- 根据 depth 计算水平视差 `shift_px`。
- 把 depth 分成 2 个 soft layers。
- 对每一层分别生成左眼 warp 和右眼 warp。
- 用 layer weights 把两层 composite 成最终左右眼。
- 根据 depth/shift 边缘生成 occlusion mask。
- 对遮挡/空洞区域做 edge-aware hole fill。
- 最后输出 Half-SBS 或 Full-SBS。

## Fast Baseline 与 Quality 4K 的区别

`fast baseline` 是按深度直接平移像素。

`quality_4k + 2 layers` 是按深度分层平移，再合成，并修补遮挡边缘。

它的目标不是让画面变化很大，而是让前后景边缘、遮挡区域、左右眼一致性更稳，减少简单 depth-shift 常见的拉伸、破洞和边界伪影。
