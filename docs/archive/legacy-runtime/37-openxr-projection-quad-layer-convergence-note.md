# OpenXR Projection / Quad Layer 架构收敛记录

## 1. 背景

本记录用于修正 `docs/35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md` 与 `docs/36-OpenXR_Asynchronous_Decoupled_Rendering_Implementation_Plan.md` 中对 Quad Layer 承载范围过宽的设计假设。

近期 D3D11/OpenXR 重构中尝试让 Quad Layer 接管虚拟键盘、FPS/OSD、操作指南等 2D 面板，并通过额外 foreground Projection Layer 解决 Quad Layer 遮挡手柄、激光、光圈的问题。实测与代码分析表明，该方向不适合作为常驻架构：

- OpenXR 普通 Quad Layer 不与主 Projection Layer 共享 depth buffer。
- Quad Layer 的遮挡关系主要由 `xrEndFrame` 提交顺序决定。
- Quad Layer 放在 Projection 后提交时，会覆盖主 Projection 内的手柄、激光、光圈。
- 为了把手柄重新盖到 Quad 上而新增 foreground Projection Layer，会每帧额外渲染左右眼 full-resolution swapchain，并增加一个 composition layer。
- 该 foreground layer 成本足以解释 XR loop / 输入帧率从约 55fps 降到约 30fps 的回退。

结论：**foreground Projection Layer 不应作为常驻方案；手柄、激光、光圈必须回主 Projection。键盘、操作指南、FPS/OSD 可继续用 Quad Layer，但必须控制位置，避免遮挡手柄模型和激光主体。**

## 2. 最新层职责

### 2.1 主 Projection Layer

主 Projection Layer 是 OpenXR 硬实时主路径，必须每帧提交，并承担所有需要 3D 空间关系或深度遮挡的内容：

- 虚拟显示屏 3D 平面。
- 房间背景 / panorama / sky sphere / equirect fallback。
- 手柄模型。
- 激光射线。
- 屏幕与键盘命中光圈。
- 屏幕边框。
- 环境 Glow、四面幕墙光斑、墙面反射的最终合成。

手柄、激光、光圈与虚拟屏幕共享同一个 Projection depth 空间，避免额外 foreground layer 带来的双眼 full-resolution 重绘成本。

### 2.2 Quad Layer

Quad Layer 只用于不要求参与主 Projection 深度遮挡的轻量 2D 覆盖内容：

- 临时 HUD。
- 纯 2D 信息层。
- 不需要被手柄遮挡的低频提示。
- 交互虚拟键盘、操作指南、FPS/OSD、热参数提示等面板，前提是位置足够远，不遮挡手柄模型和激光主体。
- 调试用短时覆盖层。
- 未来可选的非交互覆盖内容。

Quad Layer 默认应隐藏，只在操作或调试需要时显示。常驻 Quad 不应遮挡手柄、激光或主虚拟屏幕交互。

### 2.3 Glow 的归属修正

Glow 的计算可以异步，但环境类 Glow 不应作为最前层 Quad 直接覆盖画面。

正确归属：

```text
Glow / 墙面反射计算      -> 异步生成 safe texture
Glow / 墙面反射显示      -> 主 Projection 采样 safe texture 合成
纯 HUD 式 glow 提示       -> 可选 Quad Layer，低 alpha / additive，且不遮挡交互主体
```

如果 Glow 表示房间、幕墙、屏幕周边的环境效果，就属于 3D 场景，应由主 Projection 中的几何或 shader 采样异步结果完成显示。

## 3. 需要还原或停用的尝试

以下尝试应从常驻路径撤回：

1. **撤回 foreground Projection Layer 常驻方案**
   - 不再每帧创建/更新额外 foreground projection swapchain。
   - 不再通过 foreground layer 补画手柄、激光、光圈。
   - 不再在 `xrEndFrame` 常规提交中额外增加 foreground projection composition layer。

2. **收窄 Quad 接管范围**
   - 虚拟键盘、操作指南、FPS/OSD、热参数提示可继续在 Quad Layer。
   - 必须通过距离和位置保证它们不遮挡手柄模型与激光主体。
   - Glow、墙面反射、幕墙环境光效不得作为常驻最前 Quad。

3. **撤回主 Projection 默认跳过手柄的逻辑**
   - 手柄、激光、光圈必须在主 Projection 中稳定绘制。
   - 不应因为 `_overlay_quads_handle_2d_panels=True` 就跳过主 Projection controller/laser 渲染。

4. **避免为了遮挡问题继续给 foreground layer 打补丁**
   - foreground layer 的问题是架构成本，不是局部 bug。
   - 继续优化 foreground draw call、降低局部 shader 成本，不能解决 full-resolution 双眼额外 layer 的根本开销。

## 4. 可以保留的成果

前期尝试不是全部废弃，下列成果仍可保留并复用：

- D3D11 Projection 主屏路径已经打通。
- OpenXR 输入、射线命中、屏幕调节、热参数状态可继续共用。
- Quad overlay 的 RGBA 纹理生成函数可以继续作为 Quad 面板来源，必要时也可被主 Projection 面板复用。
- 中文字体、操作指南、FPS/OSD、键盘图像生成逻辑可以保留，但应异步或低频生成。
- 键盘 hover / held 状态、光圈参数、D3D11 controller/laser 渲染补齐逻辑可继续使用。
- OpenXR layer 深度边界已经确认，可作为后续设计约束。

## 5. 后续实现方向

### 5.0 当前实际实现方式

当前代码已按混合方案收敛：

- `OpenXRFrameRenderer` 不再调用 foreground Projection Layer。
- `ProjectionLayerPresenter` 不再维护 foreground swapchain / foreground composition layer。
- 主 Projection 每帧绘制虚拟屏幕、手柄模型、激光、光圈。
- D3D11 主 eye render 内直接绘制 laser hit circle，不再依赖 foreground layer。
- Quad Layer 继续承载虚拟键盘、操作指南、FPS/OSD、热参数提示等 2D 面板。
- Quad 面板必须通过位置和距离控制，避免遮挡手柄模型与激光主体。
- Glow / 墙面反射 / 幕墙光效后续回主 Projection，以异步 safe texture 形式被采样合成。

### 5.1 主 Projection 内部轻量合成

主 Projection 每帧只做最终空间绘制：

```text
xrWaitFrame / locate_views / input poll
获取最新或复用的虚拟屏幕纹理
采样 latest safe glow / light texture
绘制虚拟屏幕 3D 平面
绘制手柄 / 激光 / 光圈
xrEndFrame
```

键盘、FPS/OSD、操作指南当前继续由 Quad Layer 显示。主 Projection 后续只需要采样异步完成的 Glow / light texture，不在 XR loop 中执行字体 rasterize、PIL 画图、HDR 解码、mipmap 生成、复杂 blur、模型解析或大数组构建。

### 5.2 可异步移出的重活

以下工作应从 XR loop 移出：

- 键盘贴图生成。
- 操作指南贴图生成。
- FPS / OSD 文本贴图生成。
- 中文字体 rasterize。
- Glow、光斑、墙面反射、亮度提取、降采样、模糊。
- HDR / panorama 解码、tone mapping、cubemap/equirect 转换、mipmap。
- 手柄模型 glTF 解析、mesh 上传、材质贴图加载、VAO / Buffer / SRV 创建。
- 键盘 layout、hit rect、文字位置预计算。
- 屏幕、键盘、面板几何顶点重建。
- FPSBreakdown 长字符串拼接与日志输出。
- TensorRT build、warmup、shape cache、CUDA graph capture。

这些任务完成后只发布 `latest_ready_*` 结果。主 Projection 没拿到新结果时继续复用旧 safe 结果，不等待。

### 5.3 必须留在 XR loop 的工作

以下工作不能异步移出：

- `xrWaitFrame` / `xrBeginFrame` / `xrEndFrame`。
- OpenXR view locate 与 controller input poll。
- 最新头姿下的 view/projection matrix。
- 虚拟屏幕最终绘制到 eye texture。
- 手柄、激光、光圈的最终空间绘制。
- 当前帧 layer 提交。

判断标准：**改变这一帧空间位置的，留在主 XR loop；只改变纹理内容或资源准备的，移出主 XR loop。**

## 6. 性能判断

当前性能判断：

- Foreground layer：每帧额外左右眼 full-resolution swapchain acquire/wait/render/release，并增加 composition layer，是应撤回的主要掉帧来源。
- 主 Projection 手柄/激光/光圈：跟随最新姿态，每帧必须绘制，成本可控。
- Quad 面板：继续承载键盘、操作指南、FPS/OSD，改动最小；只要位置足够远，不遮挡手柄模型与激光主体，就可以保留。
- Glow：不能作为常驻最前 Quad 覆盖层；后续应异步生成纹理，由主 Projection 采样合成。

因此为了恢复 XR loop 稳定性，应优先删除/停用 foreground layer，并让手柄、激光、光圈回主 Projection；交互面板暂不强制回 Projection。

## 7. 当前实施状态

1. 已停用 foreground Projection Layer 常驻提交。
2. 已恢复主 Projection 绘制手柄、激光、光圈。
3. 虚拟键盘、操作指南、FPS/OSD 当前继续保留 Quad Layer；后续只调整位置，避免遮挡手柄和激光主体。
4. Glow / 墙面反射未来回主 Projection 采样异步 safe texture 合成。
5. 面板内容生成、Glow、HDR/panorama 处理后续继续迁移到异步 safe texture 模型。
6. 后续更新 35/36 文档中“Glow / UI / 虚拟键盘 -> Quad overlay”的过宽描述，改为以本记录的职责边界为准。
